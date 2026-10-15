"""Jackhmmer orchestration layer (pyhmmer backend) -- the rewrite of run_jack.py.

Replaces the old subprocess-CLI + never-wired F1-F4 JSON version. Now the
algorithm-under-test is an AssembledConfig of *components* (inclusion strategy,
convergence rule, filter cascade, scoring/weighting/eff-seq via Builder, and
optional C-kernel patches). For each query sequence this module:

1. Builds a hand-architecture plan7.Builder from the assembled builder kwargs.
2. Calls pyhmmer.hmmer.jackhmmer with the assembled pipeline kwargs (F1/F2/F3,
   incE/incT/incdomE/incdomT, bias_filter) + select_hits callback.
3. Takes the selected HMM checkpoint and runs pyhmmer.hmmer.hmmsearch against
   the ASTRAL db.
4. Scores TP/FP against the SCOP ground-truth class set of the query (loaded
   from dir.cla.scope -- NOT derived from the query's own hits, fixing the old
   self-referential TP bug), and returns a unified EvalResult.

C-kernel patches: if the AssembledConfig carries C patches, this module does
NOT apply them itself -- that requires recompilation and real binary execution
(see run_hmmer_cli.py). run_jack.py is the pure pyhmmer (Python-space) path.
"""

from __future__ import annotations

import csv
import glob
import os
import statistics
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .eval_schema import EvalResult, UnitTestResult

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[1]

# --- paths (env-overridable; repo ships no big data files) ------------------
ASTRAL_DB = os.getenv("ASTRAL_DB", str(ROOT / "data/scop_benchmark/astral-scopedom-seqres-gd-sel-gs-bib-40-2.08.fa"))
UNIREF_DB = os.getenv("UNIREF_DB", str(ROOT / "data/uniref90.fasta"))
SCOP_TXT = os.getenv("SCOP_TXT", str(ROOT / "data/scop_benchmark/dir.cla.scope.2.08-stable.txt"))

DIR_DATA_TEST = SCRIPT_DIR / "data_test"
TIMING_FILE = SCRIPT_DIR / "timing_records.csv"
DIR_RES = SCRIPT_DIR / "res_results"
DEFAULT_PYHMMER_MAX_DB_BYTES = 2 * 1024 * 1024 * 1024


def _load_scop_lineages(scop_txt: str | Path | None = None) -> dict[str, str]:
    """Load domain_id -> complete SCOP SCCS from dir.cla.scope.

    An SCCS such as ``a.1.1.1`` encodes class, fold, superfamily, and family.
    Keeping all four levels is required for a valid remote-homology benchmark;
    reducing it to the first class letter incorrectly labels unrelated folds as
    true positives.

    dir.cla.scope verbatim layout (whitespace-separated):
        Sid  PDB  Residues  Classification  Sunid  Description
    so the classification id (e.g. "a.1.1.1") is column index 3. We retain the
    complete value and also tolerate legacy/short lines where it appears later.
    """
    m: dict[str, str] = {}
    path = str(scop_txt or SCOP_TXT)
    if not os.path.exists(path):
        return m
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            p = line.split()
            if len(p) < 2:
                continue
            sid = p[0]
            cls = None
            if len(p) >= 4 and _looks_like_scop_id(p[3]):
                cls = p[3]
            else:
                for tok in p[1:]:
                    if _looks_like_scop_id(tok):
                        cls = tok
                        break
            if cls:
                m[sid] = cls
    return m


# Compatibility for callers outside this package. Values are now complete SCCS
# strings, despite the legacy function name.
def _load_scop_classes(scop_txt: str | Path | None = None) -> dict[str, str]:
    return _load_scop_lineages(scop_txt)


def _looks_like_scop_id(tok: str) -> bool:
    """A SCOP classification id looks like 'a.1.1.1' (class.fold.superfam.fam)."""
    if not tok or "." not in tok:
        return False
    head = tok.split(".")[0]
    return len(head) == 1 and head.isalpha()


def _query_scop_lineage(query_id: str, scop: dict[str, str]) -> str | None:
    """Return the query's complete SCCS ground-truth lineage."""
    raw = query_id.split()[0]
    return _lookup_scop(raw, scop)


def _query_scop_class(query_id: str, scop: dict[str, str]) -> str | None:
    """Compatibility alias returning the complete SCCS, not only its class."""
    return _query_scop_lineage(query_id, scop)


@dataclass(frozen=True)
class ScopScore:
    """Remote-homology counts using SCOP superfamily/fold relationships."""

    tp: int = 0
    fp: int = 0
    ignored: int = 0
    unclassified: int = 0

    def meta(self) -> dict[str, Any]:
        return {
            "policy": "same_superfamily_positive_different_fold_negative",
            "tp": self.tp,
            "fp": self.fp,
            "ignored_same_fold_different_superfamily": self.ignored,
            "unclassified": self.unclassified,
        }


def _score_scop_names(
    names: list[str], scop: dict[str, str], query_lineage: str | None
) -> ScopScore:
    """Score names using same-superfamily positives and different-fold negatives.

    Targets in the same fold but a different superfamily are ignored. This
    avoids treating structurally related but non-homologous targets as either
    obvious positives or obvious negatives. Targets without SCOP ground truth
    are also excluded and reported separately.
    """
    if query_lineage is None:
        return ScopScore(unclassified=len(names))
    query_parts = query_lineage.split(".")
    if len(query_parts) < 3:
        return ScopScore(unclassified=len(names))
    query_fold = tuple(query_parts[:2])
    query_superfamily = tuple(query_parts[:3])
    tp = fp = ignored = unclassified = 0
    for name in names:
        lineage = _lookup_scop(name, scop)
        if lineage is None:
            unclassified += 1
            continue
        parts = lineage.split(".")
        if len(parts) < 3:
            unclassified += 1
        elif tuple(parts[:3]) == query_superfamily:
            tp += 1
        elif tuple(parts[:2]) != query_fold:
            fp += 1
        else:
            ignored += 1
    return ScopScore(tp=tp, fp=fp, ignored=ignored, unclassified=unclassified)


def _score_hits(hits, scop: dict[str, str], query_lineage: str | None) -> ScopScore:
    names = [
        h.name.decode() if isinstance(h.name, (bytes, bytearray)) else str(h.name)
        for h in hits
    ]
    return _score_scop_names(names, scop, query_lineage)


def _lookup_scop(identifier: str, scop: dict[str, str]) -> str | None:
    raw = identifier.split()[0]
    candidates = [raw]
    if raw.startswith("d"):
        candidates.append(raw[1:])
    else:
        candidates.append(f"d{raw}")
    for candidate in candidates:
        lineage = scop.get(candidate)
        if lineage:
            return lineage
    return None


def _as_iteration_results(obj: Any) -> list[Any]:
    """Normalize pyhmmer checkpoint output to a list of IterationResult objects."""
    if hasattr(obj, "iteration"):
        return [obj]
    return list(obj)


def _tophits_counts(hits: Any) -> dict[str, int]:
    """Return aggregate pyhmmer hit counters without retaining target names."""
    counts = {
        "total": 0,
        "reported": 0,
        "included": 0,
        "new": 0,
        "dropped": 0,
        "duplicate": 0,
    }
    for hit in hits:
        counts["total"] += 1
        for name in ("reported", "included", "new", "dropped", "duplicate"):
            counts[name] += bool(getattr(hit, name, False))
    for name in (
        "searched_models",
        "searched_nodes",
        "searched_sequences",
        "searched_residues",
    ):
        value = getattr(hits, name, None)
        if value is not None:
            counts[name] = int(value)
    return counts


@contextmanager
def _profile_phase(phases: dict[str, float], name: str):
    """Accumulate one phase without changing the evaluation cost clock."""
    started_at = time.perf_counter()
    try:
        yield
    finally:
        phases[name] = phases.get(name, 0.0) + (time.perf_counter() - started_at)


def _iteration_profile(result: Any) -> dict[str, Any]:
    """Build a privacy-safe summary for one pyhmmer jackhmmer iteration."""
    return {
        "iteration": int(getattr(result, "iteration", 0) or 0),
        "converged": bool(getattr(result, "converged", False)),
        "hits": _tophits_counts(getattr(result, "hits", ())),
    }


def _numeric_stats(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {"count": 0, "min": 0.0, "max": 0.0, "mean": 0.0, "median": 0.0}
    return {
        "count": len(values),
        "min": min(values),
        "max": max(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
    }


def _query_profile(
    *,
    query_id: str,
    status: str,
    cost_s: float,
    iterations: int,
    phase_wall_s: dict[str, float],
    iteration_results: list[dict[str, Any]],
    final_hits: dict[str, int] | None,
) -> dict[str, Any]:
    phases = {name: round(float(value), 6) for name, value in phase_wall_s.items()}
    denominator = float(cost_s) if cost_s > 0 else sum(phases.values())
    fractions = {
        name: round(value / denominator, 6) if denominator else 0.0
        for name, value in phases.items()
    }
    return {
        "schema_version": "pyhmmer_query_profile.v1",
        "query_id": query_id,
        "status": status,
        "cost_s": cost_s,
        "iterations": iterations,
        "nhits": None if final_hits is None else final_hits.get("total", 0),
        "phase_wall_s": phases,
        "phase_wall_fraction": fractions,
        "dominant_phase": max(phases, key=phases.get) if phases else None,
        "iteration_results": iteration_results,
        "final_hits": final_hits,
    }


def _aggregate_profiles(
    profiles: list[dict[str, Any]],
    *,
    stage: str,
    status: str,
    cpus: int,
    max_iter: int,
    query_count: int,
    cost_s: float,
) -> dict[str, Any]:
    """Aggregate per-query pyhmmer timings into bounded optimization evidence."""
    phase_totals: dict[str, float] = {}
    query_costs: list[float] = []
    iteration_counts: list[float] = []
    iteration_hit_counts: dict[str, list[float]] = {}
    for profile in profiles:
        query_costs.append(float(profile.get("cost_s", 0.0) or 0.0))
        iteration_counts.append(float(profile.get("iterations", 0) or 0))
        for phase, value in (profile.get("phase_wall_s") or {}).items():
            phase_totals[str(phase)] = phase_totals.get(str(phase), 0.0) + float(value or 0.0)
        for item in profile.get("iteration_results") or []:
            for name, value in (item.get("hits") or {}).items():
                if isinstance(value, (int, float)):
                    iteration_hit_counts.setdefault(str(name), []).append(float(value))

    phase_totals = {name: round(value, 6) for name, value in phase_totals.items()}
    denominator = float(cost_s) if cost_s > 0 else sum(phase_totals.values())
    phase_fractions = {
        name: round(value / denominator, 6) if denominator else 0.0
        for name, value in phase_totals.items()
    }
    dominant_phase = max(phase_totals, key=phase_totals.get) if phase_totals else None
    algorithm_phases = {
        name: value for name, value in phase_totals.items()
        if name not in {"load_query", "load_targets", "load_astral"}
    }
    dominant_algorithm_phase = (
        max(algorithm_phases, key=algorithm_phases.get) if algorithm_phases else None
    )
    top_slowest = sorted(
        (dict(profile) for profile in profiles),
        key=lambda item: float(item.get("cost_s", 0.0) or 0.0),
        reverse=True,
    )[:10]
    return {
        "schema_version": "pyhmmer_eval_profile.v1",
        "backend": "pyhmmer",
        "stage": stage,
        "status": status,
        "cpus": cpus,
        "max_iter": max_iter,
        "query_count": query_count,
        "profiled_query_count": len(profiles),
        "cost_s": cost_s,
        "phase_wall_s": phase_totals,
        "phase_wall_fraction": phase_fractions,
        "algorithm_phase_wall_s": algorithm_phases,
        "dominant_phase": dominant_phase,
        "dominant_algorithm_phase": dominant_algorithm_phase,
        "query_cost_stats": _numeric_stats(query_costs),
        "iteration_stats": _numeric_stats(iteration_counts),
        "iteration_hit_stats": {
            name: _numeric_stats(values) for name, values in iteration_hit_counts.items()
        },
        "top_slowest_queries": top_slowest,
    }


def _convergence_rule(cfg) -> str | None:
    spec = cfg.specs.get("convergence_rule")
    payload = getattr(spec, "payload", {}) or {}
    return payload.get("rule")


def _fixed_iteration_budget(cfg, max_iter: int) -> int:
    """Map convergence_rule.fixed(n) to a real pyhmmer max_iterations value."""
    spec = cfg.specs.get("convergence_rule")
    payload = getattr(spec, "payload", {}) or {}
    if payload.get("rule") == "fixed":
        try:
            return max(1, int(payload.get("n", max_iter)))
        except (TypeError, ValueError):
            return max_iter
    return max_iter


def _pyhmmer_db_limit(max_target_bytes: int | None) -> int | None:
    if max_target_bytes is not None:
        return max_target_bytes
    raw = os.getenv("PYHMMER_MAX_DB_BYTES")
    if raw:
        try:
            return int(raw)
        except ValueError:
            return DEFAULT_PYHMMER_MAX_DB_BYTES
    return DEFAULT_PYHMMER_MAX_DB_BYTES


def _too_large_for_pyhmmer(path: str, max_target_bytes: int | None) -> tuple[bool, int, int | None]:
    limit = _pyhmmer_db_limit(max_target_bytes)
    try:
        size = os.path.getsize(path)
    except OSError:
        return False, 0, limit
    return bool(limit is not None and size > limit), size, limit


def run_variant(
    query_seq_path: str,
    cfg,
    *,
    cpus: int = 1,
    max_iter: int = 3,
    stage: str = "smoke",
    astral_db: str | Path | None = None,
    uniref_db: str | Path | None = None,
    scop_txt: str | Path | None = None,
    max_target_bytes: int | None = None,
) -> EvalResult:
    """Run one jackhmmer variant for one query sequence, return EvalResult.

    cfg: an AssembledConfig (see assembler.py). C patches must be evaluated by
    the compiled-HMMER backend, not this pyhmmer backend.
    """
    from pyhmmer.easel import Alphabet, SequenceFile, DigitalSequenceBlock
    from pyhmmer.plan7 import Builder, Background
    from pyhmmer.hmmer import jackhmmer

    astral_path = str(astral_db or ASTRAL_DB)
    uniref_path = str(uniref_db or UNIREF_DB)
    scop_path = str(scop_txt or SCOP_TXT)
    meta: dict[str, Any] = {
        "backend": "pyhmmer",
        "astral_db": astral_path,
        "uniref_db": uniref_path,
        "scop_txt": scop_path,
        "max_iter": max_iter,
    }
    errors: list[str] = []
    unit_tests: list[UnitTestResult] = []

    try:
        alphabet = Alphabet.amino()
        with SequenceFile(query_seq_path, digital=True, alphabet=alphabet) as qf:
            queries = list(qf)
    except FileNotFoundError:
        return EvalResult(stage=stage, status="run_error",
                          errors=[f"query seq not found: {query_seq_path}"], meta=meta)
    if not queries:
        return EvalResult(stage=stage, status="run_error",
                          errors=["empty query sequence file"], meta=meta)

    query = queries[0]
    query_id = query.name.decode() if isinstance(query.name, (bytes, bytearray)) else str(query.name)
    meta["query_id"] = query_id

    if not os.path.exists(uniref_path):
        return EvalResult(stage=stage, status="run_error",
                          errors=[f"UNIREF_DB not found: {uniref_path}"], meta=meta)
    too_large, db_size, db_limit = _too_large_for_pyhmmer(uniref_path, max_target_bytes)
    if too_large:
        meta.update({"backend_limit": "pyhmmer_materializes_target_db", "db_size_bytes": db_size,
                     "max_db_bytes": db_limit})
        return EvalResult(stage=stage, status="skipped", n_seqs=1, meta=meta,
                          errors=[
                              f"UNIREF_DB is {db_size} bytes, above pyhmmer materialization limit "
                              f"{db_limit}; use hmmer_cli backend for full-scale runs."
                          ])

    builder_kwargs = dict(cfg.builder_kwargs())
    pipeline_kwargs = dict(cfg.pipeline_kwargs())
    inc = cfg.specs.get("inclusion_strategy")
    if inc:
        pipeline_kwargs.update(inc.pipeline_kwargs)

    t0 = time.time()
    iterations_run = 0
    final_hmm = None
    logical_stop_iter = None
    phase_wall_s: dict[str, float] = {}
    iteration_profile_sources: list[Any] = []

    def attach_profile(status: str, cost_s: float, result_hits: Any | None = None) -> None:
        iteration_results = [_iteration_profile(item) for item in iteration_profile_sources]
        final_hits = _tophits_counts(result_hits) if result_hits is not None else None
        meta["profile"] = _query_profile(
            query_id=query_id,
            status=status,
            cost_s=cost_s,
            iterations=iterations_run,
            phase_wall_s=phase_wall_s,
            iteration_results=iteration_results,
            final_hits=final_hits,
        )

    try:
        with _profile_phase(phase_wall_s, "pipeline_setup"):
            Background(alphabet)
            builder = Builder(alphabet, architecture="hand", **builder_kwargs)
            select_hits = cfg.select_hits()
            select_cb = select_hits() if callable(select_hits) else None

        with _profile_phase(phase_wall_s, "load_targets"):
            with SequenceFile(uniref_path, digital=True, alphabet=alphabet) as tf:
                targets = list(tf)
            target_block = DigitalSequenceBlock(alphabet, targets)

        with _profile_phase(phase_wall_s, "jackhmmer_iterations"):
            should_stop_factory = cfg.should_stop_factory()
            should_stop = should_stop_factory() if callable(should_stop_factory) else None
            opt_iter = _fixed_iteration_budget(cfg, max_iter)
            meta["effective_max_iter"] = opt_iter
            if _convergence_rule(cfg) not in (None, "fixed"):
                meta["convergence_cost_saving"] = False
            results = jackhmmer(
                query, target_block,
                max_iterations=opt_iter,
                checkpoints=True,
                cpus=cpus,
                builder=builder,
                select_hits=select_cb,
                **pipeline_kwargs,
            )

            last = None
            stopped = False
            for outer in results:
                inner_results = _as_iteration_results(outer)
                if inner_results:
                    iterations_run = max(iterations_run, max(int(r.iteration) for r in inner_results))
                for idx, r in enumerate(inner_results):
                    last = r
                    iteration_profile_sources.append(r)
                    if callable(should_stop) and should_stop(idx, r):
                        logical_stop_iter = int(r.iteration)
                        stopped = True
                        break
                if stopped:
                    break
        final_hmm = getattr(last, "hmm", None) if last is not None else None
    except Exception as e:
        cost_s = round(time.time() - t0, 4)
        attach_profile("run_error", cost_s)
        return EvalResult(stage=stage, status="run_error",
                          errors=[f"jackhmmer failed: {type(e).__name__}: {e}"],
                          cost_s=cost_s,
                          n_seqs=1, iterations=iterations_run, meta=meta)

    if logical_stop_iter is not None and logical_stop_iter != iterations_run:
        meta["logical_stop_iter"] = logical_stop_iter
        meta["convergence_cost_saving"] = False

    if final_hmm is None:
        cost_s = round(time.time() - t0, 4)
        attach_profile("run_error", cost_s)
        return EvalResult(stage=stage, status="run_error",
                          errors=["jackhmmer produced no HMM"],
                          cost_s=cost_s,
                          n_seqs=1, iterations=iterations_run, meta=meta)

    try:
        from pyhmmer.plan7 import Pipeline
        from pyhmmer.easel import SequenceFile as SF
        with _profile_phase(phase_wall_s, "load_astral"):
            with SF(astral_path, digital=True, alphabet=alphabet) as af:
                astral = list(af)
            astral_block = DigitalSequenceBlock(alphabet, astral)
        with _profile_phase(phase_wall_s, "final_hmmsearch"):
            pli = Pipeline(alphabet, **{k: v for k, v in pipeline_kwargs.items()
                                        if k in ("F1", "F2", "F3", "bias_filter",
                                                 "E", "T", "domE", "domT")})
            hits = pli.search_hmm(final_hmm, astral_block)
            hits.sort()
    except Exception as e:
        cost_s = round(time.time() - t0, 4)
        attach_profile("run_error", cost_s)
        return EvalResult(stage=stage, status="run_error",
                          errors=[f"hmmsearch failed: {type(e).__name__}: {e}"],
                          cost_s=cost_s,
                          n_seqs=1, iterations=iterations_run, meta=meta)

    cost = round(time.time() - t0, 4)

    scop = _load_scop_lineages(scop_path)
    query_lineage = _query_scop_lineage(query_id, scop)
    if query_lineage is None:
        attach_profile("skipped", cost, hits)
        return EvalResult(stage=stage, status="skipped", cost_s=cost,
                          n_seqs=1, iterations=iterations_run, meta=meta,
                          errors=["no SCOP lineage for query; cannot score TP/FP"])
    score = _score_hits(hits, scop, query_lineage)
    meta["nhits"] = len(hits)
    meta["scop_scoring"] = score.meta()
    attach_profile("ok", cost, hits)
    res = EvalResult(stage=stage, status="ok", tp=score.tp, fp=score.fp, cost_s=cost,
                     n_seqs=1, iterations=iterations_run, unit_tests=unit_tests,
                     errors=errors, meta=meta)
    res.ratio()
    return res


def _query_domain_key(query_id: str) -> str:
    """Normalize a query id (strip leading 'd', extension) for SCOP lookup."""
    k = query_id.split()[0]
    return k[1:] if k.startswith("d") else k


def query_fastas(data_dir: Path | str) -> list[str]:
    """Return benchmark query FASTAs, excluding helper DB/materialized files."""
    ddir = Path(data_dir)
    seqs = sorted(glob.glob(str(ddir / "all_samples.part_*.fasta")))
    if seqs:
        return seqs
    skipped = {"all_samples.fasta", "uniref_smoke.fasta"}
    return sorted(
        str(p) for p in ddir.glob("*.fasta")
        if p.name not in skipped and not p.name.startswith("uniref")
    )


def validate_query_fastas(query_paths: list[str]) -> None:
    """Fail closed unless every benchmark file contains one unique FASTA record."""
    seen: dict[str, str] = {}
    for raw_path in query_paths:
        path = Path(raw_path)
        headers: list[str] = []
        has_sequence = False
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                headers.append(line[1:].strip())
            elif headers:
                has_sequence = True
        if len(headers) != 1:
            raise ValueError(f"query FASTA {path} contains {len(headers)} sequence records; expected exactly 1")
        if not headers[0] or not has_sequence:
            raise ValueError(f"query FASTA {path} has no non-empty sequence")
        display_id = headers[0].split()[0]
        if display_id in seen:
            raise ValueError(
                f"duplicate query display ID {display_id!r} in {seen[display_id]} and {path}"
            )
        seen[display_id] = str(path)


def run_all_queries(
    cfg,
    data_dir: Path | None = None,
    *,
    cpus: int = 1,
    stage: str = "smoke",
    max_iter: int = 3,
    astral_db: str | Path | None = None,
    uniref_db: str | Path | None = None,
    scop_txt: str | Path | None = None,
    max_target_bytes: int | None = None,
) -> EvalResult:
    """Run a variant over all query fastas in data_dir, aggregate one EvalResult."""
    ddir = Path(data_dir) if data_dir else DIR_DATA_TEST
    seqs = query_fastas(ddir)
    meta = {
        "backend": "pyhmmer",
        "data_dir": str(ddir),
        "astral_db": str(astral_db or ASTRAL_DB),
        "uniref_db": str(uniref_db or UNIREF_DB),
        "scop_txt": str(scop_txt or SCOP_TXT),
        "max_iter": max_iter,
    }
    if not seqs:
        return EvalResult(stage=stage, status="run_error",
                         errors=[f"no query fastas under {ddir}"],
                         meta=meta)
    agg_tp = agg_fp = 0
    total_cost = 0.0
    max_iters = 0
    errs: list[str] = []
    n = 0
    statuses: list[str] = []
    query_profiles: list[dict[str, Any]] = []
    scop_counts = {"tp": 0, "fp": 0, "ignored_same_fold_different_superfamily": 0, "unclassified": 0}
    for sp in seqs:
        r = run_variant(
            sp, cfg, cpus=cpus, max_iter=max_iter, stage=stage,
            astral_db=astral_db, uniref_db=uniref_db, scop_txt=scop_txt,
            max_target_bytes=max_target_bytes,
        )
        n += 1
        statuses.append(r.status)
        agg_tp += r.tp
        agg_fp += r.fp
        total_cost += r.cost_s
        max_iters = max(max_iters, r.iterations)
        errs.extend(r.errors)
        profile = r.meta.get("profile") if isinstance(r.meta, dict) else None
        if isinstance(profile, dict):
            stored_profile = dict(profile)
            stored_profile.setdefault("query_idx", n)
            query_profiles.append(stored_profile)
        scoring = r.meta.get("scop_scoring") if isinstance(r.meta, dict) else None
        if isinstance(scoring, dict):
            for key in scop_counts:
                scop_counts[key] += int(scoring.get(key, 0) or 0)
    if all(s == "ok" for s in statuses):
        status = "ok"
    elif all(s == "skipped" for s in statuses):
        status = "skipped"
    else:
        status = "run_error"
    meta["scop_scoring"] = {
        "policy": "same_superfamily_positive_different_fold_negative",
        **scop_counts,
    }
    aggregate_cost = round(total_cost, 4)
    meta["profile"] = _aggregate_profiles(
        query_profiles,
        stage=stage,
        status=status,
        cpus=cpus,
        max_iter=max_iter,
        query_count=n,
        cost_s=aggregate_cost,
    )
    res = EvalResult(
        stage=stage,
        status=status,
        tp=agg_tp, fp=agg_fp, cost_s=aggregate_cost,
        n_seqs=n, iterations=max_iters, errors=errs,
        meta=meta,
    )
    res.ratio()
    return res


def write_timing(seq_time: list[tuple[str, float]]) -> None:
    """Write timing_records.csv with the unified SeqID,Time_s schema."""
    TIMING_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(TIMING_FILE, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["SeqID", "Time_s"])
        for sid, t in seq_time:
            w.writerow([sid, t if t is not None else "FAIL"])


if __name__ == "__main__":
    from .assembler import assemble
    from .components.base import ComponentModification
    mods = [
        ComponentModification(contract_name="inclusion_strategy",
                              python_payload={"rule": "evalue", "incE": 1e-3}),
        ComponentModification(contract_name="filter_cascade",
                              python_payload={"F1": 0.02, "F2": 1e-3, "F3": 1e-5}),
        ComponentModification(contract_name="scoring_system",
                              python_payload={"matrix": "BLOSUM62"}),
    ]
    cfg = assemble(mods)
    if cfg.errors:
        print("assemble errors:", cfg.errors)
    else:
        r = run_all_queries(cfg, stage="smoke")
        print(r.model_dump_json(indent=2))
