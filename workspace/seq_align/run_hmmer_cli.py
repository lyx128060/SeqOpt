"""Compiled-HMMER CLI backend for seq_align C patches.

This backend is used when an assembled variant contains C_PATCH components. It
never mutates the host/bind-mounted HMMER checkout directly. Local evaluations
build an isolated source copy; Slurm evaluations build one immutable,
content-addressed binary bundle and reuse it across all query-array workers.
"""

from __future__ import annotations

import glob
import hashlib
import json
import logging
import os
import re
from collections import Counter
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

from .eval_schema import EvalResult
from .unified_diff import PatchApplyError, apply_unified_diff
from . import run_jack
from .hmmer_stage_timing import (
    INSTRUMENTED_ENV_VAR,
    INSTRUMENTED_FILE,
    build_stage_timing,
    load_instrumented_stage_times,
    read_pipeline_evidence,
)
from .hmmer_instrument import (
    PROF_HEADER,
    PROF_SOURCE,
    InstrumentationError,
    enable_profiling_build,
    instrument_tree,
    resolve_instrumentation_targets,
    verify_contract_symbols,
    verify_instrumented_artifacts,
)

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[1]
logger = logging.getLogger(__name__)
DEFAULT_HMMER_ROOT = ROOT / "tools" / "hmmer-3.4"
DEFAULT_HMMER_SRC_ROOT = DEFAULT_HMMER_ROOT / "src"

# Where the profiling shim (p7_evo_prof.h/.c) lives. It is copied next to the
# HMMER sources it instruments, so `#include "p7_evo_prof.h"` resolves at build
# time without touching the pristine checkout.
PROFILING_DIR = ROOT / "harness_plugins" / "profiling"

# Appended to build_context so a profiling bundle never collides with (or silently
# reuses) the plain bundle for the same patch set.
PROFILE_BUILD_CONTEXT_SUFFIX = "+p7_evo_prof"

PROFILE_QUERY_LIMIT = 200
PROFILE_TOP_N = 10
BUNDLE_SCHEMA_VERSION = 1
BUNDLE_MANIFEST_NAME = "manifest.json"

# Which tool applies the agent's C patch: "auto" tries git first and falls back
# to the stdlib applier in `unified_diff`, "git"/"python" pin one of them.
PATCH_TOOL_ENV_VAR = "SEQ_ALIGN_PATCH_TOOL"
PATCH_TOOLS = ("auto", "git", "python")


class BackendUnsupported(ValueError):
    """Raised when a Python component cannot be represented by HMMER CLI flags."""


def run_all_queries(
    cfg,
    data_dir: Path | str | None = None,
    *,
    cpus: int = 1,
    stage: str = "full",
    max_iter: int = 5,
    astral_db: str | Path | None = None,
    uniref_db: str | Path | None = None,
    scop_txt: str | Path | None = None,
    hmmer_src_root: str | Path | None = None,
    work_root: str | Path | None = None,
    keep_work: bool = False,
    external_profiler: str = "off",
    perf_top_queries: int = 0,
    perf_top_symbols: int = 20,
    perf_frequency: int = 99,
    perf_command: str = "dominant",
    query_timeout: float | None = None,
    resume_provider=None,
    on_query_complete=None,
    hmmer_bundle_dir: str | Path | None = None,
    hmmer_bundle_id: str | None = None,
    profiling: bool = False,
) -> EvalResult:
    """Run compiled HMMER over all query fastas and aggregate an EvalResult."""
    ddir = Path(data_dir) if data_dir else run_jack.DIR_DATA_TEST
    seqs = [str(Path(p).resolve()) for p in run_jack.query_fastas(ddir)]
    astral_path = str(Path(astral_db or run_jack.ASTRAL_DB).resolve())
    uniref_path = str(Path(uniref_db or run_jack.UNIREF_DB).resolve())
    scop_path = str(Path(scop_txt or run_jack.SCOP_TXT).resolve())
    source_root = Path(hmmer_src_root or DEFAULT_HMMER_ROOT).resolve()
    eval_root = Path(work_root).resolve() if work_root is not None else Path(tempfile.mkdtemp(prefix="evomaster-hmmer-")).resolve()

    meta: dict[str, Any] = {
        "backend": "hmmer_cli",
        "scale": stage,
        "data_dir": str(ddir),
        "astral_db": astral_path,
        "uniref_db": uniref_path,
        "scop_txt": scop_path,
        "hmmer_src": str(source_root),
        "work_root": str(eval_root),
        "c_patches": [s.c_target_file for s in cfg.c_patches],
        "allocated_cpus": cpus,
        "hmmer_worker_threads": _hmmer_worker_threads(cpus),
        "query_timeout_s": query_timeout,
    }

    if not seqs:
        return EvalResult(stage=stage, status="run_error",
                          errors=[f"no query fastas under {ddir}"], meta=meta)
    if hmmer_bundle_dir is None and not source_root.exists():
        return EvalResult(stage=stage, status="compile_error",
                          errors=[f"HMMER source root not found: {source_root}"], meta=meta)
    for label, path in (("UNIREF_DB", uniref_path), ("ASTRAL_DB", astral_path)):
        if not os.path.exists(path):
            return EvalResult(stage=stage, status="run_error",
                              errors=[f"{label} not found: {path}"], meta=meta)

    try:
        cli_opts = _cli_options(cfg, max_iter=max_iter)
    except BackendUnsupported as e:
        return EvalResult(stage=stage, status="skipped", errors=[str(e)], meta=meta)

    prebuilt_dir = os.getenv("SEQ_ALIGN_HMMER_BIN_DIR")
    bundle_manifest: dict[str, Any] | None = None
    try:
        if hmmer_bundle_dir is not None:
            bundle_root = Path(hmmer_bundle_dir).resolve()
            bundle_manifest, jackhmmer, hmmsearch = load_binary_bundle(
                bundle_root,
                expected_bundle_id=hmmer_bundle_id,
                expected_patch_set_sha256=_patch_set_digest(cfg),
            )
            build_root = None
            stock_provenance = {}
            meta["prebuilt_hmmer"] = True
            meta["hmmer_bundle"] = {
                **bundle_manifest,
                "path": str(bundle_root),
            }
        # ``not profiling``: a prebuilt plain binary has no p7_evo_prof shim compiled
        # in, so serving a profiling request from it would produce a pass that
        # completes ``ok`` with no stage report and no error -- the exact silent
        # "no p7_evo_prof.txt" failure this branch used to cause. A profiling request
        # therefore falls through to the instrumented build below.
        elif prebuilt_dir and not cfg.c_patches and not profiling:
            jackhmmer = Path(prebuilt_dir) / "jackhmmer"
            hmmsearch = Path(prebuilt_dir) / "hmmsearch"
            missing = [str(path) for path in (jackhmmer, hmmsearch) if not path.is_file()]
            if missing:
                raise RuntimeError(f"prebuilt HMMER binaries are missing: {missing}")
            build_root = None
            stock_provenance = {}
            meta["prebuilt_hmmer"] = True
        else:
            if eval_root.exists():
                shutil.rmtree(eval_root)
            eval_root.mkdir(parents=True, exist_ok=True)
            build_root = eval_root / "hmmer-src"
            _copy_source(source_root, build_root)
            if cfg.c_patches:
                stock_jackhmmer, stock_hmmsearch = _build_hmmer(build_root, cpus=cpus, default_hmmer_root=source_root)
                stock_provenance = _source_tree_provenance(
                    build_root, cfg, binaries=(stock_jackhmmer, stock_hmmsearch)
                )
                _apply_patches(build_root, cfg)
            else:
                stock_provenance = {}
                _apply_patches(build_root, cfg)
            # Instrument only now. Two reasons: the stock build above has already
            # finished, so the timers cannot contaminate stock provenance; and the
            # rewritten .c files get fresh mtimes, which is what forces make to
            # recompile them (a CFLAGS-only change would not).
            local_instrumentation: dict[str, Any] = {}
            if profiling:
                resolved_targets, resolution = resolve_instrumentation_targets(build_root / "src")
                local_instrumentation = {
                    "resolution": resolution,
                    "resolved_targets": [list(item) for item in resolved_targets],
                }
                local_instrumentation["report"] = instrument_tree(
                    build_root / "src",
                    prof_header=PROFILING_DIR / PROF_HEADER,
                    prof_source=PROFILING_DIR / PROF_SOURCE,
                    targets=resolved_targets,
                )
            jackhmmer, hmmsearch = _build_hmmer(
                build_root,
                cpus=cpus,
                default_hmmer_root=source_root,
                profiling=profiling,
                instrumentation=local_instrumentation or None,
            )

        if bundle_manifest is not None:
            build_provenance = dict(bundle_manifest.get("build") or {})
            build_provenance.setdefault("mode", "content_addressed_bundle")
            build_provenance.setdefault("runtime_effective", bundle_manifest.get("runtime_effective"))
        elif build_root is not None:
            build_provenance = _build_provenance(
                source_root=source_root,
                build_root=build_root,
                cfg=cfg,
                stock_provenance=stock_provenance,
                binaries=(jackhmmer, hmmsearch),
            )
        else:
            build_provenance = {"mode": "prebuilt", "runtime_effective": True}
        meta.update({
            "jackhmmer_binary": str(jackhmmer),
            "hmmsearch_binary": str(hmmsearch),
            "jackhmmer_sha256": _hash_file(jackhmmer),
            "hmmsearch_sha256": _hash_file(hmmsearch),
            "max_iter": cli_opts["max_iter"],
            "build": build_provenance,
            "runtime_effective": build_provenance.get("runtime_effective"),
        })
    except EvaluationInterrupted:
        if not keep_work:
            _cleanup(eval_root)
        raise
    except Exception as e:
        # Keep failed build trees by default so config.log and compiler logs
        # remain available for Slurm diagnosis; successful runs still clean up.
        preserve_failed_work = keep_work or os.getenv("SEQ_ALIGN_KEEP_FAILED_WORK", "1").lower() not in {"0", "false", "no"}
        if not preserve_failed_work:
            _cleanup(eval_root)
        return EvalResult(stage=stage, status="compile_error",
                          errors=[f"compiled HMMER setup failed: {type(e).__name__}: {e}"], meta=meta)

    agg_tp = agg_fp = 0
    total_cost = 0.0
    max_iters = 0
    errors: list[str] = []
    statuses: list[str] = []
    query_profiles: list[dict[str, Any]] = []
    query_results: list[dict[str, Any]] = []
    scop_counts = {"tp": 0, "fp": 0, "ignored_same_fold_different_superfamily": 0, "unclassified": 0}
    scop = run_jack._load_scop_lineages(scop_path)
    resumed_results: dict[int, EvalResult] = {}
    if callable(resume_provider):
        supplied = resume_provider({
            "jackhmmer_sha256": meta["jackhmmer_sha256"],
            "hmmsearch_sha256": meta["hmmsearch_sha256"],
        })
        if isinstance(supplied, dict):
            resumed_results = {
                int(idx): (value if isinstance(value, EvalResult) else EvalResult.model_validate(value))
                for idx, value in supplied.items()
            }

    aggregate_profile: dict[str, Any] = {}
    try:
        for idx, query in enumerate(seqs, start=1):
            if idx in resumed_results:
                qres = resumed_results[idx]
            else:
                try:
                    qres = _run_one_query(
                        query, idx, jackhmmer=jackhmmer, hmmsearch=hmmsearch,
                        run_root=eval_root / "queries", cpus=cpus,
                        astral_db=astral_path, uniref_db=uniref_path,
                        scop=scop, cli_opts=cli_opts, stage=stage,
                        query_timeout=query_timeout, profiling=profiling,
                    )
                except EvaluationInterrupted as exc:
                    # Preserve identity when a custom/monkeypatched query runner
                    # raises instead of returning an EvalResult.
                    exc.query_idx = idx
                    exc.query_path = query
                    raise
                if callable(on_query_complete):
                    on_query_complete(idx, query, eval_root / "queries" / f"query_{idx:04d}", qres)
            statuses.append(qres.status)
            agg_tp += qres.tp
            agg_fp += qres.fp
            total_cost += qres.cost_s
            max_iters = max(max_iters, qres.iterations)
            errors.extend(qres.errors)
            query_results.append({
                "query_index": idx - 1,
                "query_id": qres.meta.get("query_id") or _query_id(query),
                "status": qres.status,
                "tp": qres.tp,
                "fp": qres.fp,
                "tp_fp_ratio": qres.tp_fp_ratio,
                "cost_s": qres.cost_s,
                "iterations": qres.iterations,
                "nhits": qres.meta.get("nhits"),
                "time_limit_s": query_timeout,
                "error": qres.errors[0] if qres.errors else None,
            })
            scoring = qres.meta.get("scop_scoring") if isinstance(qres.meta, dict) else None
            if isinstance(scoring, dict):
                for key in scop_counts:
                    scop_counts[key] += int(scoring.get(key, 0) or 0)
            profile = (qres.meta or {}).get("profile")
            if isinstance(profile, dict):
                query_profiles.append(profile)
        if all(s == "ok" for s in statuses):
            status = "ok"
        elif statuses and all(s in {"ok", "timeout"} for s in statuses):
            status = "timeout"
        elif all(s == "skipped" for s in statuses):
            status = "skipped"
        else:
            status = "run_error"
        total_cost = round(total_cost, 4)
        aggregate_profile = _aggregate_profiles(
            query_profiles,
            stage=stage,
            status=status,
            cpus=cpus,
            max_iter=cli_opts["max_iter"],
            query_count=len(seqs),
            cost_s=total_cost,
        )
        _attach_external_profile(
            aggregate_profile,
            query_profiles,
            jackhmmer=jackhmmer,
            hmmsearch=hmmsearch,
            eval_root=eval_root,
            cpus=cpus,
            astral_db=astral_path,
            uniref_db=uniref_path,
            cli_opts=cli_opts,
            external_profiler=external_profiler,
            perf_top_queries=perf_top_queries,
            perf_top_symbols=perf_top_symbols,
            perf_frequency=perf_frequency,
            perf_command=perf_command,
            cfg=cfg,
            source_root=source_root,
        )
    finally:
        if not keep_work:
            _cleanup(eval_root)

    meta["profile"] = aggregate_profile
    meta["query_results"] = query_results
    meta["scop_scoring"] = {
        "policy": "same_superfamily_positive_different_fold_negative",
        **scop_counts,
    }
    res = EvalResult(
        stage=stage, status=status,
        tp=agg_tp, fp=agg_fp, cost_s=total_cost,
        n_seqs=len(seqs), iterations=max_iters,
        errors=errors, meta=meta,
    )
    res.ratio()
    return res


def _copy_source(source_root: Path, build_root: Path) -> None:
    ignore = shutil.ignore_patterns(
        ".git", "*.o", "*.a", "*.so", "*.dylib", "*.exe",
        "jackhmmer", "hmmsearch",
    )
    shutil.copytree(source_root, build_root, symlinks=True, ignore=ignore)


def _apply_patches(build_root: Path, cfg) -> str:
    """Apply the variant's C patches to the private source copy.

    Returns the applier used ("git", "python", or "none"), so callers can record
    it in provenance.

    `git apply --recount` stays the first choice, so behaviour is unchanged
    wherever git is installed. Where it is not -- the Slurm compute nodes on the
    target cluster have no git, only the login node does -- the built-in applier
    in `unified_diff` takes over instead of failing the build with a bare
    FileNotFoundError. Override with SEQ_ALIGN_PATCH_TOOL=auto|git|python.
    """
    if not cfg.c_patches:
        return "none"
    patch_file = build_root.parent / "combined.patch"
    parts = [_normalize_patch_paths(spec.c_patch or "", spec.c_target_file or "") for spec in cfg.c_patches]
    patch_text = "\n".join(parts) + "\n"
    with patch_file.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(patch_text)

    choice = _patch_tool_choice()
    git_error: str | None = None
    if choice in ("auto", "git"):
        if shutil.which("git") is None:
            git_error = "git not found on PATH"
        else:
            try:
                env = _isolated_git_env(build_root)
                _run(["git", "apply", "--check", "--recount", str(patch_file)], cwd=build_root, label="git apply --check", env=env)
                _run(["git", "apply", "--recount", str(patch_file)], cwd=build_root, label="git apply", env=env)
                return "git"
            except EvaluationInterrupted:
                # A timeout or a signalled git is an infrastructure problem, not
                # a missing tool: retrying it with the fallback would only add
                # another hung call.
                raise
            except Exception as exc:  # noqa: BLE001 - any git failure falls back
                git_error = f"{type(exc).__name__}: {exc}"
        if choice == "git":
            raise RuntimeError(
                f"git apply failed and SEQ_ALIGN_PATCH_TOOL=git forbids the built-in applier: {git_error}"
            )

    try:
        applied = apply_unified_diff(patch_text, build_root)
    except (PatchApplyError, OSError) as exc:
        detail = f"built-in applier: {exc}"
        if git_error:
            detail = f"git ({git_error}); {detail}"
        raise RuntimeError(f"applying the C patch failed -- {detail}") from exc

    logger.warning(
        "git apply unavailable (%s); applied the C patch with the built-in applier: %s",
        git_error,
        ", ".join(applied),
    )
    return "python"


def _patch_tool_choice() -> str:
    choice = (os.environ.get(PATCH_TOOL_ENV_VAR) or "auto").strip().lower()
    if choice not in PATCH_TOOLS:
        raise RuntimeError(f"{PATCH_TOOL_ENV_VAR} must be one of {PATCH_TOOLS}, got {choice!r}")
    return choice


def _build_hmmer(
    build_root: Path,
    *,
    cpus: int,
    default_hmmer_root: Path,
    profiling: bool = False,
    instrumentation: dict[str, Any] | None = None,
) -> tuple[Path, Path]:
    build_root.mkdir(parents=True, exist_ok=True)

    # Configure the isolated copy. Invoking the configure script from the
    # original checkout can make Autoconf resolve srcdir to the wrong tree.
    configure_bin = str(build_root / "configure")
    if not (build_root / "configure").exists() or not (build_root / "Makefile.in").exists():
        raise RuntimeError(f"HMMER源码不完整，缺少configure相关文件: {default_hmmer_root}")

    if not (build_root / "Makefile").exists():
        env = os.environ.copy()
        # Do not let a user/Conda toolchain override the cluster compiler.
        for name in ("GCC_EXEC_PREFIX", "COMPILER_PATH", "LIBRARY_PATH"):
            env.pop(name, None)
        # The HMMER checkout ships a helper named `gcc`; remove source/build
        # directories from PATH so Autoconf resolves the system compiler.
        blocked = {str(default_hmmer_root.resolve()), str(build_root.resolve())}
        env["PATH"] = os.pathsep.join(
            item for item in env.get("PATH", "").split(os.pathsep)
            if item and str(Path(item).resolve()) not in blocked
        )
        env.setdefault("CC", "/usr/bin/gcc")
        _run([configure_bin, "--enable-threads"], cwd=build_root, label="configure", env=env)
    if profiling:
        # Turn the stage timers on in the *generated* Makefiles (top level AND the
        # implementation sub-Makefile that actually compiles the contract
        # functions). Applied after configure so Makefile.in (a tracked file) is
        # untouched and the plain build path never sees it. Idempotent.
        makefile_edit = enable_profiling_build(build_root / "src" / "Makefile")
        if instrumentation is not None:
            instrumentation["makefile_edit"] = makefile_edit
    _run(["make", "-j", str(max(1, cpus))], cwd=build_root, label="make")

    jackhmmer = build_root / "src" / "jackhmmer"
    hmmsearch = build_root / "src" / "hmmsearch"
    missing = [str(p) for p in (jackhmmer, hmmsearch) if not p.is_file()]
    if missing:
        raise RuntimeError(f"build completed but binaries are missing: {missing}")
    if profiling and instrumentation is not None:
        # Refuse to hand back a profiling build whose timers never reached the
        # binary. Verification also inspects the instrumented *sources*, so this
        # cannot be satisfied by a manifest claiming -DP7_EVO_PROF.
        instrumentation["binary_verification"] = verify_instrumented_artifacts(
            {"jackhmmer": jackhmmer, "hmmsearch": hmmsearch},
            instrumentation.get("resolved_targets") or (),
            file_root=build_root / "src",
        )
    return jackhmmer, hmmsearch



def _instrumentation_query(
    *,
    source_root: Path,
    profiling: bool,
) -> tuple[tuple[tuple[str, str, str, str], ...], dict[str, Any]]:
    """Resolve what a build of ``source_root`` must instrument, or nothing when off.

    Used both for a fresh build and for re-verifying a cache hit, so both paths
    agree on the expected target set. A pristine checkout has no generated
    Makefile, so the implementation directory is taken from the single
    ``impl_*/Makefile.in`` configure would select -- the same choice
    ``impl_dir_from_makefile`` reads back after configure.
    """
    if not profiling:
        return (), {}
    makefile = source_root / "src" / "Makefile"
    if makefile.is_file():
        return resolve_instrumentation_targets(source_root / "src")
    candidates = sorted((source_root / "src").glob("impl_*/Makefile.in"))
    if len(candidates) != 1:
        raise RuntimeError(
            "cannot resolve the instrumentation implementation before configure: "
            f"expected exactly one impl_*/Makefile.in under {source_root / 'src'}, found "
            f"{[p.name for p in candidates]}"
        )
    return resolve_instrumentation_targets(
        source_root / "src",
        makefile=_write_impl_dir_probe(makefile.parent, candidates[0].name),
    )


def _write_impl_dir_probe(src_dir: Path, impl_dir: str) -> Path:
    """Write a tiny temp Makefile carrying ``IMPLDIR`` so the resolver can read it.

    ``impl_dir_from_makefile`` only needs the one variable; this keeps a single
    implementation of the IMPLDIR parsing instead of duplicating the contract
    binding rules in a second place.
    """
    probe = src_dir / ".seqalign-impl-dir.probe.mk"
    probe.write_text(f"IMPLDIR = {impl_dir}\n", encoding="utf-8")
    return probe


def _quarantine_bundle(destination: Path, *, reason: str, bundle_id: str) -> Path | None:
    """Move an invalid cached bundle aside instead of deleting it.

    ``runs/`` is never deleted from; the invalid artifact is renamed into a
    timestamped sibling and an invalidation record is written next to it so the
    rebuild is auditable. Returns the quarantine directory.
    """
    if not destination.exists():
        return None
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    quarantine = destination.parent / f".invalid-{bundle_id[:12]}-{stamp}"
    suffix = 0
    while quarantine.exists():
        suffix += 1
        quarantine = destination.parent / f".invalid-{bundle_id[:12]}-{stamp}-{suffix}"
    try:
        os.replace(destination, quarantine)
    except OSError as exc:
        raise RuntimeError(f"could not quarantine invalid bundle {destination}: {exc}") from exc
    try:
        for root, dirs, files in os.walk(quarantine):
            for name in dirs + files:
                path = Path(root) / name
                try:
                    path.chmod(0o755 if path.is_dir() else 0o644)
                except OSError:
                    pass
    except OSError:
        pass
    record = {
        "status": "invalidated",
        "bundle_id": bundle_id,
        "quarantined_to": str(quarantine),
        "reason": reason,
        "invalidated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (quarantine / "INVALIDATED.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    return quarantine


def binary_bundle_identity(
    cfg,
    *,
    hmmer_src_root: str | Path | None = None,
    build_context: str = "host",
    profiling: bool = False,
) -> dict[str, str | int]:
    """Return the content identity used by an immutable HMMER bundle."""
    source_root = Path(hmmer_src_root or DEFAULT_HMMER_ROOT).resolve()
    source_sha256 = _source_tree_digest(source_root)
    patch_set_sha256 = _patch_set_digest(cfg)
    # The profiling variant is genuinely a different artifact (-DP7_EVO_PROF plus
    # the shim object), so it must not share a bundle id with the plain build.
    # Fold the flag into build_context rather than adding a payload key: that leaves
    # every pre-existing bundle id byte-identical, so no cached bundle is invalidated.
    # Idempotent, because the Slurm build worker round-trips the *already suffixed*
    # context back through this function.
    effective_context = build_context
    if profiling and not build_context.endswith(PROFILE_BUILD_CONTEXT_SUFFIX):
        effective_context = f"{build_context}{PROFILE_BUILD_CONTEXT_SUFFIX}"
    identity_payload = {
        "schema_version": BUNDLE_SCHEMA_VERSION,
        "source_tree_sha256": source_sha256,
        "patch_set_sha256": patch_set_sha256,
        "build_context": effective_context,
        "configure_policy": "use_existing_makefile_else_enable_threads",
    }
    encoded = json.dumps(identity_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {
        **identity_payload,
        "bundle_id": hashlib.sha256(encoded).hexdigest(),
    }


def build_binary_bundle(
    cfg,
    bundle_dir: str | Path,
    *,
    cpus: int = 4,
    hmmer_src_root: str | Path | None = None,
    build_context: str = "host",
    expected_bundle_id: str | None = None,
    profiling: bool = False,
) -> dict[str, Any]:
    """Build and atomically publish one content‑addressed HMMER bundle."""
    source_root = Path(hmmer_src_root or DEFAULT_HMMER_ROOT).resolve()


    destination = Path(bundle_dir).resolve()
    identity = binary_bundle_identity(
        cfg,
        hmmer_src_root=source_root,
        build_context=build_context,
        profiling=profiling,
    )
    bundle_id = str(identity["bundle_id"])
    if expected_bundle_id and bundle_id != expected_bundle_id:
        raise RuntimeError(
            "HMMER source or patch set changed after submission: "
            f"expected bundle {expected_bundle_id}, got {bundle_id}"
        )
    if destination.exists():
        manifest, cached_jackhmmer, cached_hmmsearch = load_binary_bundle(
            destination,
            expected_bundle_id=bundle_id,
            expected_patch_set_sha256=str(identity["patch_set_sha256"]),
        )
        # A cache hit is *not* evidence that the artifact is valid. The bundle id
        # covers the source tree, the patch set and the build context -- not the
        # build logic -- so a bundle produced by an older, broken profiling build
        # would otherwise be reused silently (exactly what happened on 2026-10-08:
        # wrappers linked, timers compiled out). Re-verify the published artifact
        # and the instrumented sources; rebuild when verification fails.
        try:
            resolved_targets, resolution = _instrumentation_query(source_root=source_root, profiling=profiling)
            if profiling:
                verification = verify_instrumented_artifacts(
                    {"jackhmmer": cached_jackhmmer, "hmmsearch": cached_hmmsearch},
                    resolved_targets,
                    file_root=destination / "src",
                )
                cached_instrumentation = {
                    "resolution": resolution,
                    "resolved_targets": [list(item) for item in resolved_targets],
                    "binary_verification": verification,
                    "cache_reverified": True,
                }
            else:
                # Non-profiling candidate bundles still have to contain the patched
                # contract entry symbols, otherwise the "patch" never compiled in.
                verification = verify_contract_symbols(
                    {"jackhmmer": cached_jackhmmer, "hmmsearch": cached_hmmsearch},
                    cfg,
                )
                cached_instrumentation = {"binary_verification": verification, "cache_reverified": True}
        except InstrumentationError as exc:
            reason = f"cache re-verification failed: {exc}"
            logger.warning("HMMER bundle %s failed re-verification, rebuilding: %s", bundle_id[:12], exc)
            _quarantine_bundle(destination, reason=reason, bundle_id=bundle_id)
        else:
            build = dict(manifest.get("build") or {})
            build["instrumentation"] = cached_instrumentation
            manifest = {**manifest, "build": build}
            return manifest



    destination.parent.mkdir(parents=True, exist_ok=True)
    work_parent_value = os.getenv("SEQ_ALIGN_HMMER_BUILD_WORK_ROOT", "").strip()
    work_parent = Path(work_parent_value).resolve() if work_parent_value else None
    if work_parent is not None:
        work_parent.mkdir(parents=True, exist_ok=True)
    staging_root = Path(tempfile.mkdtemp(
        prefix=f"evomaster-hmmer-build-{bundle_id[:12]}-",
        dir=work_parent,
    ))
    publish_root: Path | None = None
    started = time.monotonic()
    try:
        build_root = staging_root / "hmmer-src"
        _copy_source(source_root, build_root)
        if cfg.c_patches:
            stock_jackhmmer, stock_hmmsearch = _build_hmmer(build_root, cpus=cpus, default_hmmer_root=source_root)
            stock_provenance = _source_tree_provenance(
                build_root,
                cfg,
                binaries=(stock_jackhmmer, stock_hmmsearch),
            )
            _apply_patches(build_root, cfg)
        else:
            stock_provenance = {}
        # Instrument after the stock build, for the same two reasons as the local
        # path: stock provenance stays clean, and the rewritten units get fresh
        # mtimes so make actually recompiles them under the new flags.
        instrumentation: dict[str, Any] = {}
        if profiling:
            # Anchor the target set to the implementation this build compiles
            # (IMPLDIR in the generated Makefile) instead of to a hardcoded
            # impl_* directory, and fail loudly when a contract target cannot
            # carry a timer in that implementation.
            resolved_targets, resolution = resolve_instrumentation_targets(build_root / "src")
            instrumentation = {
                "resolution": resolution,
                "resolved_targets": [
                    list(item) for item in resolved_targets
                ],
            }
            instrumentation["report"] = instrument_tree(
                build_root / "src",
                prof_header=PROFILING_DIR / PROF_HEADER,
                prof_source=PROFILING_DIR / PROF_SOURCE,
                targets=resolved_targets,
            )
        jackhmmer, hmmsearch = _build_hmmer(
            build_root,
            cpus=cpus,
            default_hmmer_root=source_root,
            profiling=profiling,
            instrumentation=instrumentation or None,
        )
        build_provenance = _build_provenance(
            source_root=source_root,
            build_root=build_root,
            cfg=cfg,
            stock_provenance=stock_provenance,
            binaries=(jackhmmer, hmmsearch),
        )

        staged_bundle = staging_root / "bundle"
        bin_dir = staged_bundle / "bin"
        bin_dir.mkdir(parents=True)
        staged_binaries: dict[str, dict[str, Any]] = {}
        for name, source in (("jackhmmer", jackhmmer), ("hmmsearch", hmmsearch)):
            target = bin_dir / name
            shutil.copy2(source, target)
            target.chmod(0o555)
            staged_binaries[name] = {
                "path": f"bin/{name}",
                "size": target.stat().st_size,
                "sha256": _hash_file(target),
            }
        # Keep the instrumented units inside the bundle: a later cache hit
        # re-verifies that the sources really carry the wrapper bodies, which is
        # the check that catches a bundle built by code that instrumented nothing.
        staged_instrumented_sources: list[str] = []
        if profiling and instrumentation.get("resolved_targets"):
            for rel_path, _function, _slot, _contract in instrumentation["resolved_targets"]:
                source = build_root / "src" / rel_path
                if not source.is_file():
                    raise RuntimeError(
                        f"instrumented source missing after build, cannot publish bundle: {source}"
                    )
                target = staged_bundle / rel_path
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
                target.chmod(0o644)
                staged_instrumented_sources.append(rel_path)
        manifest: dict[str, Any] = {
            "schema_version": BUNDLE_SCHEMA_VERSION,
            "bundle_id": bundle_id,
            "build_context": build_context,
            "source_tree_sha256": identity["source_tree_sha256"],
            "patch_set_sha256": identity["patch_set_sha256"],
            "binaries": staged_binaries,
            "instrumented_sources": staged_instrumented_sources,
            "build": {
                **build_provenance,
                "mode": "content_addressed_bundle",
                # Auditability for the profiling variant: which impl directory the
                # build compiled, exactly what was instrumented, whether the shim
                # define was injected, and the post-build symbol verification. All
                # of this used to be unrecorded, so a profiling bundle that timed
                # nothing was indistinguishable from one that timed everything.
                "instrumentation": instrumentation or None,
                "excluded_from_cost_s": True,
            },
            "runtime_effective": build_provenance.get("runtime_effective"),
            "build_wall_s": round(time.monotonic() - started, 4),
            "excluded_from_cost_s": True,
        }
        manifest_path = staged_bundle / BUNDLE_MANIFEST_NAME
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
        publish_root = Path(tempfile.mkdtemp(prefix=f".{bundle_id[:12]}-", dir=destination.parent))
        publish_bundle = publish_root / "bundle"
        shutil.copytree(staged_bundle, publish_bundle)
        try:
            os.replace(publish_bundle, destination)
        except OSError:
            # A concurrent retry may have published the same content first.
            if not destination.exists():
                raise
            manifest, _, _ = load_binary_bundle(
                destination,
                expected_bundle_id=bundle_id,
                expected_patch_set_sha256=str(identity["patch_set_sha256"]),
            )
        else:
            (destination / BUNDLE_MANIFEST_NAME).chmod(0o444)
            for name in ("jackhmmer", "hmmsearch"):
                (destination / "bin" / name).chmod(0o555)
            (destination / "bin").chmod(0o555)
            destination.chmod(0o555)
        return manifest
    finally:
        _cleanup(staging_root)
        if publish_root is not None:
            _cleanup(publish_root)


def load_binary_bundle(
    bundle_dir: str | Path,
    *,
    expected_bundle_id: str | None = None,
    expected_patch_set_sha256: str | None = None,
) -> tuple[dict[str, Any], Path, Path]:
    """Validate an immutable bundle and return its manifest and binaries."""
    root = Path(bundle_dir).resolve()
    manifest_path = root / BUNDLE_MANIFEST_NAME
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"invalid HMMER bundle manifest {manifest_path}: {exc}") from exc
    if manifest.get("schema_version") != BUNDLE_SCHEMA_VERSION:
        raise RuntimeError(f"unsupported HMMER bundle schema: {manifest.get('schema_version')}")
    bundle_id = str(manifest.get("bundle_id") or "")
    if expected_bundle_id and bundle_id != expected_bundle_id:
        raise RuntimeError(f"HMMER bundle ID mismatch: expected {expected_bundle_id}, got {bundle_id}")
    patch_digest = str(manifest.get("patch_set_sha256") or "")
    if expected_patch_set_sha256 and patch_digest != expected_patch_set_sha256:
        raise RuntimeError(
            "HMMER bundle patch set mismatch: "
            f"expected {expected_patch_set_sha256}, got {patch_digest}"
        )

    resolved: dict[str, Path] = {}
    binaries = manifest.get("binaries")
    if not isinstance(binaries, dict):
        raise RuntimeError("HMMER bundle manifest has no binaries map")
    for name in ("jackhmmer", "hmmsearch"):
        artifact = binaries.get(name)
        if not isinstance(artifact, dict):
            raise RuntimeError(f"HMMER bundle manifest has no {name} artifact")
        relative = Path(str(artifact.get("path") or ""))
        candidate = (root / relative).resolve()
        try:
            candidate.relative_to(root)
        except ValueError as exc:
            raise RuntimeError(f"HMMER bundle {name} path escapes bundle root") from exc
        if not candidate.is_file():
            raise RuntimeError(f"HMMER bundle binary is missing: {candidate}")
        if not os.access(candidate, os.X_OK):
            raise RuntimeError(f"HMMER bundle binary is not executable: {candidate}")
        actual_hash = _hash_file(candidate)
        if actual_hash != artifact.get("sha256"):
            raise RuntimeError(
                f"HMMER bundle binary hash mismatch for {name}: "
                f"expected {artifact.get('sha256')}, got {actual_hash}"
            )
        resolved[name] = candidate
    return manifest, resolved["jackhmmer"], resolved["hmmsearch"]


def _patch_set_digest(cfg) -> str:
    patches = [
        {
            "component": spec.contract_name,
            "target_file": spec.c_target_file or "",
            "patch": _normalize_patch_paths(spec.c_patch or "", spec.c_target_file or ""),
        }
        for spec in cfg.c_patches
    ]
    encoded = json.dumps(patches, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _source_tree_digest(root: Path) -> str:
    if not root.is_dir():
        raise RuntimeError(f"HMMER source root not found: {root}")
    ignored_suffixes = {".o", ".a", ".so", ".dylib", ".exe"}
    digest = hashlib.sha256()
    files = [
        path for path in root.rglob("*")
        if path.is_file()
        and ".git" not in path.relative_to(root).parts
        and path.suffix not in ignored_suffixes
        and path.name not in {"jackhmmer", "hmmsearch"}
    ]
    for path in sorted(files, key=lambda item: item.relative_to(root).as_posix()):
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _build_provenance(
    *,
    source_root: Path,
    build_root: Path,
    cfg,
    stock_provenance: dict[str, Any],
    binaries: tuple[Path, Path],
) -> dict[str, Any]:
    patched = _source_tree_provenance(build_root, cfg, binaries=binaries)
    comparisons = {
        "target_sources": _compare_artifact_lists(
            stock_provenance.get("target_sources", []),
            patched.get("target_sources", []),
        ),
        "target_objects": _compare_artifact_lists(
            stock_provenance.get("target_objects", []),
            patched.get("target_objects", []),
        ),
        "libraries": _compare_artifact_maps(
            stock_provenance.get("libraries", {}),
            patched.get("libraries", {}),
        ),
        "binaries": _compare_artifact_maps(
            stock_provenance.get("binaries", {}),
            patched.get("binaries", {}),
        ),
    }
    comparable_binaries = [item for item in comparisons["binaries"].values() if item.get("comparable")]
    runtime_effective = None
    if cfg.c_patches and comparable_binaries:
        runtime_effective = any(item.get("changed") for item in comparable_binaries)
    return {
        "source_root": str(source_root),
        "build_root": str(build_root),
        "stock": stock_provenance,
        "patched": patched,
        "comparisons": comparisons,
        "runtime_effective": runtime_effective,
    }


def _source_tree_provenance(root: Path, cfg, *, binaries: tuple[Path, Path] | None = None) -> dict[str, Any]:
    target_sources: list[dict[str, Any]] = []
    target_objects: list[dict[str, Any]] = []
    for spec in cfg.c_patches:
        target_file = spec.c_target_file or ""
        source = _resolve_target_source(root, target_file)
        target_sources.append({
            "component": spec.contract_name,
            "target_file": target_file,
            "artifact": _artifact_info(source, root=root),
        })
        target_objects.append({
            "component": spec.contract_name,
            "target_file": target_file,
            "artifact": _artifact_info(source.with_suffix(".o"), root=root),
        })
    if binaries is None:
        binary_paths = {
            "jackhmmer": root / "src" / "jackhmmer",
            "hmmsearch": root / "src" / "hmmsearch",
        }
    else:
        binary_paths = {"jackhmmer": binaries[0], "hmmsearch": binaries[1]}
    return {
        "target_sources": target_sources,
        "target_objects": target_objects,
        "libraries": {
            "libhmmer.a": _artifact_info(root / "src" / "libhmmer.a", root=root),
            # Architecture-specific kernels are archived separately from the
            # generic library. Track both paths so a versioned NEON/SSE
            # contract records which implementation archive was actually
            # rebuilt; the final executable hashes remain the authoritative
            # runtime-effectiveness gate.
            "impl_neon/libhmmerimpl.a": _artifact_info(
                root / "src" / "impl_neon" / "libhmmerimpl.a",
                root=root,
            ),
            "impl_sse/libhmmerimpl.a": _artifact_info(
                root / "src" / "impl_sse" / "libhmmerimpl.a",
                root=root,
            ),
        },
        "binaries": {name: _artifact_info(path, root=root) for name, path in binary_paths.items()},
    }


def _resolve_target_source(root: Path, target_file: str) -> Path:
    target = target_file.strip().lstrip("./")
    candidates = [root / "src" / target, root / target]
    if root.name == "src":
        candidates.extend([root.parent / target, root.parent / "src" / target])
    if target.startswith("easel/"):
        candidates.append(root / target.removeprefix("easel/"))
        if root.name == "src":
            candidates.append(root.parent / target)
    candidates.extend([
        root / "src" / Path(target).name,
        root / Path(target).name,
    ])
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return root / target if target.startswith("easel/") else root / "src" / target


def _artifact_info(path: Path, *, root: Path) -> dict[str, Any]:
    try:
        relpath = str(path.relative_to(root))
    except ValueError:
        relpath = str(path)
    info: dict[str, Any] = {"path": str(path), "relpath": relpath, "exists": path.exists()}
    if not path.exists():
        return info
    stat = path.stat()
    info.update({"size": stat.st_size, "mtime_ns": stat.st_mtime_ns, "sha256": _hash_file(path)})
    return info


def _compare_artifact_lists(before: list[dict[str, Any]], after: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for idx, after_item in enumerate(after):
        before_item = before[idx] if idx < len(before) else {}
        before_artifact = before_item.get("artifact", {})
        after_artifact = after_item.get("artifact", {})
        out.append({
            "component": after_item.get("component") or before_item.get("component"),
            "target_file": after_item.get("target_file") or before_item.get("target_file"),
            **_artifact_comparison(before_artifact, after_artifact),
        })
    return out


def _compare_artifact_maps(before: dict[str, Any], after: dict[str, Any]) -> dict[str, dict[str, Any]]:
    keys = sorted(set(before) | set(after))
    return {key: _artifact_comparison(before.get(key, {}), after.get(key, {})) for key in keys}


def _artifact_comparison(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    before_hash = before.get("sha256")
    after_hash = after.get("sha256")
    comparable = before_hash is not None and after_hash is not None
    return {
        "before": before,
        "after": after,
        "comparable": comparable,
        "changed": (before_hash != after_hash) if comparable else None,
    }


def _run_one_query(
    query_path: str,
    query_idx: int,
    *,
    jackhmmer: Path,
    hmmsearch: Path,
    run_root: Path,
    cpus: int,
    astral_db: str,
    uniref_db: str,
    scop: dict[str, str],
    cli_opts: dict[str, Any],
    stage: str,
    query_timeout: float | None = None,
    profiling: bool = False,
) -> EvalResult:
    qdir = run_root / f"query_{query_idx:04d}"
    qdir.mkdir(parents=True, exist_ok=True)
    query_id = _query_id(query_path)
    meta = {"backend": "hmmer_cli", "query_id": query_id, "query_path": query_path}
    command_profiles: dict[str, dict[str, Any]] = {}
    t0 = time.perf_counter()

    chk_prefix = qdir / "chk"
    jack_tbl = qdir / "jackhmmer.tbl"
    jack_out = qdir / "jackhmmer.out"
    cmd = [
        str(jackhmmer),
        "-N", str(cli_opts["max_iter"]),
        "--cpu", str(_hmmer_worker_threads(cpus)),
        "--noali",
        "--tblout", str(jack_tbl),
        "--chkhmm", str(chk_prefix),
        "-o", str(jack_out),
        *cli_opts["jackhmmer"],
        query_path,
        uniref_db,
    ]
    try:
        # The instrumented build writes its stage report to $P7_EVO_PROF_OUT. Point
        # it at this query's own work directory -- exactly where
        # load_instrumented_stage_times(qdir) looks for it. Without the profiling
        # build the variable is inert (the shim is a no-op), so this is safe to set
        # unconditionally on this path.
        env: dict[str, str] | None = None
        if profiling:
            env = dict(os.environ)
            env[INSTRUMENTED_ENV_VAR] = str(qdir / INSTRUMENTED_FILE)
        _, command_profiles["jackhmmer"] = _timed_run(
            cmd, cwd=qdir, label="jackhmmer", timeout=query_timeout, env=env
        )
    except QueryTimeoutError as e:
        return _timeout_result(
            stage=stage,
            query_idx=query_idx,
            query_id=query_id,
            query_path=query_path,
            qdir=qdir,
            started_at=t0,
            iterations=0,
            command_profiles=command_profiles,
            error=e,
            meta=meta,
        )
    except CommandProfileError as e:
        command_profiles["jackhmmer"] = e.profile
        cost_s = round(time.perf_counter() - t0, 4)
        meta["profile"] = _query_profile(
            query_idx=query_idx,
            query_id=query_id,
            query_path=query_path,
            qdir=qdir,
            status="run_error",
            cost_s=cost_s,
            iterations=0,
            nhits=None,
            commands=command_profiles,
        )
        return EvalResult(stage=stage, status="run_error", n_seqs=1,
                          cost_s=cost_s, meta=meta,
                          errors=[f"jackhmmer failed: {e}"])

    hmm_path, iterations = _last_checkpoint(chk_prefix)
    if hmm_path is None:
        cost_s = round(time.perf_counter() - t0, 4)
        meta["profile"] = _query_profile(
            query_idx=query_idx,
            query_id=query_id,
            query_path=query_path,
            qdir=qdir,
            status="run_error",
            cost_s=cost_s,
            iterations=0,
            nhits=None,
            commands=command_profiles,
        )
        return EvalResult(stage=stage, status="run_error", n_seqs=1,
                          cost_s=cost_s, meta=meta,
                          errors=["jackhmmer produced no --chkhmm checkpoint"])

    search_tbl = qdir / "hmmsearch.tbl"
    search_out = qdir / "hmmsearch.out"
    search_cmd = [
        str(hmmsearch),
        "--cpu", str(_hmmer_worker_threads(cpus)),
        "--noali",
        "--tblout", str(search_tbl),
        "-o", str(search_out),
        *cli_opts["hmmsearch"],
        str(hmm_path),
        astral_db,
    ]
    elapsed = time.perf_counter() - t0
    remaining_timeout = None if query_timeout is None else query_timeout - elapsed
    if remaining_timeout is not None and remaining_timeout <= 0:
        return _timeout_result(
            stage=stage,
            query_idx=query_idx,
            query_id=query_id,
            query_path=query_path,
            qdir=qdir,
            started_at=t0,
            iterations=iterations,
            command_profiles=command_profiles,
            error=QueryTimeoutError(
                label="query_total",
                timeout_s=query_timeout,
                wall_s=elapsed,
            ),
            meta=meta,
        )
    try:
        _, command_profiles["hmmsearch"] = _timed_run(
            search_cmd, cwd=qdir, label="hmmsearch", timeout=remaining_timeout
        )
    except QueryTimeoutError as e:
        return _timeout_result(
            stage=stage,
            query_idx=query_idx,
            query_id=query_id,
            query_path=query_path,
            qdir=qdir,
            started_at=t0,
            iterations=iterations,
            command_profiles=command_profiles,
            error=e,
            meta=meta,
        )
    except CommandProfileError as e:
        command_profiles["hmmsearch"] = e.profile
        cost_s = round(time.perf_counter() - t0, 4)
        meta["profile"] = _query_profile(
            query_idx=query_idx,
            query_id=query_id,
            query_path=query_path,
            qdir=qdir,
            status="run_error",
            cost_s=cost_s,
            iterations=iterations,
            nhits=None,
            commands=command_profiles,
        )
        return EvalResult(stage=stage, status="run_error", n_seqs=1,
                          iterations=iterations, cost_s=cost_s,
                          meta=meta, errors=[f"hmmsearch failed: {e}"])

    query_lineage = run_jack._query_scop_lineage(query_id, scop)
    targets = _parse_tblout_targets(search_tbl)
    nhits = len(targets)
    meta["nhits"] = nhits
    cost_s = round(time.perf_counter() - t0, 4)
    if query_lineage is None:
        meta["profile"] = _query_profile(
            query_idx=query_idx,
            query_id=query_id,
            query_path=query_path,
            qdir=qdir,
            status="skipped",
            cost_s=cost_s,
            iterations=iterations,
            nhits=nhits,
            commands=command_profiles,
        )
        return EvalResult(stage=stage, status="skipped", n_seqs=1,
                          iterations=iterations, cost_s=cost_s,
                          meta=meta, errors=["no SCOP lineage for query; cannot score TP/FP"])
    score = _score_target_names(targets, scop, query_lineage)
    meta["scop_scoring"] = score.meta()
    meta["profile"] = _query_profile(
        query_idx=query_idx,
        query_id=query_id,
        query_path=query_path,
        qdir=qdir,
        status="ok",
        cost_s=cost_s,
        iterations=iterations,
        nhits=nhits,
        commands=command_profiles,
    )
    res = EvalResult(stage=stage, status="ok", tp=score.tp, fp=score.fp, n_seqs=1,
                     iterations=iterations, cost_s=cost_s, meta=meta)
    res.ratio()
    return res


def _cli_options(cfg, *, max_iter: int) -> dict[str, Any]:
    jack_opts: list[str] = []
    search_opts: list[str] = []
    iter_budget = max_iter

    conv = cfg.specs.get("convergence_rule")
    if conv:
        payload = conv.payload or {}
        rule = payload.get("rule", "no_new_targets")
        if rule == "fixed":
            iter_budget = max(1, int(payload.get("n", max_iter)))
        elif rule != "no_new_targets":
            raise BackendUnsupported(
                f"hmmer_cli backend cannot represent convergence_rule '{rule}' with CLI flags."
            )

    filt = cfg.specs.get("filter_cascade")
    if filt:
        payload = filt.payload or {}
        pipe = filt.pipeline_kwargs or {}
        if payload.get("max"):
            jack_opts.append("--max")
            search_opts.append("--max")
        else:
            for key in ("F1", "F2", "F3"):
                if key in pipe:
                    jack_opts.extend([f"--{key}", str(pipe[key])])
                    search_opts.extend([f"--{key}", str(pipe[key])])
            if pipe.get("bias_filter") is False:
                jack_opts.append("--nobias")
                search_opts.append("--nobias")
        for key, flag in (("E", "-E"), ("T", "-T"), ("domE", "--domE"), ("domT", "--domT")):
            if key in pipe:
                jack_opts.extend([flag, str(pipe[key])])
                search_opts.extend([flag, str(pipe[key])])

    inc = cfg.specs.get("inclusion_strategy")
    if inc:
        if inc.hook_factory is not None:
            raise BackendUnsupported(
                "hmmer_cli backend cannot execute an inclusion_strategy callback"
            )
        for key in ("incE", "incT", "incdomE", "incdomT"):
            if key in inc.pipeline_kwargs:
                jack_opts.extend([f"--{key}", str(inc.pipeline_kwargs[key])])

    score = cfg.specs.get("scoring_system")
    if score:
        bkw = score.builder_kwargs or {}
        if "score_matrix" in bkw:
            jack_opts.extend(["--mx", str(bkw["score_matrix"])])
        for key in ("popen", "pextend"):
            if key in bkw:
                jack_opts.extend([f"--{key}", str(bkw[key])])

    weight = cfg.specs.get("weighting_strategy")
    if weight:
        weight_payload = weight.payload or {}
        weighting = weight_payload.get("weighting")
        flag = {"pb": "--wpb", "gsc": "--wgsc", "blosum": "--wblosum", "none": "--wnone"}.get(weighting)
        if flag:
            jack_opts.append(flag)
        if "wid" in weight_payload:
            jack_opts.extend(["--wid", str(weight_payload["wid"])])

    eff = cfg.specs.get("eff_seq_strategy")
    if eff:
        payload = eff.payload or {}
        effn = payload.get("effn")
        if effn == "eset":
            jack_opts.extend(["--eset", str(payload.get("eset_value", 100))])
        else:
            flag = {
                "eent": "--eent", "entropy": "--eent", "eentexp": "--eentexp",
                "eclust": "--eclust", "clust": "--eclust", "enone": "--enone", "none": "--enone",
            }.get(effn)
            if flag:
                jack_opts.append(flag)
        for key in ("ere", "esigma"):
            if key in payload:
                jack_opts.extend([f"--{key}", str(payload[key])])

    return {"jackhmmer": jack_opts, "hmmsearch": search_opts, "max_iter": iter_budget}


def _hmmer_worker_threads(allocated_cpus: int) -> int:
    """Return the literal HMMER ``--cpu`` value for one query.

    ``allocated_cpus`` is the number of cores *one query* may use -- never a whole
    job's allocation. It is passed straight through, because the caller has already
    accounted for HMMER's coordinator thread:

      * the Slurm worker passes the stage's ``*_QUERY_CPUS`` (one query per task),
        and the job allocation reserves one extra core per concurrent query;
      * the local backend passes the single-process core budget.

    The previous ``- 1`` here was the only place that knew about the coordinator, so
    a job-level allocation leaked into ``--cpu``: with a 45-core job HMMER would
    have been told ``--cpu 44``. The subtraction is gone and the two quantities are
    now supplied by two different callers.
    """
    return max(1, int(allocated_cpus))


def _normalize_patch_paths(diff: str, target_file: str) -> str:
    target = target_file.strip().lstrip("./")
    target_name = Path(target).name
    canonical_target = target if target.startswith(("src/", "easel/")) else f"src/{target}"

    def normalize_path_token(token: str) -> str:
        prefix = ""
        path = token
        if token.startswith("a/") or token.startswith("b/"):
            prefix, path = token[:2], token[2:]
        clean = path.lstrip("./")
        if clean.endswith(target) or clean == target_name or ("/" not in target and Path(clean).name == target_name):
            path = canonical_target
        return prefix + path

    out: list[str] = []
    for line in diff.splitlines():
        if line.startswith("diff --git "):
            parts = line.split()
            if len(parts) >= 4:
                parts[2] = normalize_path_token(parts[2])
                parts[3] = normalize_path_token(parts[3])
                line = " ".join(parts)
        elif line.startswith(("--- ", "+++ ")):
            head, rest = line[:4], line[4:]
            path, *suffix = rest.split("\t", 1)
            if path not in {"/dev/null", "dev/null"}:
                path = normalize_path_token(path)
            line = head + path + (("\t" + suffix[0]) if suffix else "")
        out.append(line)
    return "\n".join(out)


def _isolated_git_env(build_root: Path) -> dict[str, str]:
    env = dict(os.environ)
    env["GIT_CEILING_DIRECTORIES"] = str(build_root.parent)
    return env



class CommandProfileError(RuntimeError):
    """Runtime error carrying bounded command timing metadata."""

    def __init__(self, message: str, profile: dict[str, Any]):
        super().__init__(message)
        self.profile = profile


class EvaluationInterrupted(RuntimeError):
    """Evaluation stopped by an external signal or hard wall-clock bound."""

    def __init__(
        self,
        message: str,
        *,
        label: str | None = None,
        returncode: int | None = None,
    ):
        super().__init__(message)
        self.label = label
        self.returncode = returncode
        self.query_idx: int | None = None
        self.query_path: str | None = None


class QueryTimeoutError(EvaluationInterrupted, TimeoutError):
    """A bounded query command exceeded its wall-clock budget."""

    def __init__(
        self,
        *,
        label: str,
        timeout_s: float,
        wall_s: float,
        stdout_tail: str = "",
        stderr_tail: str = "",
    ):
        super().__init__(
            f"{label} exceeded query time limit {float(timeout_s):.4f}s "
            f"after {float(wall_s):.4f}s",
            label=label,
        )
        self.label = label
        self.timeout_s = float(timeout_s)
        self.wall_s = float(wall_s)
        self.stdout_tail = stdout_tail
        self.stderr_tail = stderr_tail
        self.profile = {
            "label": label,
            "status": "timeout",
            "wall_s": round(float(wall_s), 4),
            "timeout_s": round(float(timeout_s), 4),
            "returncode": None,
        }


def _tail_text(value: str | bytes | None, limit: int) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    return value[-limit:]


def _run(
    cmd: list[str],
    *,
    cwd: Path,
    label: str,
    env: dict[str, str] | None = None,
    timeout: float | None = None,
) -> subprocess.CompletedProcess[str]:
    started_at = time.perf_counter()
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(cwd),
            capture_output=True,
            text=True,
            env=env,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        wall_s = time.perf_counter() - started_at
        raise QueryTimeoutError(
            label=label,
            timeout_s=float(timeout if timeout is not None else exc.timeout),
            wall_s=wall_s,
            stdout_tail=_tail_text(exc.stdout or exc.output, 1000),
            stderr_tail=_tail_text(exc.stderr, 2000),
        ) from exc
    if proc.returncode != 0:
        stderr = (proc.stderr or "")[-2000:]
        stdout = (proc.stdout or "")[-1000:]
        if proc.returncode < 0:
            signum = -proc.returncode
            raise EvaluationInterrupted(
                f"{label} terminated by signal {signum}",
                label=label,
                returncode=proc.returncode,
            )
        raise RuntimeError(f"{label} exited {proc.returncode}; stdout={stdout!r}; stderr={stderr!r}")
    return proc


def _timed_run(
    cmd: list[str],
    *,
    cwd: Path,
    label: str,
    env: dict[str, str] | None = None,
    timeout: float | None = None,
) -> tuple[subprocess.CompletedProcess[str], dict[str, Any]]:
    t0 = time.perf_counter()
    try:
        proc = _run(cmd, cwd=cwd, label=label, env=env, timeout=timeout)
    except QueryTimeoutError as e:
        e.wall_s = max(e.wall_s, time.perf_counter() - t0)
        e.profile.update({"wall_s": round(e.wall_s, 4), "timeout_s": round(e.timeout_s, 4)})
        raise
    except EvaluationInterrupted:
        raise
    except Exception as e:
        wall_s = round(time.perf_counter() - t0, 4)
        profile = {
            "label": label,
            "status": "error",
            "wall_s": wall_s,
            "returncode": getattr(e, "returncode", None),
        }
        raise CommandProfileError(str(e), profile) from e
    wall_s = round(time.perf_counter() - t0, 4)
    profile = {"label": label, "status": "ok", "wall_s": wall_s, "returncode": proc.returncode}
    return proc, profile


def _timeout_result(
    *,
    stage: str,
    query_idx: int,
    query_id: str,
    query_path: str,
    qdir: Path,
    started_at: float,
    iterations: int,
    command_profiles: dict[str, dict[str, Any]],
    error: QueryTimeoutError,
    meta: dict[str, Any],
) -> EvalResult:
    command_profiles[error.label] = dict(error.profile)
    cost_s = round(max(error.wall_s, time.perf_counter() - started_at), 4)
    meta.update({
        "timeout": {
            "phase": error.label,
            "time_limit_s": round(error.timeout_s, 4),
            "wall_s": cost_s,
            "stdout_tail": error.stdout_tail,
            "stderr_tail": error.stderr_tail,
        },
        "profile": _query_profile(
            query_idx=query_idx,
            query_id=query_id,
            query_path=query_path,
            qdir=qdir,
            status="timeout",
            cost_s=cost_s,
            iterations=iterations,
            nhits=None,
            commands=command_profiles,
        ),
    })
    return EvalResult(
        stage=stage,
        status="timeout",
        n_seqs=1,
        iterations=iterations,
        cost_s=cost_s,
        meta=meta,
        errors=[str(error)],
    )


def _query_profile(
    *,
    query_idx: int,
    query_id: str,
    query_path: str,
    qdir: Path,
    status: str,
    cost_s: float,
    iterations: int,
    nhits: int | None,
    commands: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    chk_prefix = qdir / "chk"
    hmm_path, last_iteration = _last_checkpoint(chk_prefix)
    # Per-stage evidence. HMMER already prints a per-iteration elapsed time and the
    # per-filter pass counts into these files, so this costs no extra run and does
    # not touch cost_s. `time_s` per stage only appears if the instrumented
    # profiling build produced it -- see hmmer_stage_timing's module docstring.
    jack_text, jack_read = read_pipeline_evidence(qdir / "jackhmmer.out")
    search_text, search_read = read_pipeline_evidence(qdir / "hmmsearch.out")
    # Stage times are present only when the instrumented profiling build ran for
    # this query. Absent them the report carries counts and per-iteration elapsed
    # only -- it never invents a per-stage number.
    stage_times_s, stage_time_diagnostics = load_instrumented_stage_times(qdir)
    stage_timing = build_stage_timing(
        jackhmmer_text=jack_text,
        hmmsearch_text=search_text,
        stage_times_s=stage_times_s,
        stage_calls=(stage_time_diagnostics or {}).get("calls"),
    )
    stage_timing["evidence_reads"] = {
        "jackhmmer_out": jack_read,
        "hmmsearch_out": search_read,
        "instrumented_profile": stage_time_diagnostics,
    }
    return {
        "schema_version": "hmmer_cli_query_profile.v1",
        "backend": "hmmer_cli",
        "scope": "single_query_phase_excluding_build",
        "timer": "time.perf_counter",
        "query_idx": query_idx,
        "query_id": query_id,
        "query_path": query_path,
        "status": status,
        "cost_s": round(float(cost_s), 4),
        "iterations": int(iterations or 0),
        "nhits": nhits,
        "commands": {name: dict(profile) for name, profile in sorted(commands.items())},
        "stage_timing": stage_timing,
        "artifacts": {
            "jackhmmer_tbl": _file_profile(qdir / "jackhmmer.tbl"),
            "jackhmmer_out": _file_profile(qdir / "jackhmmer.out"),
            "hmmsearch_tbl": _file_profile(qdir / "hmmsearch.tbl"),
            "hmmsearch_out": _file_profile(qdir / "hmmsearch.out"),
            "checkpoints": {
                "count": len(glob.glob(str(chk_prefix) + "-*.hmm")),
                "last_iteration": last_iteration,
                "last_bytes": _file_profile(hmm_path)["bytes"] if hmm_path else 0,
            },
        },
    }


def _file_profile(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {"exists": False, "bytes": 0}
    try:
        if not path.exists():
            return {"exists": False, "bytes": 0}
        return {
            "exists": True,
            "bytes": path.stat().st_size,
            "sha256": _hash_file(path),
        }
    except OSError as e:
        return {"exists": path.exists(), "bytes": 0, "error": str(e)}


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
    status_counts = dict(sorted(Counter(str(p.get("status", "unknown")) for p in profiles).items()))
    command_wall_s: dict[str, float] = {}
    command_status_counts: dict[str, dict[str, int]] = {}
    costs: list[float] = []
    iterations: list[int] = []
    nhits: list[int] = []
    for profile in profiles:
        costs.append(float(profile.get("cost_s") or 0.0))
        iterations.append(int(profile.get("iterations") or 0))
        if profile.get("nhits") is not None:
            nhits.append(int(profile.get("nhits") or 0))
        commands = profile.get("commands") or {}
        if isinstance(commands, dict):
            for name, command in commands.items():
                if not isinstance(command, dict):
                    continue
                command_wall_s[name] = round(command_wall_s.get(name, 0.0) + float(command.get("wall_s") or 0.0), 4)
                counts = command_status_counts.setdefault(name, {})
                command_status = str(command.get("status") or "unknown")
                counts[command_status] = counts.get(command_status, 0) + 1
    measured_total = round(sum(command_wall_s.values()), 4)
    unattributed_s = round(max(0.0, float(cost_s) - measured_total), 4)
    total_for_fraction = measured_total if measured_total > 0 else float(cost_s)
    command_wall_fraction = {
        name: round((wall_s / total_for_fraction) if total_for_fraction else 0.0, 6)
        for name, wall_s in sorted(command_wall_s.items())
    }
    dominant_command = None
    if command_wall_s:
        dominant_command = max(command_wall_s.items(), key=lambda item: item[1])[0]
    top_queries = [
        _compact_query_profile(profile)
        for profile in sorted(profiles, key=lambda item: float(item.get("cost_s") or 0.0), reverse=True)[:PROFILE_TOP_N]
    ]
    stored_queries = [_compact_query_profile(profile) for profile in profiles[:PROFILE_QUERY_LIMIT]]
    # Session-level stage rollup: "which stage dominates across the whole eval".
    # Per-stage time_s only exists when the instrumented profiling build ran.
    stage_totals: dict[str, float] = {}
    slowest_iteration_by_query: list[tuple[float, Any]] = []
    iteration_elapsed: list[float] = []
    stage_time_available = False
    # Persist *why* the instrumented build produced nothing. The per-query
    # diagnostic ("p7_evo_prof.txt not present", "no recognised stage lines",
    # schema/call counts) used to be dropped here, so a failed profiling pass was
    # recorded as "no stage times" with no cause attached.
    instrumented_profile_diagnostic: dict[str, Any] | None = None
    per_query_instrumented_profile: list[dict[str, Any]] = []
    for profile in profiles:
        stage_timing = profile.get("stage_timing") or {}
        if stage_timing.get("stage_time_available"):
            stage_time_available = True
        evidence = stage_timing.get("evidence_reads") or {}
        diagnostic = evidence.get("instrumented_profile") if isinstance(evidence, dict) else None
        if isinstance(diagnostic, dict):
            if instrumented_profile_diagnostic is None:
                instrumented_profile_diagnostic = dict(diagnostic)
            per_query_instrumented_profile.append(
                {
                    "query_idx": profile.get("query_idx"),
                    "query_id": profile.get("query_id"),
                    "instrumented_profile": dict(diagnostic),
                }
            )
        # Per-query reports carry per-stage seconds either as a flat mapping or as
        # per-stage entries; both are summed into the same totals.
        stage_seconds = stage_timing.get("stage_time_s") or {
            name: entry.get("time_s")
            for name, entry in (stage_timing.get("contract_stages") or {}).items()
            if isinstance(entry, dict)
        }
        for name, seconds in stage_seconds.items():
            if seconds is not None:
                stage_totals[name] = round(stage_totals.get(name, 0.0) + float(seconds), 4)
        summary = stage_timing.get("jackhmmer_summary") or {}
        if summary.get("elapsed_total_s") is not None:
            iteration_elapsed.append(float(summary["elapsed_total_s"]))
            slowest_iteration_by_query.append((float(summary["elapsed_total_s"]), profile.get("query_id")))
    measured_stage_total = round(sum(stage_totals.values()), 4)
    stage_time_fraction = {
        name: round((value / measured_stage_total) if measured_stage_total else 0.0, 6)
        for name, value in sorted(stage_totals.items())
    }
    slowest_iteration_query = (
        max(slowest_iteration_by_query, key=lambda item: item[0])[1]
        if slowest_iteration_by_query
        else None
    )
    return {
        "schema_version": "hmmer_cli_eval_profile.v1",
        "backend": "hmmer_cli",
        "scope": "eval_query_phase_excluding_build",
        "timer": "time.perf_counter",
        "stage": stage,
        "status": status,
        "cpus": cpus,
        "max_iter": max_iter,
        "query_count": query_count,
        "profiled_query_count": len(profiles),
        "queries_truncated": len(profiles) > PROFILE_QUERY_LIMIT,
        "query_count_total": len(profiles),
        "status_counts": status_counts,
        "command_wall_s": dict(sorted(command_wall_s.items())),
        "command_wall_fraction": command_wall_fraction,
        "command_status_counts": {name: dict(sorted(counts.items())) for name, counts in sorted(command_status_counts.items())},
        "dominant_command": dominant_command,
        "measured_command_wall_s": measured_total,
        "unattributed_wall_s": unattributed_s,
        "query_cost_stats": _numeric_stats(costs),
        "iteration_stats": _numeric_stats(iterations),
        "nhits_stats": _numeric_stats(nhits),
        "stage_timing": {
            "time_source": "instrumented_build" if stage_time_available else "output_parse",
            "stage_time_available": stage_time_available,
            "stage_time_s": dict(sorted(stage_totals.items())),
            "stage_time_fraction": stage_time_fraction,
            "measured_stage_total_s": measured_stage_total,
            "jackhmmer_elapsed_stats": _numeric_stats(iteration_elapsed),
            "slowest_iteration_query_id": slowest_iteration_query,
            "instrumented_profile": instrumented_profile_diagnostic,
            "per_query_instrumented_profile": per_query_instrumented_profile,
            "note": (
                "stage_time_s requires the instrumented profiling build; without it "
                "only per-iteration elapsed and per-filter counts are available. "
                "instrumented_profile carries the reason when stage times are absent."
            ),
        },
        "top_slowest_queries": top_queries,
        "queries": stored_queries,
    }


def _attach_external_profile(
    aggregate_profile: dict[str, Any],
    query_profiles: list[dict[str, Any]],
    *,
    jackhmmer: Path,
    hmmsearch: Path,
    eval_root: Path,
    cpus: int,
    astral_db: str,
    uniref_db: str,
    cli_opts: dict[str, Any],
    external_profiler: str,
    perf_top_queries: int,
    perf_top_symbols: int,
    perf_frequency: int,
    perf_command: str,
    cfg: Any | None = None,
    source_root: Path | None = None,
) -> None:
    """Attach optional external profiler evidence without affecting eval status/cost."""
    mode = (external_profiler or "off").lower()
    if mode == "off":
        return
    metadata: dict[str, Any] = {
        "tool": mode,
        "status": "skipped",
        "excluded_from_cost_s": True,
        "strategy": "post_eval_top_slowest_queries",
        "top_queries_requested": max(0, int(perf_top_queries or 0)),
        "top_symbols_requested": max(1, int(perf_top_symbols or 1)),
        "frequency": max(1, int(perf_frequency or 1)),
        "command_selector": perf_command,
    }
    aggregate_profile["external_profiler"] = metadata
    if mode == "gprof":
        _attach_gprof_profile(
            aggregate_profile,
            query_profiles,
            metadata=metadata,
            cfg=cfg,
            source_root=source_root,
            eval_root=eval_root,
            cpus=cpus,
            astral_db=astral_db,
            uniref_db=uniref_db,
            cli_opts=cli_opts,
            perf_top_queries=perf_top_queries,
            perf_top_symbols=perf_top_symbols,
            perf_command=perf_command,
        )
        return
    if mode != "perf":
        metadata["reason"] = f"unsupported external_profiler={external_profiler}"
        return
    perf_bin = shutil.which("perf")
    if not perf_bin:
        metadata["reason"] = "perf executable not found"
        return
    if aggregate_profile.get("status") != "ok":
        metadata["reason"] = f"eval status is {aggregate_profile.get('status')}; profiler post-pass requires ok eval"
        return
    if int(perf_top_queries or 0) <= 0:
        metadata["reason"] = "perf_top_queries <= 0"
        return
    targets = _perf_target_commands(aggregate_profile, perf_command)
    if not targets:
        metadata["reason"] = "no target command selected for perf profiling"
        return
    selected = [p for p in sorted(query_profiles, key=lambda item: float(item.get("cost_s") or 0.0), reverse=True) if p.get("status") == "ok"]
    selected = selected[: max(0, int(perf_top_queries or 0))]
    if not selected:
        metadata["reason"] = "no successful query profiles available for perf profiling"
        return

    profiled: list[dict[str, Any]] = []
    for profile in selected:
        for command in targets:
            profiled.append(_profile_query_command_with_perf(
                profile,
                command,
                perf_bin=perf_bin,
                jackhmmer=jackhmmer,
                hmmsearch=hmmsearch,
                eval_root=eval_root,
                cpus=cpus,
                astral_db=astral_db,
                uniref_db=uniref_db,
                cli_opts=cli_opts,
                frequency=max(1, int(perf_frequency or 1)),
                top_symbols=max(1, int(perf_top_symbols or 1)),
            ))
    metadata["profiled_commands"] = [
        {
            "query_idx": item.get("query_idx"),
            "query_id": item.get("query_id"),
            "command": item.get("command"),
            "status": item.get("status"),
            "symbol_count": len(item.get("symbols") or []),
            "reason": item.get("reason"),
        }
        for item in profiled
    ]
    ok_count = sum(1 for item in profiled if item.get("status") == "ok")
    if ok_count == len(profiled) and profiled:
        metadata["status"] = "ok"
    elif ok_count:
        metadata["status"] = "partial"
    elif any(item.get("status") == "error" for item in profiled):
        metadata["status"] = "error"
    else:
        metadata["status"] = "skipped"
    symbol_hotspots, command_hotspots = _aggregate_symbol_profiles(profiled, top_symbols=max(1, int(perf_top_symbols or 1)))
    if symbol_hotspots:
        aggregate_profile["symbol_hotspots"] = symbol_hotspots
        aggregate_profile["command_symbol_hotspots"] = command_hotspots
    elif "reason" not in metadata:
        metadata["reason"] = _external_profile_empty_reason(profiled)


def _external_profile_empty_reason(profiled: list[dict[str, Any]]) -> str:
    for item in profiled:
        if item.get("status") == "error" and item.get("reason"):
            return f"profiler command failed: {item.get('reason')}"
    for item in profiled:
        if item.get("reason"):
            return f"profiler skipped: {item.get('reason')}"
    return "no symbols parsed from profiler reports"


def _attach_gprof_profile(
    aggregate_profile: dict[str, Any],
    query_profiles: list[dict[str, Any]],
    *,
    metadata: dict[str, Any],
    cfg: Any | None,
    source_root: Path | None,
    eval_root: Path,
    cpus: int,
    astral_db: str,
    uniref_db: str,
    cli_opts: dict[str, Any],
    perf_top_queries: int,
    perf_top_symbols: int,
    perf_command: str,
) -> None:
    gprof_bin = shutil.which("gprof")
    if not gprof_bin:
        metadata["reason"] = "gprof executable not found"
        return
    if cfg is None or source_root is None:
        metadata["reason"] = "gprof profiling requires cfg and source_root"
        return
    if aggregate_profile.get("status") != "ok":
        metadata["reason"] = f"eval status is {aggregate_profile.get('status')}; profiler post-pass requires ok eval"
        return
    if int(perf_top_queries or 0) <= 0:
        metadata["reason"] = "perf_top_queries <= 0"
        return
    targets = _perf_target_commands(aggregate_profile, perf_command)
    if not targets:
        metadata["reason"] = "no target command selected for gprof profiling"
        return
    selected = [p for p in sorted(query_profiles, key=lambda item: float(item.get("cost_s") or 0.0), reverse=True) if p.get("status") == "ok"]
    selected = selected[: max(0, int(perf_top_queries or 0))]
    if not selected:
        metadata["reason"] = "no successful query profiles available for gprof profiling"
        return

    try:
        gprof_jackhmmer, gprof_hmmsearch = _build_gprof_hmmer_copy(
            source_root=source_root,
            cfg=cfg,
            eval_root=eval_root,
            cpus=cpus,
        )
    except EvaluationInterrupted:
        raise
    except Exception as exc:
        metadata["status"] = "error"
        metadata["reason"] = f"gprof build failed: {_short_error(exc)}"
        return

    profiled: list[dict[str, Any]] = []
    for profile in selected:
        for command in targets:
            profiled.append(_profile_query_command_with_gprof(
                profile,
                command,
                gprof_bin=gprof_bin,
                jackhmmer=gprof_jackhmmer,
                hmmsearch=gprof_hmmsearch,
                eval_root=eval_root,
                cpus=cpus,
                astral_db=astral_db,
                uniref_db=uniref_db,
                cli_opts=cli_opts,
                top_symbols=max(1, int(perf_top_symbols or 1)),
            ))
    metadata["profiled_commands"] = [
        {
            "query_idx": item.get("query_idx"),
            "query_id": item.get("query_id"),
            "command": item.get("command"),
            "status": item.get("status"),
            "symbol_count": len(item.get("symbols") or []),
            "reason": item.get("reason"),
        }
        for item in profiled
    ]
    ok_count = sum(1 for item in profiled if item.get("status") == "ok")
    if ok_count == len(profiled) and profiled:
        metadata["status"] = "ok"
    elif ok_count:
        metadata["status"] = "partial"
    elif any(item.get("status") == "error" for item in profiled):
        metadata["status"] = "error"
    else:
        metadata["status"] = "skipped"
    symbol_hotspots, command_hotspots = _aggregate_symbol_profiles(profiled, top_symbols=max(1, int(perf_top_symbols or 1)))
    if symbol_hotspots:
        aggregate_profile["symbol_hotspots"] = symbol_hotspots
        aggregate_profile["command_symbol_hotspots"] = command_hotspots
    elif "reason" not in metadata:
        metadata["reason"] = _external_profile_empty_reason(profiled)


def _build_gprof_hmmer_copy(
    *,
    source_root: Path,
    cfg: Any,
    eval_root: Path,
    cpus: int,
) -> tuple[Path, Path]:
    build_root = eval_root / "external_profiler" / "gprof" / "hmmer-src-gprof"
    if build_root.exists():
        shutil.rmtree(build_root)
    _copy_source(Path(source_root), build_root)
    _apply_patches(build_root, cfg)
    env = dict(os.environ)
    env.pop("CFLAGS", None)
    env.pop("LDFLAGS", None)
    env.update({"CFLAGS": "-pg -O2", "LDFLAGS": "-pg"})
    configure_bin = str(build_root / "configure")
    for name in ("GCC_EXEC_PREFIX", "COMPILER_PATH", "LIBRARY_PATH"):
        env.pop(name, None)
    _run([configure_bin, "--enable-threads"], cwd=build_root, label="configure gprof", env=env)
    _run(["make", "-j", str(max(1, cpus))], cwd=build_root, label="make gprof", env=env)
    jackhmmer = build_root / "src" / "jackhmmer"
    hmmsearch = build_root / "src" / "hmmsearch"
    missing = [str(p) for p in (jackhmmer, hmmsearch) if not p.exists()]
    if missing:
        raise RuntimeError(f"gprof build completed but binaries are missing: {missing}")
    return jackhmmer, hmmsearch



def _profile_query_command_with_gprof(
    profile: dict[str, Any],
    command: str,
    *,
    gprof_bin: str,
    jackhmmer: Path,
    hmmsearch: Path,
    eval_root: Path,
    cpus: int,
    astral_db: str,
    uniref_db: str,
    cli_opts: dict[str, Any],
    top_symbols: int,
) -> dict[str, Any]:
    query_idx = int(profile.get("query_idx") or 0)
    query_id = str(profile.get("query_id") or query_idx or "unknown")
    result: dict[str, Any] = {"query_idx": query_idx, "query_id": query_id, "command": command, "status": "skipped"}
    commands = profile.get("commands") or {}
    command_profile = commands.get(command) if isinstance(commands, dict) else None
    if not isinstance(command_profile, dict) or command_profile.get("status") != "ok":
        result["reason"] = "original command profile is unavailable or not ok"
        return result
    command_wall_s = float(command_profile.get("wall_s") or 0.0)
    prof_dir = eval_root / "external_profiler" / "gprof" / f"query_{query_idx:04d}" / command
    prof_dir.mkdir(parents=True, exist_ok=True)
    replay = _perf_replay_command(
        profile,
        command,
        jackhmmer=jackhmmer,
        hmmsearch=hmmsearch,
        eval_root=eval_root,
        prof_dir=prof_dir,
        cpus=cpus,
        astral_db=astral_db,
        uniref_db=uniref_db,
        cli_opts=cli_opts,
    )
    if replay is None:
        result["reason"] = "could not reconstruct command for gprof replay"
        return result
    gmon_out = prof_dir / "gmon.out"
    if gmon_out.exists():
        gmon_out.unlink()
    binary = jackhmmer if command == "jackhmmer" else hmmsearch
    try:
        replay_profile = _run_gprof_replay(replay, cwd=prof_dir, command=command)
        if not gmon_out.exists():
            result.update({"status": "error", "reason": "gprof replay produced no gmon.out"})
            return result
        report_text = _run_gprof_report(gprof_bin, binary=binary, gmon_out=gmon_out, cwd=prof_dir)
        symbols = _parse_gprof_report(report_text, top_n=top_symbols, command=command, query_idx=query_idx, query_id=query_id)
    except EvaluationInterrupted:
        raise
    except Exception as exc:
        result.update({"status": "error", "reason": _short_error(exc)})
        return result
    if not symbols:
        result["reason"] = "no symbols parsed"
        return result
    for symbol in symbols:
        symbol["approx_command_wall_s"] = round(command_wall_s * float(symbol.get("self_pct") or 0.0) / 100.0, 4)
        symbol["profiled_command_wall_s"] = round(command_wall_s, 4)
    result.update({
        "status": "ok",
        "wall_s": replay_profile.get("wall_s"),
        "returncode": replay_profile.get("returncode"),
        "symbols": symbols,
    })
    return result


def _run_gprof_replay(cmd: list[str], *, cwd: Path, command: str) -> dict[str, Any]:
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True)
    wall_s = round(time.perf_counter() - t0, 4)
    if proc.returncode != 0:
        raise RuntimeError(f"gprof replay {command} exited {proc.returncode}: {_bounded_stderr(proc)}")
    return {"status": "ok", "wall_s": wall_s, "returncode": proc.returncode}


def _run_gprof_report(gprof_bin: str, *, binary: Path, gmon_out: Path, cwd: Path, max_chars: int = 200_000) -> str:
    proc = subprocess.run([gprof_bin, str(binary), str(gmon_out)], cwd=str(cwd), capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"gprof exited {proc.returncode}: {_bounded_stderr(proc)}")
    return (proc.stdout or "")[:max_chars]


def _parse_gprof_report(text: str, *, top_n: int, command: str | None = None, query_idx: int | None = None, query_id: str | None = None) -> list[dict[str, Any]]:
    """Parse bounded gprof flat-profile rows."""
    rows: list[dict[str, Any]] = []
    in_flat_profile = False
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("Flat profile:"):
            in_flat_profile = True
            continue
        if in_flat_profile and stripped.startswith("Call graph"):
            break
        if stripped.startswith("Each sample"):
            continue
        if stripped.startswith("%") or stripped.startswith("time"):
            in_flat_profile = True
            continue
        if not in_flat_profile and not re.match(r"^\d+(?:\.\d+)?\s+\d", stripped):
            continue
        match = re.match(
            r"^(?P<pct>\d+(?:\.\d+)?)\s+(?P<cumulative>\d+(?:\.\d+)?)\s+(?P<self>\d+(?:\.\d+)?)(?:\s+(?P<rest>.+))?$",
            stripped,
        )
        if not match:
            continue
        rest = (match.group("rest") or "").strip()
        parts = rest.split()
        if not parts:
            continue
        symbol_start = 0
        while symbol_start < len(parts) - 1 and _gprof_numeric_field(parts[symbol_start]):
            symbol_start += 1
        symbol = " ".join(parts[symbol_start:]).strip()
        symbol = re.sub(r"\s+\[\d+\]$", "", symbol)
        if not symbol:
            continue
        rows.append({
            "rank": len(rows) + 1,
            "command": str(command or ""),
            "query_idx": query_idx,
            "query_id": query_id,
            "symbol": symbol,
            "dso": "gprof",
            "self_pct": round(float(match.group("pct")), 4),
            "self_s": round(float(match.group("self")), 4),
            "profile_source": "gprof_flat_profile",
        })
        if len(rows) >= top_n:
            break
    return rows


def _gprof_numeric_field(value: str) -> bool:
    return bool(re.match(r"^\d+(?:\.\d+)?(?:/\d+(?:\.\d+)?)?$", value))


def _perf_target_commands(aggregate_profile: dict[str, Any], selector: str) -> list[str]:
    selector_l = (selector or "dominant").lower()
    available = set((aggregate_profile.get("command_wall_s") or {}).keys())
    if selector_l == "both":
        return [name for name in ("jackhmmer", "hmmsearch") if name in available]
    if selector_l in {"jackhmmer", "hmmsearch"}:
        return [selector_l] if selector_l in available else []
    dominant = aggregate_profile.get("dominant_command")
    return [str(dominant)] if dominant in {"jackhmmer", "hmmsearch"} else []


def _profile_query_command_with_perf(
    profile: dict[str, Any],
    command: str,
    *,
    perf_bin: str,
    jackhmmer: Path,
    hmmsearch: Path,
    eval_root: Path,
    cpus: int,
    astral_db: str,
    uniref_db: str,
    cli_opts: dict[str, Any],
    frequency: int,
    top_symbols: int,
) -> dict[str, Any]:
    query_idx = int(profile.get("query_idx") or 0)
    query_id = str(profile.get("query_id") or query_idx or "unknown")
    result: dict[str, Any] = {"query_idx": query_idx, "query_id": query_id, "command": command, "status": "skipped"}
    commands = profile.get("commands") or {}
    command_profile = commands.get(command) if isinstance(commands, dict) else None
    if not isinstance(command_profile, dict) or command_profile.get("status") != "ok":
        result["reason"] = "original command profile is unavailable or not ok"
        return result
    command_wall_s = float(command_profile.get("wall_s") or 0.0)
    prof_dir = eval_root / "external_profiler" / f"query_{query_idx:04d}" / command
    prof_dir.mkdir(parents=True, exist_ok=True)
    replay = _perf_replay_command(
        profile,
        command,
        jackhmmer=jackhmmer,
        hmmsearch=hmmsearch,
        eval_root=eval_root,
        prof_dir=prof_dir,
        cpus=cpus,
        astral_db=astral_db,
        uniref_db=uniref_db,
        cli_opts=cli_opts,
    )
    if replay is None:
        result["reason"] = "could not reconstruct command for perf replay"
        return result
    perf_data = prof_dir / "perf.data"
    try:
        record_profile = _run_perf_record(perf_bin, replay, perf_data=perf_data, cwd=prof_dir, frequency=frequency)
        report_text = _run_perf_report(perf_bin, perf_data=perf_data, cwd=prof_dir)
        symbols = _parse_perf_report(report_text, top_n=top_symbols, command=command, query_idx=query_idx, query_id=query_id)
    except EvaluationInterrupted:
        raise
    except Exception as exc:
        result.update({"status": "error", "reason": _short_error(exc)})
        return result
    if not symbols:
        result["reason"] = "no symbols parsed"
        return result
    for symbol in symbols:
        symbol["approx_command_wall_s"] = round(command_wall_s * float(symbol.get("self_pct") or 0.0) / 100.0, 4)
        symbol["profiled_command_wall_s"] = round(command_wall_s, 4)
    result.update({
        "status": "ok",
        "wall_s": record_profile.get("wall_s"),
        "returncode": record_profile.get("returncode"),
        "symbols": symbols,
    })
    return result


def _perf_replay_command(
    profile: dict[str, Any],
    command: str,
    *,
    jackhmmer: Path,
    hmmsearch: Path,
    eval_root: Path,
    prof_dir: Path,
    cpus: int,
    astral_db: str,
    uniref_db: str,
    cli_opts: dict[str, Any],
) -> list[str] | None:
    query_path = str(profile.get("query_path") or "")
    if command == "jackhmmer" and query_path:
        return [
            str(jackhmmer),
            "-N", str(cli_opts["max_iter"]),
            "--cpu", str(_hmmer_worker_threads(cpus)),
            "--noali",
            "--tblout", str(prof_dir / "jackhmmer.tbl"),
            "--chkhmm", str(prof_dir / "chk"),
            "-o", str(prof_dir / "jackhmmer.out"),
            *cli_opts["jackhmmer"],
            query_path,
            uniref_db,
        ]
    if command == "hmmsearch":
        qidx = int(profile.get("query_idx") or 0)
        qdir = eval_root / "queries" / f"query_{qidx:04d}"
        hmm_path, _ = _last_checkpoint(qdir / "chk")
        if hmm_path is None:
            return None
        return [
            str(hmmsearch),
            "--cpu", str(_hmmer_worker_threads(cpus)),
            "--noali",
            "--tblout", str(prof_dir / "hmmsearch.tbl"),
            "-o", str(prof_dir / "hmmsearch.out"),
            *cli_opts["hmmsearch"],
            str(hmm_path),
            astral_db,
        ]
    return None


def _run_perf_record(perf_bin: str, cmd: list[str], *, perf_data: Path, cwd: Path, frequency: int) -> dict[str, Any]:
    perf_cmd = [perf_bin, "record", "-F", str(frequency), "-g", "--output", str(perf_data), "--", *cmd]
    t0 = time.perf_counter()
    proc = subprocess.run(perf_cmd, cwd=str(cwd), capture_output=True, text=True)
    wall_s = round(time.perf_counter() - t0, 4)
    if proc.returncode != 0:
        raise RuntimeError(f"perf record exited {proc.returncode}: {_bounded_stderr(proc)}")
    return {"status": "ok", "wall_s": wall_s, "returncode": proc.returncode}


def _run_perf_report(perf_bin: str, *, perf_data: Path, cwd: Path, max_chars: int = 200_000) -> str:
    proc = subprocess.run([perf_bin, "report", "--stdio", "--input", str(perf_data)], cwd=str(cwd), capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"perf report exited {proc.returncode}: {_bounded_stderr(proc)}")
    return (proc.stdout or "")[:max_chars]


def _parse_perf_report(text: str, *, top_n: int, command: str | None = None, query_idx: int | None = None, query_id: str | None = None) -> list[dict[str, Any]]:
    """Parse bounded `perf report --stdio` symbol rows."""
    rows: list[dict[str, Any]] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        match = re.match(r"^(?P<pct>\d+(?:\.\d+)?)%\s+(?P<rest>.+)$", stripped)
        if not match:
            continue
        self_pct = float(match.group("pct"))
        rest = match.group("rest")
        parts = rest.split()
        if parts and re.match(r"^\d+(?:\.\d+)?%$", parts[0]):
            self_pct = float(parts[0][:-1])
            parts = parts[1:]
        if not parts:
            continue
        marker_idx = next((idx for idx, part in enumerate(parts) if part.startswith("[") and part.endswith("]")), None)
        if marker_idx is not None:
            parsed_command = parts[0] if marker_idx >= 2 else command
            dso = parts[marker_idx - 1] if marker_idx >= 1 else ""
            symbol = " ".join(parts[marker_idx + 1:]).strip()
        else:
            parsed_command = parts[0] if len(parts) >= 3 else command
            dso = parts[-2] if len(parts) >= 2 else ""
            symbol = parts[-1]
        if not symbol or symbol in {"[.]", "[k]"}:
            continue
        rows.append({
            "rank": len(rows) + 1,
            "command": str(command or parsed_command or ""),
            "query_idx": query_idx,
            "query_id": query_id,
            "symbol": symbol,
            "dso": dso,
            "self_pct": round(self_pct, 4),
            "profile_source": "perf_report_stdio",
        })
        if len(rows) >= top_n:
            break
    return rows


def _aggregate_symbol_profiles(profiled: list[dict[str, Any]], *, top_symbols: int) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    grouped: dict[tuple[str, str, str], dict[str, Any]] = {}
    for item in profiled:
        if item.get("status") != "ok":
            continue
        for symbol in item.get("symbols") or []:
            key = (str(symbol.get("command") or item.get("command") or ""), str(symbol.get("symbol") or ""), str(symbol.get("dso") or ""))
            if not key[1]:
                continue
            profile_source = str(symbol.get("profile_source") or item.get("profile_source") or "external_profiler")
            group = grouped.setdefault(key, {
                "command": key[0],
                "symbol": key[1],
                "dso": key[2],
                "approx_command_wall_s": 0.0,
                "max_self_pct": 0.0,
                "queries": [],
                "profile_source": profile_source,
            })
            if group.get("profile_source") != profile_source:
                group["profile_source"] = "mixed_external_profiler"
            group["approx_command_wall_s"] = round(group["approx_command_wall_s"] + float(symbol.get("approx_command_wall_s") or 0.0), 4)
            group["max_self_pct"] = max(float(group.get("max_self_pct") or 0.0), float(symbol.get("self_pct") or 0.0))
            query_ref = {"query_idx": item.get("query_idx"), "query_id": item.get("query_id")}
            if query_ref not in group["queries"]:
                group["queries"].append(query_ref)
    hotspots = sorted(grouped.values(), key=lambda row: float(row.get("approx_command_wall_s") or 0.0), reverse=True)
    out: list[dict[str, Any]] = []
    for rank, item in enumerate(hotspots[:top_symbols], start=1):
        row = dict(item)
        row["rank"] = rank
        row["queries"] = row["queries"][:5]
        out.append(row)
    by_command: dict[str, list[dict[str, Any]]] = {}
    for item in out:
        by_command.setdefault(str(item.get("command") or ""), []).append(item)
    return out, {command: rows[:top_symbols] for command, rows in sorted(by_command.items())}


def _bounded_stderr(proc: subprocess.CompletedProcess[str]) -> str:
    stderr = (proc.stderr or "")[-1000:]
    stdout = (proc.stdout or "")[-500:]
    return f"stdout={stdout!r}; stderr={stderr!r}"


def _short_error(exc: Exception) -> str:
    text = str(exc).replace("\n", " ")
    return text[:500]


def _compact_query_profile(profile: dict[str, Any]) -> dict[str, Any]:
    commands = profile.get("commands") or {}
    command_wall_s = {
        str(name): round(float(command.get("wall_s") or 0.0), 4)
        for name, command in commands.items()
        if isinstance(command, dict)
    }
    dominant_command = max(command_wall_s.items(), key=lambda item: item[1])[0] if command_wall_s else None
    return {
        "query_idx": profile.get("query_idx"),
        "query_id": profile.get("query_id"),
        "status": profile.get("status"),
        "cost_s": round(float(profile.get("cost_s") or 0.0), 4),
        "iterations": int(profile.get("iterations") or 0),
        "nhits": profile.get("nhits"),
        "command_wall_s": dict(sorted(command_wall_s.items())),
        "dominant_command": dominant_command,
        "stage_timing": _compact_stage_timing(profile.get("stage_timing")),
    }


def _compact_stage_timing(stage_timing: Any) -> dict[str, Any]:
    """Reduce a per-query stage report to what is worth storing per query.

    The full report keeps every iteration; stored/top-N profiles keep only the
    headline facts, so eval records do not grow with iteration count. The
    instrumented-profile diagnostic is kept because it is the *reason* a
    profiling run produced no stage times -- without it the record only says
    "no stage times", which is what made the 2026-10-08 probe failure
    undiagnosable after the fact.
    """
    if not isinstance(stage_timing, dict) or not stage_timing:
        return {}
    compact: dict[str, Any] = {
        "time_source": stage_timing.get("time_source"),
        "stage_time_available": bool(stage_timing.get("stage_time_available")),
    }
    evidence = stage_timing.get("evidence_reads")
    if isinstance(evidence, dict):
        compact["evidence_reads"] = dict(evidence)
    summary = stage_timing.get("jackhmmer_summary")
    if isinstance(summary, dict):
        compact["iterations_reported"] = summary.get("iterations_reported")
        compact["jackhmmer_elapsed_total_s"] = summary.get("elapsed_total_s")
        compact["slowest_iteration_round"] = summary.get("slowest_iteration_round")
        compact["slowest_iteration_elapsed_s"] = summary.get("slowest_iteration_elapsed_s")
    stage_time_s = {
        name: entry["time_s"]
        for name, entry in (stage_timing.get("contract_stages") or {}).items()
        if isinstance(entry, dict) and entry.get("time_s") is not None
    }
    if stage_time_s:
        compact["stage_time_s"] = dict(sorted(stage_time_s.items()))
    return compact


def _numeric_stats(values: list[float] | list[int]) -> dict[str, Any]:
    if not values:
        return {"count": 0}
    sorted_values = sorted(float(v) for v in values)
    count = len(sorted_values)
    return {
        "count": count,
        "min": round(sorted_values[0], 4),
        "mean": round(sum(sorted_values) / count, 4),
        "p50": round(_percentile(sorted_values, 0.50), 4),
        "p90": round(_percentile(sorted_values, 0.90), 4),
        "max": round(sorted_values[-1], 4),
    }


def _percentile(sorted_values: list[float], fraction: float) -> float:
    if len(sorted_values) == 1:
        return sorted_values[0]
    pos = fraction * (len(sorted_values) - 1)
    lower = int(pos)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = pos - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def _last_checkpoint(prefix: Path) -> tuple[Path | None, int]:
    files = glob.glob(str(prefix) + "-*.hmm")
    best: tuple[int, Path] | None = None
    for fp in files:
        m = re.search(r"-(\d+)\.hmm$", fp)
        if not m:
            continue
        iteration = int(m.group(1))
        if best is None or iteration > best[0]:
            best = (iteration, Path(fp))
    if best is None:
        return None, 0
    return best[1], best[0]


def _query_id(query_path: str) -> str:
    with open(query_path, "r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if line.startswith(">"):
                return line[1:].strip().split()[0]
    return Path(query_path).stem


def _parse_tblout_targets(tblout: Path) -> list[str]:
    targets: list[str] = []
    with open(tblout, "r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if not line.strip() or line.startswith("#"):
                continue
            targets.append(line.split()[0])
    return targets


def _score_target_names(
    targets: list[str], scop: dict[str, str], query_lineage: str
) -> run_jack.ScopScore:
    return run_jack._score_scop_names(targets, scop, query_lineage)


def _hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _cleanup(path: Path) -> None:
    shutil.rmtree(path, ignore_errors=True)
