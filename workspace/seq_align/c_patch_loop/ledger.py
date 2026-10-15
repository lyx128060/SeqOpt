"""Auditable CSV ledger of every evaluation/screening record a C-patch run produced.

The store under ``<run_dir>/c_patch_loop/`` stays the single source of truth. This
module is *only* a derived view: it reads the immutable records and writes one CSV.
It never writes into the run directory, never advances the loop, and never needs the
loop to be running.

Record kinds emitted (``record_kind`` column)
---------------------------------------------
``evaluation``
    One row per ``evaluations/*.json`` record.
``attempt``
    One row per ``evaluation_attempts/*.json`` record. The store writes each
    measurement TWICE -- a ``started`` row (``parent_attempt_id: null``) and a
    terminal ``complete``/``interrupted`` row whose ``parent_attempt_id`` is the
    started row's ``attempt_id``. Those are two distinct ``attempt_id`` values, so
    the ledger emits **two rows and does not collapse them**: every persisted
    attempt record gets exactly one row. The pair stays auditable through the extra
    ``attempt_id`` / ``parent_attempt_id`` / ``status`` columns. A terminal row
    carries the metrics; its ``started`` parent legitimately has none.
``screening_rejection``
    One row per inner-trial screening rejection. The controller labels them
    ``rejected_screening_not_faster`` and stores them **only** in
    ``indexes/run_summary.json`` and/or ``slot_events/*.json`` -- the decision is
    never persisted into ``evaluations/`` (the underlying smoke evaluation may be
    persisted, but with a different ``decision``). These rows therefore have an
    EMPTY required ``eval_id`` column and a ``note`` saying why. When the row names a
    ``final_eval_id``/``failed_eval_id`` that *does* exist in ``evaluations/``, that
    record is used as the *linked evaluation* for the config/db/bundle/patch columns
    and is echoed in the extra ``referenced_eval_id`` column -- the required
    ``eval_id`` stays empty because the screening *rejection* is not an evaluation
    record.

Column sourcing (the point of the ledger: a config field is not a fact)
----------------------------------------------------------------------
``actual_cpus`` is the cores the *dispatched query* actually got, resolved in this
order, and ``actual_cpus_source`` names the winner so a config echo can never be
mistaken for a measurement:

1. ``meta.query_cpus_per_query`` (slurm worker: the literal ``--cpu`` dispatched)
2. ``meta.hmmer_worker_threads_per_query`` (same value, newer name)
3. ``meta.hmmer_worker_threads`` (``run_hmmer_cli`` local meta)
4. ``meta.slurm_resources_per_query.cpus`` (the job allocation)
5. ``meta.allocated_cpus`` (local ``run_hmmer_cli`` echo)
6. ``eval_config.cpus`` (a pure configuration echo -- said so by the source column)

``job_concurrency`` is ``meta.max_parallel_queries`` else ``meta.queries_per_job``.
``queries_per_job`` is ``meta.queries_per_job`` else the Slurm ``submission.json``
under the row's own ``round_*/slurm/*/`` (never borrowed from another round) else the
``eval_config`` echo.

DB hashes come from ``meta`` (facts recorded by the worker), then
``dataset_manifest.<db>.sha256`` / ``.input_sha256``, then the per-query
``query_completions[].input_sha256`` map, then -- only when the recorded path exists
on disk -- a recomputation labelled ``recomputed`` in
``{uniref,astral,scop}_sha256_source``. A different file than the one recorded is
never hashed.

``query_set_sha256`` reuses ``comparability.query_set_sha256`` (the same digest the
comparability gate builds) and ``query_set_sha256_source`` says which construction
was used, because a set built from committed ``query_completions`` is not the same
object as one built from the planned ``dataset_manifest``.

Derived columns that are NOT stored verbatim are documented where they are built
(``patch_signature``, ``patch_diff_sha256``, ``tp_fp_ratio``, ``committed_source``).

Read-only guarantee
-------------------
``LoopStore.__init__`` creates the whole directory tree, so constructing one would
write into the run directory. Instead the read surface is reused through
``_ReadOnlyStoreView`` (the same directory attribute names) plus unbound
``LoopStore`` helpers such as ``read_index`` and ``store.load_json`` /
``store.hash_file`` / ``store.sha256_text`` / ``store.canonical_json`` and
``comparability.query_set_sha256``. The list-returning ``LoopStore.iter_*`` helpers
are deliberately not used for the record directories: they call ``load_json``
unprotected, so one corrupt record would abort the whole read. ``_iter_json_records``
reuses ``store.load_json`` per file and skips (with a stderr warning) whatever is
malformed, which is what keeps the ledger exportable from a partially damaged store.

CLI
---
    python -m seq_align.c_patch_loop.ledger --run-dir RUN --out FILE.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

from ..pending_mods import normalize_run_dir
from .comparability import query_set_sha256 as _comparability_query_set_sha256
from .store import LoopStore, canonical_json, hash_file, load_json, sha256_text

__all__ = [
    "LEDGER_COLUMNS",
    "LEDGER_EXTRA_COLUMNS",
    "LEDGER_HEADER",
    "build_ledger_rows",
    "export_ledger_csv",
    "main",
]

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

#: The required columns, in the documented order. Extra provenance columns are
#: appended AFTER this block, so the required prefix is exactly this tuple.
LEDGER_COLUMNS: tuple[str, ...] = (
    "run",
    "outer_round",
    "inner_trial",
    "created_at",
    "patchset_id",
    "eval_id",
    "attempt_kind",
    "stage",
    "backend",
    "actual_cpus",
    "job_concurrency",
    "queries_per_job",
    "n_queries",
    "query_set_sha256",
    "uniref_db_path",
    "uniref_db_sha256",
    "astral_path",
    "astral_sha256",
    "scop_path",
    "scop_sha256",
    "hmmer_source_sha256",
    "patch_set_sha256",
    "bundle_id",
    "build_context",
    "changed_components",
    "patch_signature",
    "patch_diff_sha256",
    "tp",
    "fp",
    "tp_fp_ratio",
    "cost_s",
    "median_pair_speedup_fraction",
    "pair_speedup_fractions",
    "primary_candidate_cost_s",
    "decision",
    "committed",
    "compared_against_parent",
    "first_error",
)

#: Provenance/audit columns. Required because ``actual_cpus_source`` and the
#: ``*_sha256_source`` columns are demanded by the ledger contract.
LEDGER_EXTRA_COLUMNS: tuple[str, ...] = (
    "actual_cpus_source",
    "uniref_db_sha256_source",
    "astral_sha256_source",
    "scop_sha256_source",
    "query_set_sha256_source",
    "hmmer_source_sha256_source",
    "committed_source",
    "record_kind",
    "status",
    "attempt_id",
    "parent_attempt_id",
    "record_id",
    "referenced_eval_id",
    "screen_speedup_fraction",
    "screen_required_speedup_fraction",
    "source_file",
    "note",
)

LEDGER_HEADER: tuple[str, ...] = LEDGER_COLUMNS + LEDGER_EXTRA_COLUMNS

_SCREENING_DECISION = "rejected_screening_not_faster"
_SCREENING_NOTE = (
    "screening rejection persisted only in indexes/run_summary.json and/or "
    "slot_events/*.json; eval_id deliberately empty because the rejection decision "
    "is never stored in evaluations/"
)


# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------


def _present(value: Any) -> bool:
    """True when a value was genuinely recorded (not ``None`` and not empty)."""
    return value is not None and value != ""


def _first_present(*values: Any) -> Any:
    for value in values:
        if _present(value):
            return value
    return None


def _text(value: Any) -> str:
    """CSV-safe text for a recorded value; empty string for absent values."""
    if not _present(value):
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return repr(value)
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return str(value)


def _warn(message: str, stats: dict[str, Any] | None = None) -> None:
    """Print a warning to stderr and count it; never raise for store damage."""
    print(f"ledger: warning: {message}", file=sys.stderr)
    if stats is not None:
        stats["warnings"] = int(stats.get("warnings", 0)) + 1


def _skip_malformed(path: Path, kind: str, exc: BaseException, stats: dict[str, Any] | None) -> None:
    if stats is not None:
        stats["skipped_malformed_json"] = int(stats.get("skipped_malformed_json", 0)) + 1
    _warn(f"skipping malformed {kind} record {path}: {type(exc).__name__}: {exc}", stats)


def _as_float(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


# ---------------------------------------------------------------------------
# Read-only view of the store layout
# ---------------------------------------------------------------------------


class _ReadOnlyStoreView:
    """The ``LoopStore`` directory attributes without ``LoopStore.__init__``'s mkdirs.

    ``LoopStore.__init__`` calls ``mkdir(parents=True, exist_ok=True)`` on sixteen
    paths. The ledger contract is "must not write anything except the CSV you name",
    so the store is consulted through this view instead: the attribute names match
    ``LoopStore``'s (so its unbound read helpers work) but nothing is created.
    """

    def __init__(self, run_dir: str | Path) -> None:
        self.run_dir = normalize_run_dir(run_dir)
        self.root = self.run_dir / "c_patch_loop"
        for name in (
            "context_packs",
            "algorithm_plans",
            "patch_plans",
            "implementation_trials",
            "patchsets",
            "evaluations",
            "evaluation_attempts",
            "query_completions",
            "invocations",
            "invocation_summaries",
            "slot_events",
            "model_traces",
            "migrations",
            "experience",
            "indexes",
            "work",
        ):
            setattr(self, f"{name}_dir", self.root / name)


def _iter_json_records(
    directory: Path,
    *,
    kind: str,
    stats: dict[str, Any] | None,
) -> Iterator[tuple[Path, dict[str, Any]]]:
    """Yield ``(path, record)`` for every readable dict-shaped JSON file.

    Reuses ``store.load_json``; a corrupt or non-dict file is counted and warned
    about instead of aborting the export (requirement: a malformed store file must
    not stop the good rows from being exported).
    """
    if not directory.is_dir():
        return
    for path in sorted(directory.glob("*.json")):
        try:
            data = load_json(path)
        except (OSError, ValueError) as exc:  # JSONDecodeError is a ValueError
            _skip_malformed(path, kind, exc, stats)
            continue
        if not isinstance(data, dict):
            _skip_malformed(
                path,
                kind,
                ValueError(f"expected a JSON object, got {type(data).__name__}"),
                stats,
            )
            continue
        yield path, data


def _read_index(view: _ReadOnlyStoreView, name: str, stats: dict[str, Any] | None) -> Any:
    """Reuse ``LoopStore.read_index`` but survive a corrupt index file."""
    try:
        return LoopStore.read_index(view, name)
    except (OSError, ValueError) as exc:
        _warn(f"ignoring unreadable index {name}.json: {type(exc).__name__}: {exc}", stats)
        return None


def _index_rows(data: Any) -> list[dict[str, Any]]:
    """Both dict-shaped (``{"results": [...]}``) and list-shaped index files."""
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    if isinstance(data, dict):
        rows = data.get("results")
        if isinstance(rows, list):
            return [item for item in rows if isinstance(item, dict)]
    return []


# ---------------------------------------------------------------------------
# Index-derived joins
# ---------------------------------------------------------------------------


class _IndexView:
    """Everything the ledger needs that only the derived indexes know.

    These are joins/cross-checks, never the source of a metric: the required
    numbers all come from the evaluation/attempt records themselves.
    """

    def __init__(self, view: _ReadOnlyStoreView, stats: dict[str, Any] | None) -> None:
        self.rows: list[dict[str, Any]] = []
        for name in ("run_summary", "hierarchical_run_summary"):
            data = _read_index(view, name, stats)
            self.rows.extend(_index_rows(data))

        self.eval_to_inner: dict[str, Any] = {}
        self.patchset_round_to_inner: dict[tuple[str, str], Any] = {}
        self.patchset_to_inner: dict[str, Any] = {}
        self.round_inner_to_parent: dict[tuple[str, str], str] = {}
        self.patchset_to_parent: dict[str, str] = {}

        for row in self.rows:
            outer = _first_present(
                row.get("outer_idx"), row.get("round_idx"), row.get("outer_round")
            )
            inner = _first_present(row.get("inner_idx"), row.get("inner_trial"))
            patchset_id = _text(row.get("patchset_id"))
            parent = _first_present(
                row.get("parent_patchset_id"), row.get("comparison_patchset_id")
            )

            # The row's *subject* evaluation. ``final_eval_id``/``failed_eval_id``
            # are the measurements this row is about, so they define inner_trial.
            for key in ("final_eval_id", "failed_eval_id"):
                eval_id = _text(row.get(key))
                if eval_id and _present(inner):
                    self.eval_to_inner.setdefault(eval_id, inner)
            # Secondary links (static verification, parents, winners) are still the
            # same slot, so they only fill gaps.
            for key in (
                "static_eval_id",
                "winner_eval_id",
                "committed_eval_id",
                "outer_parent_eval_id",
            ):
                eval_id = _text(row.get(key))
                if eval_id and _present(inner):
                    self.eval_to_inner.setdefault(eval_id, inner)

            if patchset_id and _present(inner):
                self.patchset_to_inner.setdefault(patchset_id, inner)
                self.patchset_round_to_inner.setdefault((patchset_id, _text(outer)), inner)
            if patchset_id and _present(parent):
                self.patchset_to_parent.setdefault(patchset_id, _text(parent))
            if _present(outer) and _present(inner) and _present(parent):
                self.round_inner_to_parent.setdefault(
                    (_text(outer), _text(inner)), _text(parent)
                )

        self.accepted_eval_ids: set[str] = set()
        self.current_best_eval_ids: set[str] = set()
        if view.indexes_dir.is_dir():
            for path in sorted(view.indexes_dir.glob("*.json")):
                stem = path.stem
                if not (stem.startswith("accepted_lineage") or stem.startswith("current_best")):
                    continue
                data = _read_index(view, stem, stats)
                target = (
                    self.accepted_eval_ids
                    if stem.startswith("accepted_lineage")
                    else self.current_best_eval_ids
                )
                items = _index_rows(data)
                if not items and isinstance(data, dict):
                    items = [data]
                for item in items:
                    eval_id = _text(item.get("eval_id"))
                    if eval_id:
                        target.add(eval_id)

    def inner_for_eval(self, eval_id: str, patchset_id: str, outer_round: str) -> str:
        value = self.eval_to_inner.get(eval_id)
        if not _present(value):
            value = self.patchset_round_to_inner.get((patchset_id, outer_round))
        if not _present(value):
            value = self.patchset_to_inner.get(patchset_id)
        return _text(value)

    def inner_for_patchset(self, patchset_id: str, outer_round: str) -> str:
        value = self.patchset_round_to_inner.get((patchset_id, outer_round))
        if not _present(value):
            value = self.patchset_to_inner.get(patchset_id)
        return _text(value)

    def parent_for(self, outer_round: str, inner_trial: str, patchset_id: str) -> str:
        value = self.round_inner_to_parent.get((outer_round, inner_trial))
        if not _present(value):
            value = self.patchset_to_parent.get(patchset_id)
        return _text(value)

    def commit_state(self, eval_id: str) -> tuple[str, str]:
        if not eval_id:
            return "", ""
        if eval_id in self.accepted_eval_ids:
            return "true", "accepted_lineage"
        if eval_id in self.current_best_eval_ids:
            return "true", "current_best"
        return "false", ""


# ---------------------------------------------------------------------------
# Slurm submission lookup (queries_per_job)
# ---------------------------------------------------------------------------


class _SubmissionLookup:
    """``round_*/slurm/*/submission.json`` -> ``queries_per_job`` (real dispatch)."""

    def __init__(self, view: _ReadOnlyStoreView, stats: dict[str, Any] | None) -> None:
        self._view = view
        self._stats = stats
        self._round_dirs: dict[str, Path | None] = {}
        self._cache: dict[tuple[str, str], str] = {}

    def _round_dir(self, round_idx: str) -> Path | None:
        if round_idx in self._round_dirs:
            return self._round_dirs[round_idx]
        found: Path | None = None
        try:
            target = int(str(round_idx))
        except (TypeError, ValueError):
            target = None
        if target is not None:
            for candidate in sorted(self._view.run_dir.glob("round_*")):
                suffix = candidate.name.split("_", 1)[-1]
                try:
                    index = int(suffix)
                except ValueError:
                    continue
                if index == target:
                    found = candidate
                    break
        self._round_dirs[round_idx] = found
        return found

    def queries_per_job(self, round_idx: str, job_dir_hint: str) -> str:
        """Never guesses across rounds: one candidate, or the hinted one, or empty."""
        key = (_text(round_idx), _text(job_dir_hint))
        if key in self._cache:
            return self._cache[key]

        candidates: list[Path] = []
        if _present(key[1]):
            hinted = Path(key[1]) / "submission.json"
            if hinted.is_file():
                candidates.append(hinted)
        round_dir = self._round_dir(key[0]) if _present(key[0]) else None
        if round_dir is not None:
            slurm_dir = round_dir / "slurm"
            if slurm_dir.is_dir():
                candidates.extend(sorted(slurm_dir.glob("*/submission.json")))

        chosen: Path | None = None
        if len(candidates) == 1:
            chosen = candidates[0]
        elif len(candidates) > 1 and _present(key[1]):
            hinted_name = Path(key[1]).name
            for candidate in candidates:
                if candidate.parent.name == hinted_name:
                    chosen = candidate
                    break

        value = ""
        if chosen is not None:
            try:
                data = load_json(chosen)
            except (OSError, ValueError) as exc:
                _skip_malformed(chosen, "submission", exc, self._stats)
                data = None
            if isinstance(data, Mapping):
                value = _text(data.get("queries_per_job"))
        self._cache[key] = value
        return value


# ---------------------------------------------------------------------------
# Patchset facts
# ---------------------------------------------------------------------------


class _PatchsetFacts:
    """Changed components / signature / diff digest, from the patchset record."""

    def __init__(self, view: _ReadOnlyStoreView, stats: dict[str, Any] | None) -> None:
        self._dir = view.patchsets_dir
        self._stats = stats
        self._cache: dict[str, tuple[str, str, str]] = {}

    def _load(self, patchset_id: str) -> dict[str, Any] | None:
        if not patchset_id:
            return None
        path = self._dir / f"{patchset_id}.json"
        if not path.is_file():
            return None
        try:
            data = load_json(path)
        except (OSError, ValueError) as exc:
            _skip_malformed(path, "patchset", exc, self._stats)
            return None
        return data if isinstance(data, dict) else None

    def facts(self, patchset_id: str) -> tuple[str, str, str]:
        """``(changed_components, patch_signature, patch_diff_sha256)``.

        ``changed_components`` is the ``;``-joined ``c_target_file`` list in patch
        order (deduplicated). ``patch_diff_sha256`` is ``sha256`` of the unified diff
        texts concatenated with ``"\\n"`` in patch order (empty when no patch carries
        diff text). ``patch_signature`` is a content digest of the ordered patch
        fingerprints (component, target file, target functions, per-patch diff
        digest) -- a patchset has no stored ``patch_signature`` field, so it is
        derived from the diff content exactly as the column contract requires and
        never from anything else.
        """
        if patchset_id in self._cache:
            return self._cache[patchset_id]
        result = ("", "", "")
        record = self._load(patchset_id)
        if record is not None:
            patches = record.get("patches")
            if isinstance(patches, list) and patches:
                components: list[str] = []
                diffs: list[str] = []
                payload: list[dict[str, Any]] = []
                for patch in patches:
                    if not isinstance(patch, Mapping):
                        continue
                    target = _text(patch.get("c_target_file"))
                    if target and target not in components:
                        components.append(target)
                    diff_text = patch.get("unified_diff")
                    diff_text = diff_text if isinstance(diff_text, str) else ""
                    diffs.append(diff_text)
                    payload.append(
                        {
                            "component_name": patch.get("component_name"),
                            "c_target_file": patch.get("c_target_file"),
                            "c_target_functions": list(patch.get("c_target_functions") or []),
                            "diff_sha256": sha256_text(diff_text)
                            if diff_text
                            else _text(patch.get("diff_sha256")),
                        }
                    )
                diff_sha = sha256_text("\n".join(diffs)) if any(diffs) else ""
                signature = sha256_text(canonical_json(payload)) if payload else ""
                result = (";".join(components), signature, diff_sha)
        self._cache[patchset_id] = result
        return result


# ---------------------------------------------------------------------------
# Column resolvers
# ---------------------------------------------------------------------------


def _resolve_actual_cpus(meta: Mapping[str, Any], eval_config: Mapping[str, Any]) -> tuple[Any, str]:
    """The cores the dispatched query actually got, plus the name of that source."""
    value = meta.get("query_cpus_per_query")
    if _present(value):
        return value, "meta.query_cpus_per_query"
    value = meta.get("hmmer_worker_threads_per_query")
    if _present(value):
        return value, "meta.hmmer_worker_threads_per_query"
    value = meta.get("hmmer_worker_threads")
    if _present(value):
        return value, "meta.hmmer_worker_threads"
    resources = meta.get("slurm_resources_per_query")
    if isinstance(resources, Mapping) and _present(resources.get("cpus")):
        return resources.get("cpus"), "meta.slurm_resources_per_query.cpus"
    value = meta.get("allocated_cpus")
    if _present(value):
        return value, "meta.allocated_cpus"
    value = eval_config.get("cpus")
    if _present(value):
        return value, "eval_config.cpus"
    return None, ""


def _resolve_db_sha(
    *,
    meta: Mapping[str, Any],
    dataset_manifest: Mapping[str, Any],
    input_map: Mapping[str, Any],
    meta_key: str,
    manifest_key: str,
    input_key: str,
    path_value: str,
    fallback: tuple[str, str],
    digest_cache: dict[str, str],
    recompute: bool,
    stats: dict[str, Any] | None,
) -> tuple[str, str]:
    """Resolve one database sha256 and label exactly where it came from."""
    value = meta.get(meta_key)
    if _present(value):
        return _text(value), f"meta.{meta_key}"
    info = dataset_manifest.get(manifest_key)
    if isinstance(info, Mapping):
        for key in ("sha256", "input_sha256"):
            if _present(info.get(key)):
                return _text(info.get(key)), f"dataset_manifest.{manifest_key}.{key}"
    if _present(input_map.get(input_key)):
        return _text(input_map.get(input_key)), f"query_completions.input_sha256.{input_key}"
    if _present(fallback[0]):
        return fallback[0], fallback[1]
    if recompute and _present(path_value):
        target = Path(path_value)
        try:
            is_file = target.is_file()
        except OSError:
            is_file = False
        if is_file:
            digest = digest_cache.get(path_value)
            if digest is None:
                try:
                    digest = hash_file(target)
                except OSError as exc:
                    _warn(f"could not hash {target}: {type(exc).__name__}: {exc}", stats)
                    digest = ""
                digest_cache[path_value] = digest
            if digest:
                return digest, "recomputed"
    return "", ""


def _query_set_from_items(items: Sequence[Mapping[str, Any]]) -> str:
    """Reuse ``comparability.query_set_sha256`` so the digest matches the gate's."""
    if not items:
        return ""
    return _comparability_query_set_sha256(list(items)) or ""


def _hash_manifest(completions: Sequence[Mapping[str, Any]]) -> str:
    return _query_set_from_items(
        [
            {
                "query_id": _text(item.get("query_id")),
                "query_sha256": _text(item.get("query_sha256")),
            }
            for item in completions
        ]
    )


def _paired_block(meta: Mapping[str, Any], metric_delta: Mapping[str, Any]) -> Mapping[str, Any]:
    paired = meta.get("paired_current_best")
    if isinstance(paired, Mapping) and paired:
        return paired
    paired = metric_delta.get("paired_confirmation")
    if isinstance(paired, Mapping) and paired:
        return paired
    return {}


# ---------------------------------------------------------------------------
# Row assembly
# ---------------------------------------------------------------------------


def _blank_row() -> dict[str, str]:
    return {name: "" for name in LEDGER_HEADER}


def _base_row(
    *,
    run_name: str,
    record_kind: str,
    source_file: str,
    record_id: str,
    created_at: Any,
    outer_round: Any,
    inner_trial: Any,
    patchset_id: str,
) -> dict[str, str]:
    row = _blank_row()
    row["run"] = run_name
    row["record_kind"] = record_kind
    row["source_file"] = source_file
    row["record_id"] = record_id
    row["created_at"] = _text(created_at)
    row["outer_round"] = _text(outer_round)
    row["inner_trial"] = _text(inner_trial)
    row["patchset_id"] = patchset_id
    return row


def _apply_result_columns(
    row: dict[str, str],
    *,
    result: Mapping[str, Any],
    record: Mapping[str, Any],
) -> None:
    """tp / fp / tp_fp_ratio / cost_s -- recorded first, ratio derived if absent."""
    tp = _first_present(result.get("tp"), record.get("tp"))
    fp = _first_present(result.get("fp"), record.get("fp"))
    row["tp"] = _text(tp)
    row["fp"] = _text(fp)

    ratio = _first_present(result.get("tp_fp_ratio"), record.get("tp_fp_ratio"))
    if not _present(ratio):
        tp_value = _as_float(tp)
        fp_value = _as_float(fp)
        if tp_value is not None and fp_value is not None and (tp_value + fp_value) > 0:
            # tp/(tp+fp) is EvalSummary's own definition; derived, not invented.
            ratio = tp_value / (tp_value + fp_value)
    row["tp_fp_ratio"] = _text(ratio)

    row["cost_s"] = _text(_first_present(result.get("cost_s"), record.get("cost_s")))


def _apply_first_error(
    row: dict[str, str],
    *,
    record: Mapping[str, Any],
    result: Mapping[str, Any],
) -> None:
    errors = record.get("errors")
    if isinstance(errors, list) and errors:
        row["first_error"] = _text(errors[0])
        return
    if _present(record.get("error")):
        # screening rows record a single ``error`` string, not a list
        row["first_error"] = _text(record.get("error"))
        return
    result_errors = result.get("errors")
    if isinstance(result_errors, list) and result_errors:
        row["first_error"] = _text(result_errors[0])


def _rel(view: _ReadOnlyStoreView, path: Path) -> str:
    """Store-relative provenance string, stable across run-dir spellings."""
    try:
        return str(path.relative_to(view.run_dir))
    except ValueError:
        # Same fallback shape as the real layout, so the column stays byte-stable
        # even when the caller passed a differently-spelled run directory.
        return f"c_patch_loop/{path.parent.name}/{path.name}"


def build_ledger_rows(
    run_dir: Path | str,
    *,
    stats: dict[str, Any] | None = None,
    recompute_db_hashes: bool = True,
) -> list[dict[str, str]]:
    """Every evaluation, attempt, and index-only screening rejection in one list.

    Deterministic: the same store always produces the same rows in the same order.
    """
    view = _ReadOnlyStoreView(run_dir)
    run_name = view.run_dir.name or str(view.run_dir)
    if stats is not None:
        stats.setdefault("skipped_malformed_json", 0)
        stats.setdefault("warnings", 0)

    if not view.root.is_dir():
        _warn(f"no c_patch_loop store under {view.run_dir}", stats)
        if stats is not None:
            stats["rows"] = 0
        return []

    index = _IndexView(view, stats)
    submissions = _SubmissionLookup(view, stats)
    patchsets = _PatchsetFacts(view, stats)
    digest_cache: dict[str, str] = {}
    rows: list[dict[str, str]] = []
    attempt_kinds: dict[str, str] = {}

    # ---- pass 1: query_completions (an attempt's own per-query facts) -------
    completions_by_attempt: dict[str, list[dict[str, Any]]] = {}
    completion_by_id: dict[str, dict[str, Any]] = {}
    for _path, completion in _iter_json_records(
        view.query_completions_dir, kind="query_completion", stats=stats
    ):
        record = {
            "query_completion_id": _text(completion.get("query_completion_id")),
            "attempt_id": _text(completion.get("attempt_id")),
            "query_id": _text(completion.get("query_id")),
            "query_sha256": _text(completion.get("query_sha256")),
            "input_sha256": _mapping(completion.get("input_sha256")),
        }
        completions_by_attempt.setdefault(record["attempt_id"], []).append(record)
        if record["query_completion_id"]:
            completion_by_id[record["query_completion_id"]] = record

    def completions_for(
        attempt_ids: Sequence[str], completion_ids: Sequence[str]
    ) -> list[dict[str, Any]]:
        found: list[dict[str, Any]] = []
        for completion_id in completion_ids:
            record = completion_by_id.get(str(completion_id))
            if record is not None:
                found.append(record)
        for attempt_id in attempt_ids:
            found.extend(completions_by_attempt.get(str(attempt_id), []))
        # Deduplicate by query so a resumed-and-completed query is counted once.
        by_query: dict[str, dict[str, Any]] = {}
        for record in found:
            key = record["query_id"] or record["query_completion_id"]
            by_query.setdefault(key, record)
        return list(by_query.values())

    def input_map_for(completions: Sequence[Mapping[str, Any]]) -> dict[str, str]:
        merged: dict[str, str] = {}
        for record in completions:
            info = record.get("input_sha256")
            if not isinstance(info, Mapping):
                continue
            for key, value in info.items():
                if _present(value) and not _present(merged.get(str(key))):
                    merged[str(key)] = _text(value)
        return merged

    # ---- pass 2: evaluations (facts + the attempt -> evaluation link) ------
    eval_facts: dict[str, dict[str, Any]] = {}
    attempt_to_eval: dict[str, list[str]] = {}
    eval_by_fingerprint: dict[str, list[str]] = {}
    for path, record in _iter_json_records(view.evaluations_dir, kind="evaluation", stats=stats):
        eval_id = _text(record.get("eval_id")) or path.stem
        result = _mapping(record.get("eval_result"))
        meta = _mapping(result.get("meta"))
        eval_config = dict(_mapping(record.get("eval_config")))
        dataset_manifest = _mapping(record.get("dataset_manifest"))
        attempt_ids = [_text(item) for item in (record.get("attempt_ids") or []) if _present(item)]
        primary_attempt_id = _text(record.get("primary_attempt_id"))
        completion_ids = [
            _text(item) for item in (record.get("query_completion_ids") or []) if _present(item)
        ]
        completions = completions_for(attempt_ids, completion_ids)
        input_map = input_map_for(completions)
        patchset_id = _text(record.get("patchset_id"))
        build = _mapping(record.get("build"))
        bundle = _mapping(meta.get("hmmer_bundle"))
        source_tree_sha256 = _text(bundle.get("source_tree_sha256"))

        def eval_db(
            meta_key: str, manifest_key: str, config_key: str, input_key: str
        ) -> tuple[str, str, str]:
            path_value = _text(
                _first_present(meta.get(config_key), eval_config.get(config_key))
            )
            if not path_value:
                info = dataset_manifest.get(manifest_key)
                if isinstance(info, Mapping):
                    path_value = _text(info.get("path"))
            sha, sha_source = _resolve_db_sha(
                meta=meta,
                dataset_manifest=dataset_manifest,
                input_map=input_map,
                meta_key=meta_key,
                manifest_key=manifest_key,
                input_key=input_key,
                path_value=path_value,
                fallback=("", ""),
                digest_cache=digest_cache,
                recompute=recompute_db_hashes,
                stats=stats,
            )
            return path_value, sha, sha_source

        uniref_path, uniref_sha, uniref_source = eval_db(
            "uniref_db_sha256", "uniref_db", "uniref_db", "uniref_db"
        )
        astral_path, astral_sha, astral_source = eval_db(
            "astral_db_sha256", "astral_db", "astral_db", "astral_db"
        )
        scop_path, scop_sha, scop_source = eval_db(
            "scop_txt_sha256", "scop_txt", "scop_txt", "scop_txt"
        )

        query_set = _hash_manifest(completions)
        query_set_source = "query_completions" if query_set else ""
        if not query_set:
            query_files = dataset_manifest.get("query_files")
            if isinstance(query_files, list) and query_files:
                items = []
                for item in query_files:
                    if not isinstance(item, Mapping):
                        continue
                    items.append(
                        {
                            "query_id": _text(item.get("query_id"))
                            or Path(_text(item.get("path"))).name,
                            "query_sha256": _text(item.get("sha256")),
                        }
                    )
                query_set = _query_set_from_items(items)
                query_set_source = "dataset_manifest.query_files" if query_set else ""
        if not query_set:
            names = dataset_manifest.get("queries")
            if isinstance(names, list) and names:
                query_set = _query_set_from_items(
                    [{"query_id": _text(name), "query_sha256": ""} for name in names]
                )
                query_set_source = "dataset_manifest.queries(names_only)" if query_set else ""

        hmmer_source_sha256 = _text(
            _first_present(
                source_tree_sha256,
                meta.get("hmmer_source_sha256"),
                eval_config.get("hmmer_source_sha256"),
            )
        )
        hmmer_source_source = ""
        if _present(source_tree_sha256):
            hmmer_source_source = "meta.hmmer_bundle.source_tree_sha256"
        elif _present(meta.get("hmmer_source_sha256")):
            hmmer_source_source = "meta.hmmer_source_sha256"
        elif _present(eval_config.get("hmmer_source_sha256")):
            hmmer_source_source = "eval_config.hmmer_source_sha256"
        if not hmmer_source_sha256:
            for key in ("source_tree_sha256", "hmmer_source_sha256"):
                value = _text(build.get(key))
                if value:
                    hmmer_source_sha256 = value
                    hmmer_source_source = f"build.{key}"
                    break
        if not hmmer_source_sha256:
            # A local (non-bundle) run records the source identity as a per-file
            # ``source_hashes`` map, not one tree hash. Reducing it needs a
            # construction, so the digest is labelled as one -- it is never presented
            # as the bundle's ``source_tree_sha256``.
            source_hashes = record.get("source_hashes")
            if isinstance(source_hashes, Mapping) and source_hashes:
                hmmer_source_sha256 = sha256_text(canonical_json(source_hashes))
                hmmer_source_source = "record.source_hashes(digest_of_recorded_file_hashes)"

        metric_delta = _mapping(record.get("metric_delta"))
        paired = dict(_paired_block(meta, metric_delta))
        errors = record.get("errors")

        eval_facts[eval_id] = {
            "eval_id": eval_id,
            "patchset_id": patchset_id,
            "round_idx": record.get("round_idx"),
            "stage": _text(record.get("stage")),
            "status": _text(record.get("status")),
            "decision": _text(record.get("decision")),
            "created_at": record.get("created_at"),
            "eval_config": eval_config,
            "dataset_manifest": dataset_manifest,
            "primary_attempt_id": primary_attempt_id,
            "attempt_ids": attempt_ids,
            "requested_queries": record.get("requested_queries"),
            "n_seqs": _first_present(result.get("n_seqs"), meta.get("query_count")),
            "query_results_count": len(meta.get("query_results"))
            if isinstance(meta.get("query_results"), list)
            else 0,
            "query_set": query_set,
            "query_set_source": query_set_source,
            "meta": meta,
            "input_map": input_map,
            "path": path,
            "uniref_path": uniref_path,
            "uniref_sha": uniref_sha,
            "uniref_source": uniref_source,
            "astral_path": astral_path,
            "astral_sha": astral_sha,
            "astral_source": astral_source,
            "scop_path": scop_path,
            "scop_sha": scop_sha,
            "scop_source": scop_source,
            "hmmer_source_sha256": hmmer_source_sha256,
            "hmmer_source_source": hmmer_source_source,
            "patch_set_sha256": _text(
                _first_present(bundle.get("patch_set_sha256"), build.get("patch_set_sha256"))
            ),
            "bundle_id": _text(
                _first_present(bundle.get("bundle_id"), build.get("bundle_id"))
            ),
            "build_context": _text(
                _first_present(bundle.get("build_context"), build.get("build_context"))
            ),
            "paired": paired,
            "paired_parent": _text(paired.get("current_best_patchset_id")),
            "first_error": _text(errors[0]) if isinstance(errors, list) and errors else "",
            "result": result,
        }
        for attempt_id in attempt_ids:
            attempt_to_eval.setdefault(attempt_id, []).append(eval_id)
        fingerprint = _text(record.get("evaluation_fingerprint"))
        if fingerprint:
            eval_by_fingerprint.setdefault(fingerprint, []).append(eval_id)

    # ---- attempt rows: one per persisted attempt record --------------------
    for path, record in _iter_json_records(
        view.evaluation_attempts_dir, kind="evaluation_attempt", stats=stats
    ):
        attempt_id = _text(record.get("attempt_id")) or path.stem
        attempt_kind = _text(record.get("attempt_kind"))
        attempt_kinds[attempt_id] = attempt_kind
        patchset_id = _text(record.get("patchset_id"))
        outer_round = record.get("round_idx")
        inner_trial = index.inner_for_patchset(patchset_id, _text(outer_round))
        linked_eval_ids = sorted(attempt_to_eval.get(attempt_id, []))
        if not linked_eval_ids:
            # A ``started`` attempt is never named by an evaluation's ``attempt_ids``
            # (only its terminal child is), but both records carry the same recorded
            # ``evaluation_fingerprint``. Use that identity only when it names exactly
            # one evaluation; otherwise leave the row unlinked rather than guess.
            fingerprint = _text(record.get("evaluation_fingerprint"))
            candidates = sorted(set(eval_by_fingerprint.get(fingerprint, [])))
            if len(candidates) == 1:
                linked_eval_ids = candidates
        linked_eval = eval_facts.get(linked_eval_ids[0]) if linked_eval_ids else None

        row = _base_row(
            run_name=run_name,
            record_kind="attempt",
            source_file=_rel(view, path),
            record_id=attempt_id,
            created_at=record.get("created_at"),
            outer_round=outer_round,
            inner_trial=inner_trial,
            patchset_id=patchset_id,
        )
        # The store links evaluation -> attempt (``attempt_ids``); the reverse is a
        # real recorded link when exactly one evaluation names this attempt.
        row["eval_id"] = (
            ";".join(linked_eval_ids)
            if len(linked_eval_ids) > 1
            else (linked_eval_ids[0] if linked_eval_ids else "")
        )
        row["attempt_id"] = attempt_id
        row["parent_attempt_id"] = _text(record.get("parent_attempt_id"))
        row["attempt_kind"] = attempt_kind
        row["stage"] = _text(record.get("stage"))
        row["status"] = _text(record.get("status"))

        aggregate = _mapping(record.get("aggregate_result"))
        meta = _mapping(aggregate.get("meta"))
        eval_config = dict(linked_eval["eval_config"]) if linked_eval else {}
        dataset_manifest = linked_eval["dataset_manifest"] if linked_eval else {}

        completion_ids = [str(item) for item in (record.get("query_completion_ids") or [])]
        completions = completions_for([attempt_id], completion_ids)
        input_map = input_map_for(completions)
        if not input_map and linked_eval is not None:
            input_map = dict(linked_eval["input_map"])

        row["backend"] = _text(_first_present(meta.get("backend"), eval_config.get("backend")))
        actual_cpus, actual_source = _resolve_actual_cpus(meta, eval_config)
        row["actual_cpus"] = _text(actual_cpus)
        row["actual_cpus_source"] = actual_source
        row["job_concurrency"] = _text(
            _first_present(meta.get("max_parallel_queries"), meta.get("queries_per_job"))
        )
        row["queries_per_job"] = _text(
            _first_present(
                meta.get("queries_per_job"),
                submissions.queries_per_job(_text(outer_round), _text(meta.get("job_dir"))),
                eval_config.get("queries_per_job"),
            )
        )
        n_queries = _first_present(
            record.get("requested_queries"), aggregate.get("n_seqs"), meta.get("query_count")
        )
        if not _present(n_queries):
            query_results = meta.get("query_results")
            if isinstance(query_results, list) and query_results:
                n_queries = len(query_results)
        row["n_queries"] = _text(n_queries)

        query_set = _hash_manifest(completions)
        query_set_source = "query_completions" if query_set else ""
        if not query_set and linked_eval is not None:
            query_set = linked_eval["query_set"]
            query_set_source = (
                f"linked_eval:{linked_eval['query_set_source']}" if query_set else ""
            )
        row["query_set_sha256"] = query_set
        row["query_set_sha256_source"] = query_set_source

        uniref_path = _text(
            _first_present(
                meta.get("uniref_db"),
                eval_config.get("uniref_db"),
                linked_eval["uniref_path"] if linked_eval else None,
            )
        )
        astral_path = _text(
            _first_present(
                meta.get("astral_db"),
                eval_config.get("astral_db"),
                linked_eval["astral_path"] if linked_eval else None,
            )
        )
        scop_path = _text(
            _first_present(
                meta.get("scop_txt"),
                eval_config.get("scop_txt"),
                linked_eval["scop_path"] if linked_eval else None,
            )
        )

        def attempt_db(
            meta_key: str,
            manifest_key: str,
            config_key: str,
            input_key: str,
            fallback: tuple[str, str],
        ) -> tuple[str, str]:
            path_value = _text(
                _first_present(meta.get(config_key), eval_config.get(config_key))
            )
            if not path_value:
                info = dataset_manifest.get(manifest_key)
                if isinstance(info, Mapping):
                    path_value = _text(info.get("path"))
            return _resolve_db_sha(
                meta=meta,
                dataset_manifest=dataset_manifest,
                input_map=input_map,
                meta_key=meta_key,
                manifest_key=manifest_key,
                input_key=input_key,
                path_value=path_value,
                fallback=fallback,
                digest_cache=digest_cache,
                recompute=recompute_db_hashes,
                stats=stats,
            )

        fallback_uniref = (
            (linked_eval["uniref_sha"], linked_eval["uniref_source"]) if linked_eval else ("", "")
        )
        fallback_astral = (
            (linked_eval["astral_sha"], linked_eval["astral_source"]) if linked_eval else ("", "")
        )
        fallback_scop = (
            (linked_eval["scop_sha"], linked_eval["scop_source"]) if linked_eval else ("", "")
        )
        uniref_sha, uniref_source = attempt_db(
            "uniref_db_sha256", "uniref_db", "uniref_db", "uniref_db", fallback_uniref
        )
        astral_sha, astral_source = attempt_db(
            "astral_db_sha256", "astral_db", "astral_db", "astral_db", fallback_astral
        )
        scop_sha, scop_source = attempt_db(
            "scop_txt_sha256", "scop_txt", "scop_txt", "scop_txt", fallback_scop
        )
        row["uniref_db_path"] = uniref_path
        row["uniref_db_sha256"] = uniref_sha
        row["uniref_db_sha256_source"] = uniref_source
        row["astral_path"] = astral_path
        row["astral_sha256"] = astral_sha
        row["astral_sha256_source"] = astral_source
        row["scop_path"] = scop_path
        row["scop_sha256"] = scop_sha
        row["scop_sha256_source"] = scop_source

        bundle = _mapping(meta.get("hmmer_bundle"))
        build = _mapping(record.get("build"))
        source_tree_sha256 = _text(bundle.get("source_tree_sha256"))
        row["hmmer_source_sha256"] = _text(
            _first_present(
                source_tree_sha256,
                meta.get("hmmer_source_sha256"),
                eval_config.get("hmmer_source_sha256"),
                linked_eval["hmmer_source_sha256"] if linked_eval else None,
            )
        )
        if _present(source_tree_sha256):
            row["hmmer_source_sha256_source"] = "meta.hmmer_bundle.source_tree_sha256"
        elif _present(meta.get("hmmer_source_sha256")):
            row["hmmer_source_sha256_source"] = "meta.hmmer_source_sha256"
        elif _present(eval_config.get("hmmer_source_sha256")):
            row["hmmer_source_sha256_source"] = "eval_config.hmmer_source_sha256"
        elif linked_eval and linked_eval["hmmer_source_source"]:
            row["hmmer_source_sha256_source"] = (
                f"linked_eval:{linked_eval['hmmer_source_source']}"
            )

        row["patch_set_sha256"] = _text(
            _first_present(
                bundle.get("patch_set_sha256"),
                build.get("patch_set_sha256"),
                linked_eval["patch_set_sha256"] if linked_eval else None,
            )
        )
        row["bundle_id"] = _text(
            _first_present(
                bundle.get("bundle_id"),
                build.get("bundle_id"),
                linked_eval["bundle_id"] if linked_eval else None,
            )
        )
        row["build_context"] = _text(
            _first_present(
                bundle.get("build_context"),
                build.get("build_context"),
                linked_eval["build_context"] if linked_eval else None,
            )
        )

        changed, signature, diff_sha = patchsets.facts(patchset_id)
        row["changed_components"] = changed
        row["patch_signature"] = signature
        row["patch_diff_sha256"] = diff_sha

        _apply_result_columns(row, result=aggregate, record=record)
        # attempts carry no decision and no paired block: derive nothing.
        row["decision"] = ""
        row["committed"] = ""
        row["committed_source"] = ""
        row["compared_against_parent"] = index.parent_for(
            _text(outer_round), inner_trial, patchset_id
        )
        _apply_first_error(row, record=record, result=aggregate)
        rows.append(row)

    # ---- evaluation rows (attempt_kinds is now complete) -------------------
    for eval_id, facts in eval_facts.items():
        outer_round = facts["round_idx"]
        inner_trial = index.inner_for_eval(eval_id, facts["patchset_id"], _text(outer_round))
        row = _base_row(
            run_name=run_name,
            record_kind="evaluation",
            source_file=_rel(view, facts["path"]),
            record_id=eval_id,
            created_at=facts["created_at"],
            outer_round=outer_round,
            inner_trial=inner_trial,
            patchset_id=facts["patchset_id"],
        )
        row["eval_id"] = eval_id
        row["stage"] = facts["stage"]
        row["status"] = facts["status"]
        row["decision"] = facts["decision"]
        # attempt_kind lives on the attempt record; the evaluation names its primary
        # attempt explicitly, so this is a recorded link, not a guess.
        row["attempt_kind"] = attempt_kinds.get(facts["primary_attempt_id"], "")
        row["backend"] = _text(
            _first_present(facts["meta"].get("backend"), facts["eval_config"].get("backend"))
        )

        meta = facts["meta"]
        actual_cpus, actual_source = _resolve_actual_cpus(meta, facts["eval_config"])
        row["actual_cpus"] = _text(actual_cpus)
        row["actual_cpus_source"] = actual_source
        row["job_concurrency"] = _text(
            _first_present(meta.get("max_parallel_queries"), meta.get("queries_per_job"))
        )
        row["queries_per_job"] = _text(
            _first_present(
                meta.get("queries_per_job"),
                submissions.queries_per_job(_text(outer_round), _text(meta.get("job_dir"))),
                facts["eval_config"].get("queries_per_job"),
            )
        )
        n_queries = _first_present(
            facts["requested_queries"], facts["n_seqs"], meta.get("query_count")
        )
        if not _present(n_queries) and facts["query_results_count"]:
            n_queries = facts["query_results_count"]
        row["n_queries"] = _text(n_queries)

        row["query_set_sha256"] = facts["query_set"]
        row["query_set_sha256_source"] = facts["query_set_source"]
        row["uniref_db_path"] = facts["uniref_path"]
        row["uniref_db_sha256"] = facts["uniref_sha"]
        row["uniref_db_sha256_source"] = facts["uniref_source"]
        row["astral_path"] = facts["astral_path"]
        row["astral_sha256"] = facts["astral_sha"]
        row["astral_sha256_source"] = facts["astral_source"]
        row["scop_path"] = facts["scop_path"]
        row["scop_sha256"] = facts["scop_sha"]
        row["scop_sha256_source"] = facts["scop_source"]
        row["hmmer_source_sha256"] = facts["hmmer_source_sha256"]
        row["hmmer_source_sha256_source"] = facts["hmmer_source_source"]
        row["patch_set_sha256"] = facts["patch_set_sha256"]
        row["bundle_id"] = facts["bundle_id"]
        row["build_context"] = facts["build_context"]

        changed, signature, diff_sha = patchsets.facts(facts["patchset_id"])
        row["changed_components"] = changed
        row["patch_signature"] = signature
        row["patch_diff_sha256"] = diff_sha

        _apply_result_columns(row, result=facts["result"], record={})
        paired = facts["paired"]
        row["median_pair_speedup_fraction"] = _text(
            _first_present(
                paired.get("median_pair_speedup_fraction"), paired.get("speedup_fraction")
            )
        )
        fractions = paired.get("pair_speedup_fractions")
        row["pair_speedup_fractions"] = (
            json.dumps(fractions, ensure_ascii=False, separators=(",", ":"))
            if isinstance(fractions, list)
            else ""
        )
        row["primary_candidate_cost_s"] = _text(paired.get("primary_candidate_cost_s"))

        row["compared_against_parent"] = _text(
            _first_present(
                facts["paired_parent"],
                index.parent_for(_text(outer_round), inner_trial, facts["patchset_id"]),
            )
        )
        committed, committed_source = index.commit_state(eval_id)
        row["committed"] = committed
        row["committed_source"] = committed_source
        row["first_error"] = facts["first_error"]
        rows.append(row)

    # ---- screening rejections that exist only in the indexes/events --------
    screening_seen: set[tuple[str, ...]] = set()

    def emit_screening(row_data: Mapping[str, Any], source_file: str) -> None:
        if not (
            row_data.get("screening") is True
            or _text(row_data.get("decision")) == _SCREENING_DECISION
        ):
            return
        outer_round = _first_present(
            row_data.get("outer_idx"), row_data.get("round_idx"), row_data.get("outer_round")
        )
        inner_trial = _first_present(row_data.get("inner_idx"), row_data.get("inner_trial"))
        patchset_id = _text(row_data.get("patchset_id"))
        referenced = _text(
            _first_present(row_data.get("final_eval_id"), row_data.get("failed_eval_id"))
        )
        key = (
            _text(outer_round),
            _text(inner_trial),
            patchset_id,
            referenced,
            _text(row_data.get("decision")),
        )
        if key in screening_seen:
            return
        screening_seen.add(key)

        linked = eval_facts.get(referenced)
        row = _base_row(
            run_name=run_name,
            record_kind="screening_rejection",
            source_file=source_file,
            record_id="|".join(key),
            created_at=row_data.get("created_at"),
            outer_round=outer_round,
            inner_trial=inner_trial,
            patchset_id=patchset_id,
        )
        # eval_id stays EMPTY: the rejection decision is not an evaluation record.
        row["referenced_eval_id"] = referenced
        row["status"] = _text(row_data.get("status"))
        row["decision"] = _text(row_data.get("decision")) or _SCREENING_DECISION
        row["note"] = _SCREENING_NOTE
        row["screen_speedup_fraction"] = _text(row_data.get("screen_speedup_fraction"))
        row["screen_required_speedup_fraction"] = _text(
            row_data.get("screen_required_speedup_fraction")
        )

        if linked is not None:
            row["attempt_kind"] = attempt_kinds.get(linked["primary_attempt_id"], "")
            row["stage"] = linked["stage"]
            row["backend"] = _text(
                _first_present(linked["meta"].get("backend"), linked["eval_config"].get("backend"))
            )
            actual_cpus, actual_source = _resolve_actual_cpus(
                linked["meta"], linked["eval_config"]
            )
            row["actual_cpus"] = _text(actual_cpus)
            row["actual_cpus_source"] = f"linked_eval:{actual_source}" if actual_source else ""
            row["job_concurrency"] = _text(
                _first_present(
                    linked["meta"].get("max_parallel_queries"),
                    linked["meta"].get("queries_per_job"),
                )
            )
            row["queries_per_job"] = _text(
                _first_present(
                    linked["meta"].get("queries_per_job"),
                    submissions.queries_per_job(
                        _text(outer_round), _text(linked["meta"].get("job_dir"))
                    ),
                )
            )
            row["n_queries"] = _text(
                _first_present(
                    linked["requested_queries"],
                    linked["n_seqs"],
                    linked["meta"].get("query_count"),
                )
            )
            row["query_set_sha256"] = linked["query_set"]
            row["query_set_sha256_source"] = (
                f"linked_eval:{linked['query_set_source']}" if linked["query_set_source"] else ""
            )
            row["uniref_db_path"] = linked["uniref_path"]
            row["uniref_db_sha256"] = linked["uniref_sha"]
            row["uniref_db_sha256_source"] = linked["uniref_source"]
            row["astral_path"] = linked["astral_path"]
            row["astral_sha256"] = linked["astral_sha"]
            row["astral_sha256_source"] = linked["astral_source"]
            row["scop_path"] = linked["scop_path"]
            row["scop_sha256"] = linked["scop_sha"]
            row["scop_sha256_source"] = linked["scop_source"]
            row["hmmer_source_sha256"] = linked["hmmer_source_sha256"]
            row["hmmer_source_sha256_source"] = linked["hmmer_source_source"]
            row["patch_set_sha256"] = linked["patch_set_sha256"]
            row["bundle_id"] = linked["bundle_id"]
            row["build_context"] = linked["build_context"]
            # The screening row's own metrics are the evidence the rejection was made
            # on, so they win; the referenced evaluation only fills what the row lacks.
            _apply_result_columns(row, result=row_data, record=linked["result"])
        else:
            _apply_result_columns(row, result=row_data, record=row_data)

        changed, signature, diff_sha = patchsets.facts(patchset_id)
        row["changed_components"] = changed
        row["patch_signature"] = signature
        row["patch_diff_sha256"] = diff_sha

        row["compared_against_parent"] = _text(
            _first_present(
                row_data.get("parent_patchset_id"),
                index.parent_for(_text(outer_round), _text(inner_trial), patchset_id),
            )
        )
        # A screening rejection can never be committed; the row's own decision says so.
        row["committed"] = "false"
        row["committed_source"] = "screening_rejection"
        _apply_first_error(row, record=row_data, result=linked["result"] if linked else row_data)
        rows.append(row)

    for name in ("run_summary", "hierarchical_run_summary"):
        data = _read_index(view, name, stats)
        for item in _index_rows(data):
            emit_screening(item, f"c_patch_loop/indexes/{name}.json")

    for path, event in _iter_json_records(view.slot_events_dir, kind="slot_event", stats=stats):
        relative = _rel(view, path)
        for field in ("result", "details"):
            payload = event.get(field)
            if isinstance(payload, Mapping):
                emit_screening(payload, relative)

    # ---- stable order ------------------------------------------------------
    def sort_key(row: Mapping[str, str]) -> tuple[Any, ...]:
        def numeric(value: Any) -> tuple[int, int]:
            text = _text(value).strip()
            if not text:
                return 1, 0  # empty numerics LAST
            try:
                return 0, int(text)
            except ValueError:
                return 1, 0

        outer_flag, outer_value = numeric(row.get("outer_round"))
        inner_flag, inner_value = numeric(row.get("inner_trial"))
        return (
            row.get("run", ""),
            outer_flag,
            outer_value,
            inner_flag,
            inner_value,
            row.get("created_at", ""),
            row.get("attempt_kind", ""),
            row.get("patchset_id", ""),
            row.get("record_kind", ""),
            row.get("record_id", ""),
            row.get("source_file", ""),
        )

    rows.sort(key=sort_key)
    if stats is not None:
        stats["rows"] = len(rows)
    return rows


def export_ledger_csv(
    run_dir: Path | str,
    out_path: Path | str,
    *,
    stats: dict[str, Any] | None = None,
    recompute_db_hashes: bool = True,
    rows: Sequence[Mapping[str, Any]] | None = None,
) -> Path:
    """Write the derived ledger view to ``out_path`` and return that path.

    Writes exactly one file (plus, if needed, its parent directory); it never
    touches the run directory.
    """
    target = Path(out_path)
    if rows is None:
        rows = build_ledger_rows(
            run_dir, stats=stats, recompute_db_hashes=recompute_db_hashes
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(LEDGER_HEADER), lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({name: _text(row.get(name)) for name in LEDGER_HEADER})
    if stats is not None:
        stats["out_path"] = str(target)
    return target


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m seq_align.c_patch_loop.ledger",
        description=(
            "Export an auditable CSV ledger of every evaluation/attempt/screening "
            "record in a C-patch run directory, regenerated from the store."
        ),
    )
    parser.add_argument("--run-dir", required=True, help="Run dir (contains c_patch_loop/)")
    parser.add_argument("--out", required=True, help="Destination CSV path")
    parser.add_argument(
        "--no-recompute-db-hashes",
        action="store_true",
        help="Never hash an on-disk database; leave the sha empty when the store lacks it",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    stats: dict[str, Any] = {}
    out = export_ledger_csv(
        Path(args.run_dir),
        Path(args.out),
        stats=stats,
        recompute_db_hashes=not args.no_recompute_db_hashes,
    )
    print(
        f"ledger: wrote {stats.get('rows', 0)} rows to {out} "
        f"(skipped {stats.get('skipped_malformed_json', 0)} malformed record(s))",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI hook
    raise SystemExit(main())
