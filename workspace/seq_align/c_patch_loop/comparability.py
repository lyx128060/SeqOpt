"""Refuse to compare cost numbers that were not measured the same way.

Every acceptance decision in this loop is a subtraction or a ratio of two wall
clocks. Two numbers are only comparable when *everything that changes the
measurement* is identical: backend, partition, the cores a query actually got,
how many queries shared the job, the stage, the exact query set, the iteration
budget, and the search database. The 2026-10-09 audit found comparisons being
made across local and Slurm backends, across 12h/24h configurations, and with a
baseline that a profiling pass had overwritten -- each one silently produced a
verdict from numbers that were never comparable.

This module makes that refusal explicit. ``comparison_fingerprint`` reduces an
evaluation config/meta to the fields that must match; ``compare_costs`` raises
``IncomparableMeasurement`` instead of returning a speedup when they do not.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

SCHEMA_VERSION = "seqalign_comparability.v1"

# Every key here is part of "how this number was produced". Keep them flat and
# hashable so a mismatch can be reported field by field.
REQUIRED_KEYS: tuple[str, ...] = (
    # Facts every evaluation must record, on every backend.
    "backend",
    "allocated_cpus",
    "query_threads",
    "stage",
    "query_set_sha256",
    "max_iter",
    "uniref_db_path",
    "uniref_db_sha256",
    "astral_db_path",
    "astral_db_sha256",
    "scop_txt_path",
    "scop_txt_sha256",
)

# Compared, but legitimately absent on some backends. These are properties of a
# *job*: a local evaluation has no partition, no in-job concurrency and no compiled
# bundle, and its record says so by omitting them. Requiring them would make every
# local measurement permanently incomparable with every other one -- including with
# another local measurement that also has none -- which is not what "refuse when the
# conditions differ" means. Their absence is still part of the comparison: one side
# having a partition and the other not is a mismatch.
#
# The absence cannot hide a real difference, because ``backend`` is required and
# compared: two records with no partition are both non-Slurm, and ``PARTITION_REQUIRED_FOR``
# below makes a Slurm record with no partition a refusal in its own right.
OPTIONAL_BUT_COMPARED: tuple[str, ...] = (
    "partition",
    "queries_in_job",
    "hmmer_source_sha256",
    "eval_repeats",
    "paired_current_best_repeats",
    "bundle_id",
    "build_context",
)

# A backend whose runs always take place inside an allocated job must record the
# job's own properties. Missing them there is an incomplete record, not a backend
# that has none, so it is refused.
PARTITION_REQUIRED_FOR: tuple[str, ...] = ("slurm",)

class IncomparableMeasurement(RuntimeError):
    """Two cost numbers must not be compared; the message names every difference."""

    def __init__(self, message: str, *, mismatches: Mapping[str, Any]):
        super().__init__(message)
        self.mismatches = dict(mismatches)


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def query_set_sha256(query_manifest: list[Mapping[str, Any]] | None) -> str | None:
    """Fingerprint the exact per-query set (sorted query ids + file hashes)."""
    if not query_manifest:
        return None
    items = sorted(
        (
            str(item.get("query_id") or ""),
            str(item.get("query_sha256") or ""),
        )
        for item in query_manifest
    )
    return _sha256_text(json.dumps(items, separators=(",", ":")))


def file_sha256(path: str | Path | None, *, digest_cache: Any = None) -> str | None:
    """Hash a database file, reusing the caller's digest cache when given one."""
    if not path:
        return None
    target = Path(str(path))
    if not target.is_file():
        return None
    if digest_cache is not None and hasattr(digest_cache, "sha256"):
        digest = digest_cache.sha256(target)
        if digest:
            return str(digest)
    digest = hashlib.sha256()
    with target.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def comparison_fingerprint(
    *,
    eval_config: Mapping[str, Any] | None,
    meta: Mapping[str, Any] | None = None,
    query_manifest: list[Mapping[str, Any]] | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the comparability fingerprint of one measured cost number.

    Values come from the evaluation record itself (``eval_config`` / ``meta``),
    never from a configuration file: the audit's rule is that only what the
    evaluation actually recorded may be treated as fact.

    ``allocated_cpus`` and ``query_threads`` are deliberately two fields. The first
    is the job's allocation, the second is the number of cores the query itself
    actually got (the literal HMMER ``--cpu``). A record can have 45 allocated and
    8 per query; those are not interchangeable and a change to either invalidates a
    comparison.
    """
    cfg = dict(eval_config or {})
    meta = dict(meta or {})
    slurm_resources = meta.get("slurm_resources_per_query")
    slurm_resources = slurm_resources if isinstance(slurm_resources, Mapping) else {}
    bundle = meta.get("hmmer_bundle") if isinstance(meta.get("hmmer_bundle"), Mapping) else {}

    def _first_present(*candidates: Any) -> Any:
        """First candidate that was actually recorded.

        Deliberately not ``a or b or c``: ``0`` is a real, recorded, and alarming
        value here. HMMER's ``--cpu 0`` means "detect the machine and use every
        core", so a record that says ``hmmer_worker_threads_per_query: 0`` must
        fingerprint as 0 -- treating it as absent is how a self-detecting run gets
        silently compared against an 8-thread one.
        """
        for value in candidates:
            if value is not None:
                return value
        return None

    fingerprint: dict[str, Any] = {
        "backend": cfg.get("backend"),
        "partition": _first_present(cfg.get("partition"), meta.get("partition"), slurm_resources.get("partition")),
        # Job allocation: what sbatch reserved.
        "allocated_cpus": _first_present(
            meta.get("job_cpus"),
            slurm_resources.get("cpus"),
            cfg.get("cpus"),
        ),
        # Cores the *query* actually got: HMMER worker threads, recorded by the
        # worker that ran it. Distinct from the job allocation by design.
        "query_threads": _first_present(
            meta.get("query_cpus_per_query"),
            meta.get("hmmer_worker_threads_per_query"),
            meta.get("hmmer_worker_threads"),
            meta.get("query_threads"),
            cfg.get("cpus"),
        ),
        "queries_in_job": _first_present(
            meta.get("queries_per_job"),
            meta.get("queries_in_job"),
            meta.get("max_parallel_queries"),
        ),
        "stage": cfg.get("stage"),
        "query_set_sha256": query_set_sha256(query_manifest),
        "max_iter": cfg.get("max_iter"),
        "uniref_db_path": cfg.get("uniref_db"),
        "uniref_db_sha256": meta.get("uniref_db_sha256"),
        "astral_db_path": cfg.get("astral_db"),
        "astral_db_sha256": meta.get("astral_db_sha256"),
        "scop_txt_path": cfg.get("scop_txt"),
        "scop_txt_sha256": meta.get("scop_txt_sha256"),
        "hmmer_source_sha256": _first_present(
            cfg.get("hmmer_source_sha256"),
            meta.get("hmmer_source_sha256"),
            bundle.get("source_tree_sha256"),
        ),
        "eval_repeats": cfg.get("eval_repeats"),
        "paired_current_best_repeats": cfg.get("paired_current_best_repeats"),
        "bundle_id": _first_present(meta.get("bundle_id"), bundle.get("bundle_id")),
        "build_context": _first_present(meta.get("build_context"), bundle.get("build_context")),
    }
    for key, value in (extra or {}).items():
        fingerprint[key] = value
    return fingerprint


def compare_costs(
    reference: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    reference_cost_s: float,
    candidate_cost_s: float,
) -> float:
    """Return the speedup fraction, or raise when the two numbers are not comparable.

    ``reference``/``candidate`` are comparability fingerprints (see
    ``comparison_fingerprint``). The returned value is
    ``(reference - candidate) / reference`` -- the same convention the acceptance
    gate uses -- and is only ever computed after every required field matched.
    """
    mismatches: dict[str, Any] = {}
    for key in (*REQUIRED_KEYS, *OPTIONAL_BUT_COMPARED):
        left = reference.get(key)
        right = candidate.get(key)
        if left != right:
            mismatches[key] = {"reference": left, "candidate": right}
    missing_reference = [key for key in REQUIRED_KEYS if reference.get(key) in (None, "")]
    missing_candidate = [key for key in REQUIRED_KEYS if candidate.get(key) in (None, "")]
    # A Slurm run is always inside an allocated job, so its job-scoped fields must be
    # recorded; their absence there is an incomplete record rather than a backend
    # that has none.
    for label, side, missing in (
        ("reference", reference, missing_reference),
        ("candidate", candidate, missing_candidate),
    ):
        backend = str(side.get("backend") or "").lower()
        if any(marker in backend for marker in PARTITION_REQUIRED_FOR):
            missing.extend(
                key
                for key in ("partition", "queries_in_job")
                if side.get(key) in (None, "") and key not in missing
            )
    if missing_reference or missing_candidate:
        mismatches["_missing_required"] = {
            "reference": missing_reference,
            "candidate": missing_candidate,
        }
    if mismatches:
        detail = "; ".join(
            f"{key}: reference={value!r} candidate={value!r}"
            if isinstance(value, dict) and "reference" in value
            else f"{key}: missing reference={value['reference']} candidate={value['candidate']}"
            for key, value in mismatches.items()
            if isinstance(value, dict)
        )
        raise IncomparableMeasurement(
            "refusing to compare costs measured under different conditions: " + detail,
            mismatches=mismatches,
        )
    if not reference_cost_s or reference_cost_s <= 0.0:
        raise IncomparableMeasurement(
            "refusing to compare against a non-positive reference cost",
            mismatches={"_reference_cost_s": {"reference": reference_cost_s, "candidate": candidate_cost_s}},
        )
    return (float(reference_cost_s) - float(candidate_cost_s)) / float(reference_cost_s)


def summary_fingerprint(summary: Any) -> dict[str, Any]:
    """Extract the measured-conditions fingerprint carried by an eval summary.

    Accepts an ``EvalSummary`` (fingerprint attribute), a raw ``EvalResult`` (which
    keeps it in ``meta``, because that is where the worker-side code records it), or
    a plain mapping of either shape. Returns ``{}`` when the summary has none, which
    is itself a refusal reason: the number's conditions are unknown.
    """
    if summary is None:
        return {}
    if isinstance(summary, Mapping):
        value = summary.get("comparability")
        if not isinstance(value, Mapping):
            meta = summary.get("meta")
            value = meta.get("comparability") if isinstance(meta, Mapping) else None
    else:
        value = getattr(summary, "comparability", None)
        if not isinstance(value, Mapping):
            meta = getattr(summary, "meta", None)
            value = meta.get("comparability") if isinstance(meta, Mapping) else None
    return dict(value) if isinstance(value, Mapping) else {}


def compare_summaries(reference: Any, candidate: Any) -> float:
    """Speedup of ``candidate`` over ``reference``, or refuse.

    The single entry point every acceptance decision should use. It reads the
    fingerprints off the two summaries -- never off live configuration -- and
    raises :class:`IncomparableMeasurement` naming every differing field.
    """
    reference_cost = float(getattr(reference, "cost_s", 0.0) or 0.0)
    candidate_cost = float(getattr(candidate, "cost_s", 0.0) or 0.0)
    if isinstance(reference, Mapping):
        reference_cost = float(reference.get("cost_s") or 0.0)
    if isinstance(candidate, Mapping):
        candidate_cost = float(candidate.get("cost_s") or 0.0)
    return compare_costs(
        summary_fingerprint(reference),
        summary_fingerprint(candidate),
        reference_cost_s=reference_cost,
        candidate_cost_s=candidate_cost,
    )


def mismatch_fields(exc: IncomparableMeasurement) -> list[str]:
    """The field names that made two measurements incomparable, for reporting."""
    return sorted(key for key in exc.mismatches if not key.startswith("_"))
