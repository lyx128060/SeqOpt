"""Tests for the auditable C-patch-ledger CSV export.

The properties pinned here are the ones the ledger exists for:

* every required column is present, in the documented order;
* a screening rejection that the controller persists *only* into the indexes still
  becomes a row (with an empty ``eval_id``, because no evaluation record for it
  exists);
* ``actual_cpus`` is the dispatched fact from ``meta`` and NOT the ``eval_config``
  echo, and the ``actual_cpus_source`` column says which one won;
* the export is byte-stable across invocations (it is a derived view, not a log);
* ``query_set_sha256`` follows the query *set*, so reordering is irrelevant and
  changing a file's hash is not;
* one malformed store file does not abort the export;
* ``python -m seq_align.c_patch_loop.ledger`` works and writes nothing but the CSV.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from seq_align.c_patch_loop import ledger
from seq_align.c_patch_loop.ledger import (
    LEDGER_COLUMNS,
    LEDGER_HEADER,
    build_ledger_rows,
    export_ledger_csv,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = REPO_ROOT / "workspace"

BASELINE_EVAL_ID = "eval-baseline"
CANDIDATE_EVAL_ID = "eval-candidate"
STATIC_EVAL_ID = "eval-static"
STARTED_ATTEMPT_ID = "att-started"
COMPLETE_ATTEMPT_ID = "att-complete"
CANDIDATE_PATCHSET_ID = "ps-candidate"
CANDIDATE_FINGERPRINT = "fp-candidate"
SCREENING_DECISION = "rejected_screening_not_faster"

#: The diff text of the candidate's only patch; the ledger must digest this.
CANDIDATE_DIFF = (
    "--- a/src/impl_sse/ssvfilter.c\n"
    "+++ b/src/impl_sse/ssvfilter.c\n"
    "@@ -1 +1 @@\n"
    "-  if (x) return 0;\n"
    "+  if (!x) return 0;\n"
)

QUERY_FILES = [
    {"path": "/data/q1.fasta", "sha256": "a" * 64, "size": 11},
    {"path": "/data/q2.fasta", "sha256": "b" * 64, "size": 22},
]
REORDERED_QUERY_FILES = [QUERY_FILES[1], QUERY_FILES[0]]
MUTATED_QUERY_FILES = [dict(QUERY_FILES[0]), dict(QUERY_FILES[1], sha256="c" * 64)]

DATASET_MANIFEST = {
    "data_dir": "/data",
    "query_count": 2,
    "uniref_db": {"path": "/db/uniref.fa", "sha256": "u" * 64, "exists": True},
    "astral_db": {"path": "/db/astral.fa", "sha256": "a" * 64, "exists": True},
    "scop_txt": {"path": "/db/scop.txt", "sha256": "c" * 64, "exists": True},
}

EVAL_CONFIG = {
    "backend": "hmmer_cli_slurm",
    "cpus": 8,  # the CONFIG echo: deliberately different from the dispatched fact
    "stage": "full",
    "max_iter": 1,
    "eval_repeats": 2,
    "paired_current_best_repeats": 5,
    "uniref_db": "/db/uniref.fa",
    "astral_db": "/db/astral.fa",
    "scop_txt": "/db/scop.txt",
    "hmmer_src_root": "/src/hmmer",
}

HMMER_BUNDLE = {
    "bundle_id": "b" * 64,
    "patch_set_sha256": "p" * 64,
    "build_context": "host-cluster",
    "source_tree_sha256": "s" * 64,
}


def _write_json(path: Path, payload: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _manifest_with(query_files: list[dict]) -> dict:
    manifest = dict(DATASET_MANIFEST)
    manifest["query_files"] = [dict(item) for item in query_files]
    return manifest


def make_store(run_dir: Path, *, query_files: list[dict] | None = None) -> Path:
    """A small hand-written store exercising every record kind the ledger emits."""
    files = [dict(item) for item in (query_files if query_files is not None else QUERY_FILES)]
    root = run_dir / "c_patch_loop"

    # --- evaluations: baseline, candidate (paired block), static -------------
    _write_json(
        root / "evaluations" / f"{BASELINE_EVAL_ID}.json",
        {
            "schema_version": "1.0",
            "eval_id": BASELINE_EVAL_ID,
            "patchset_id": "ps-baseline",
            "round_idx": 0,
            "stage": "full",
            "status": "ok",
            "decision": "accepted_baseline",
            "created_at": "2026-01-01T00:00:00Z",
            "eval_result": {
                "stage": "full",
                "status": "ok",
                "tp": 10,
                "fp": 1,
                "tp_fp_ratio": 10 / 11,
                "cost_s": 100.0,
                "n_seqs": 2,
                "iterations": 1,
                "errors": [],
                "meta": {
                    # THE FACT: the cores the dispatched query actually got.
                    "query_cpus_per_query": 3,
                    # The job allocation, which is a different number.
                    "slurm_resources_per_query": {"cpus": 8, "partition": "compute"},
                    "max_parallel_queries": 2,
                    "queries_per_job": 2,
                    "hmmer_bundle": dict(HMMER_BUNDLE),
                    "backend": "hmmer_cli",
                    "query_results": [
                        {"query_index": 0, "query_id": "q1", "status": "ok", "cost_s": 1.0},
                        {"query_index": 1, "query_id": "q2", "status": "ok", "cost_s": 1.1},
                    ],
                },
            },
            "eval_config": dict(EVAL_CONFIG),
            "dataset_manifest": _manifest_with(files),
            "metric_delta": {},
            "errors": [],
            "attempt_ids": [],
            "primary_attempt_id": None,
            "evaluation_fingerprint": None,
        },
    )
    _write_json(
        root / "evaluations" / f"{CANDIDATE_EVAL_ID}.json",
        {
            "schema_version": "1.0",
            "eval_id": CANDIDATE_EVAL_ID,
            "patchset_id": CANDIDATE_PATCHSET_ID,
            "round_idx": 1,
            "stage": "full",
            "status": "ok",
            "decision": "accepted_speedup_paired",
            "created_at": "2026-01-01T00:10:00Z",
            "eval_result": {
                "stage": "full",
                "status": "ok",
                "tp": 10,
                "fp": 1,
                "tp_fp_ratio": 10 / 11,
                "cost_s": 90.0,
                "n_seqs": 2,
                "iterations": 1,
                "errors": [],
                "meta": {
                    "query_cpus_per_query": 5,
                    "max_parallel_queries": 1,
                    "hmmer_bundle": dict(HMMER_BUNDLE),
                    "paired_current_best": {
                        "schema_version": "seq_align_paired_current_best.v2",
                        "status": "ok",
                        "median_pair_speedup_fraction": 0.1,
                        "speedup_fraction": 0.1,
                        "pair_speedup_fractions": [0.08, 0.12],
                        "primary_candidate_cost_s": 91.5,
                        "current_best_patchset_id": "ps-baseline",
                    },
                    "query_results": [
                        {"query_index": 0, "query_id": "q1", "status": "ok", "cost_s": 0.9},
                        {"query_index": 1, "query_id": "q2", "status": "ok", "cost_s": 1.0},
                    ],
                },
            },
            "eval_config": dict(EVAL_CONFIG),
            "dataset_manifest": _manifest_with(files),
            "metric_delta": {
                "paired_confirmation": {
                    "median_pair_speedup_fraction": 0.1,
                    "pair_speedup_fractions": [0.08, 0.12],
                    "primary_candidate_cost_s": 91.5,
                    "current_best_patchset_id": "ps-baseline",
                }
            },
            "errors": [],
            "attempt_ids": [COMPLETE_ATTEMPT_ID],
            "primary_attempt_id": COMPLETE_ATTEMPT_ID,
            "query_completion_ids": [],
            "evaluation_fingerprint": CANDIDATE_FINGERPRINT,
        },
    )
    _write_json(
        root / "evaluations" / f"{STATIC_EVAL_ID}.json",
        {
            "schema_version": "1.0",
            "eval_id": STATIC_EVAL_ID,
            "patchset_id": "ps-static",
            "round_idx": 2,
            "stage": "static_verify",
            "status": "ok",
            "decision": "accepted_static",
            "created_at": "2026-01-01T00:20:00Z",
            "eval_result": {"stage": "static_verify", "status": "ok", "errors": [], "meta": {}},
            "eval_config": dict(EVAL_CONFIG, stage="static_verify"),
            "dataset_manifest": {},
            "metric_delta": {},
            "errors": [],
            "attempt_ids": [],
            "primary_attempt_id": None,
            "evaluation_fingerprint": None,
        },
    )

    # --- attempts: the store writes a started row and a terminal row ---------
    _write_json(
        root / "evaluation_attempts" / f"{STARTED_ATTEMPT_ID}.json",
        {
            "schema_version": "1.0",
            "attempt_id": STARTED_ATTEMPT_ID,
            "invocation_id": "inv-1",
            "patchset_id": CANDIDATE_PATCHSET_ID,
            "round_idx": 1,
            "stage": "full",
            "attempt_kind": "primary",
            # Same recorded identity as the evaluation, which is how a ``started``
            # row (never named by ``attempt_ids``) is linked to its evaluation.
            "evaluation_fingerprint": CANDIDATE_FINGERPRINT,
            "status": "started",
            "parent_attempt_id": None,
            "requested_queries": 2,
            "completed_queries": 0,
            "resumed_queries": 0,
            "query_completion_ids": [],
            "binary_sha256": {},
            "timing_protocol_fingerprint": "tp-1",
            "slurm_job_ids": [],
            "aggregate_result": None,
            "interruption": None,
            "errors": [],
            "created_at": "2026-01-01T00:05:00Z",
        },
    )
    _write_json(
        root / "evaluation_attempts" / f"{COMPLETE_ATTEMPT_ID}.json",
        {
            "schema_version": "1.0",
            "attempt_id": COMPLETE_ATTEMPT_ID,
            "invocation_id": "inv-1",
            "patchset_id": CANDIDATE_PATCHSET_ID,
            "round_idx": 1,
            "stage": "full",
            "attempt_kind": "primary",
            "evaluation_fingerprint": CANDIDATE_FINGERPRINT,
            "status": "complete",
            "parent_attempt_id": STARTED_ATTEMPT_ID,
            "requested_queries": 2,
            "completed_queries": 2,
            "resumed_queries": 0,
            "query_completion_ids": ["qc-1", "qc-2"],
            "binary_sha256": {},
            "timing_protocol_fingerprint": "tp-1",
            "slurm_job_ids": ["1", "2"],
            "aggregate_result": {
                "stage": "full",
                "status": "ok",
                "tp": 10,
                "fp": 1,
                "tp_fp_ratio": 10 / 11,
                "cost_s": 90.0,
                "n_seqs": 2,
                "iterations": 1,
                "errors": [],
                "meta": {
                    "query_cpus_per_query": 5,
                    "max_parallel_queries": 1,
                    "queries_per_job": 1,
                    "hmmer_bundle": dict(HMMER_BUNDLE),
                },
            },
            "interruption": None,
            "errors": [],
            "created_at": "2026-01-01T00:09:00Z",
        },
    )

    # --- per-query completions (distinct query_sha256 values) ----------------
    for index, (query_id, sha) in enumerate(
        (("q1", "1" * 64), ("q2", "2" * 64)), start=1
    ):
        _write_json(
            root / "query_completions" / f"qc-{index}.json",
            {
                "schema_version": "1.0",
                "query_completion_id": f"qc-{index}",
                "attempt_id": STARTED_ATTEMPT_ID,
                "invocation_id": "inv-1",
                "evaluation_fingerprint": CANDIDATE_FINGERPRINT,
                "query_fingerprint": f"qf-{index}",
                "patchset_id": CANDIDATE_PATCHSET_ID,
                "stage": "full",
                "attempt_kind": "primary",
                "query_idx": index,
                "query_id": query_id,
                "query_path": f"/data/{query_id}.fasta",
                "query_sha256": sha,
                "binary_sha256": {},
                "input_sha256": {
                    "uniref_db": "u" * 64,
                    "astral_db": "a" * 64,
                    "scop_txt": "c" * 64,
                },
                "status": "ok",
                "tp": 5,
                "fp": 0,
                "cost_s": 45.0,
                "iterations": 1,
                "artifacts": {},
                "errors": [],
                "created_at": "2026-01-01T00:08:00Z",
            },
        )

    # --- patchsets ----------------------------------------------------------
    _write_json(
        root / "patchsets" / "ps-baseline.json",
        {
            "schema_version": "1.0",
            "patchset_id": "ps-baseline",
            "round_idx": 0,
            "parent_patchset_id": None,
            "base_hmmer_source": "/src/hmmer",
            "source_hashes": {},
            "patches": [],
            "created_at": "2026-01-01T00:00:00Z",
        },
    )
    _write_json(
        root / "patchsets" / f"{CANDIDATE_PATCHSET_ID}.json",
        {
            "schema_version": "1.0",
            "patchset_id": CANDIDATE_PATCHSET_ID,
            "round_idx": 1,
            "parent_patchset_id": "ps-baseline",
            "base_hmmer_source": "/src/hmmer",
            "source_hashes": {},
            "patches": [
                {
                    "patch_id": "p1",
                    "component_name": "ssvfilter",
                    "c_target_file": "src/impl_sse/ssvfilter.c",
                    "c_target_functions": ["ssv_filter"],
                    "unified_diff": CANDIDATE_DIFF,
                    "edit_synthesis_errors": [],
                    "diff_sha256": None,
                    "rationale": "",
                }
            ],
            "created_at": "2026-01-01T00:04:00Z",
        },
    )

    # --- indexes: two screening rows + the slot joins + commit evidence ------
    # A: a screening rejection with NO evaluation record anywhere in the store.
    # B: a screening rejection that names an evaluation which does exist (the
    #    controller's real shape: the smoke measurement is persisted, the
    #    rejection decision is not).
    _write_json(
        root / "indexes" / "run_summary.json",
        {
            "schema_version": "1.0",
            "terminal_state": "interrupted",
            "results": [
                {
                    "outer_idx": 2,
                    "inner_idx": 1,
                    "patchset_id": "ps-screen-a",
                    "final_eval_id": "eval-absent",
                    "decision": SCREENING_DECISION,
                    "screening": True,
                    "status": "ok",
                    "tp": 5,
                    "fp": 0,
                    "cost_s": 1.25,
                    "screen_speedup_fraction": -0.01,
                    "screen_required_speedup_fraction": 0.015,
                },
                {
                    "outer_idx": 3,
                    "inner_idx": 2,
                    "patchset_id": "ps-screen-b",
                    "final_eval_id": BASELINE_EVAL_ID,
                    "decision": SCREENING_DECISION,
                    "screening": True,
                    "status": "ok",
                    "tp": 5,
                    "fp": 0,
                    "cost_s": 1.5,
                    "screen_speedup_fraction": -0.02,
                    "screen_required_speedup_fraction": 0.015,
                },
            ],
        },
    )
    # List-shaped index file; supplies the candidate's inner_trial / parent join.
    _write_json(
        root / "indexes" / "hierarchical_run_summary.json",
        [
            {
                "outer_idx": 1,
                "inner_idx": 1,
                "patchset_id": CANDIDATE_PATCHSET_ID,
                "parent_patchset_id": "ps-baseline",
                "final_eval_id": CANDIDATE_EVAL_ID,
                "status": "committed",
                "decision": "accepted_speedup_paired",
            }
        ],
    )
    _write_json(
        root / "indexes" / "accepted_lineage.json",
        [{"eval_id": BASELINE_EVAL_ID, "round_idx": 0, "stage": "full"}],
    )
    _write_json(
        root / "indexes" / "current_best.json",
        {"eval_id": CANDIDATE_EVAL_ID, "round_idx": 1, "stage": "full"},
    )
    return run_dir


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _header(path: Path) -> list[str]:
    with path.open(encoding="utf-8", newline="") as handle:
        return next(csv.reader(handle))


def _row(rows: list[dict[str, str]], **match: str) -> dict[str, str]:
    hits = [row for row in rows if all(row.get(key) == value for key, value in match.items())]
    assert len(hits) == 1, f"expected exactly one row matching {match}, got {len(hits)}"
    return hits[0]


def _numeric(value: str) -> tuple[int, int]:
    text = (value or "").strip()
    if not text:
        return 1, 0
    try:
        return 0, int(text)
    except ValueError:
        return 1, 0


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------


def test_header_has_every_required_column_in_documented_order(tmp_path: Path) -> None:
    run_dir = make_store(tmp_path / "run-header")
    out = export_ledger_csv(run_dir, tmp_path / "header.csv")

    header = _header(out)
    assert header[: len(LEDGER_COLUMNS)] == list(LEDGER_COLUMNS)
    for name in LEDGER_COLUMNS:
        assert name in header, f"required column {name!r} missing from the header"
    assert header == list(LEDGER_HEADER)
    # The provenance columns the ledger contract demands must exist too.
    assert "actual_cpus_source" in header
    assert "uniref_db_sha256_source" in header
    assert "astral_sha256_source" in header
    assert "scop_sha256_source" in header


# ---------------------------------------------------------------------------
# Screening-only rejections
# ---------------------------------------------------------------------------


def test_screening_rejection_persisted_only_in_indexes_becomes_a_row(tmp_path: Path) -> None:
    run_dir = make_store(tmp_path / "run-screening")
    rows = build_ledger_rows(run_dir)

    assert not any(row["eval_id"] == "eval-absent" for row in rows)
    index_only = _row(
        rows, record_kind="screening_rejection", patchset_id="ps-screen-a"
    )
    assert index_only["eval_id"] == ""
    assert index_only["decision"] == SCREENING_DECISION
    assert index_only["note"], "a screening-only row must say why its eval_id is empty"
    assert index_only["outer_round"] == "2"
    assert index_only["inner_trial"] == "1"
    assert index_only["tp"] == "5"
    assert index_only["cost_s"] == "1.25"
    assert index_only["screen_speedup_fraction"] == "-0.01"
    assert index_only["screen_required_speedup_fraction"] == "0.015"
    assert index_only["committed"] == "false"

    # The second screening row *names* a persisted evaluation: the link is echoed
    # but the required eval_id column still stays empty.
    linked = _row(rows, record_kind="screening_rejection", patchset_id="ps-screen-b")
    assert linked["eval_id"] == ""
    assert linked["referenced_eval_id"] == BASELINE_EVAL_ID
    assert linked["stage"] == "full"
    assert linked["actual_cpus_source"].startswith("linked_eval:")

    # And the screening *decision* really is absent from evaluations/: the smoke
    # record that does exist carries a different decision.
    assert not any(
        row["record_kind"] == "evaluation" and row["decision"] == SCREENING_DECISION
        for row in rows
    )


# ---------------------------------------------------------------------------
# actual_cpus: the fact beats the config echo
# ---------------------------------------------------------------------------


def test_actual_cpus_is_the_dispatched_fact_not_the_eval_config_echo(tmp_path: Path) -> None:
    run_dir = make_store(tmp_path / "run-cpus")
    rows = build_ledger_rows(run_dir)

    baseline = _row(rows, record_kind="evaluation", eval_id=BASELINE_EVAL_ID)
    # The store says meta.query_cpus_per_query=3 while eval_config.cpus=8.
    assert baseline["actual_cpus"] == "3"
    assert baseline["actual_cpus"] != "8"
    assert baseline["actual_cpus_source"] == "meta.query_cpus_per_query"

    complete = _row(rows, record_kind="attempt", attempt_id=COMPLETE_ATTEMPT_ID)
    assert complete["actual_cpus"] == "5"
    assert complete["actual_cpus_source"] == "meta.query_cpus_per_query"

    # A record that only has the evaluation's config to lean on must SAY so: the
    # started attempt records no cores of its own.
    started = _row(rows, record_kind="attempt", attempt_id=STARTED_ATTEMPT_ID)
    assert started["status"] == "started"
    assert started["parent_attempt_id"] == ""
    assert started["actual_cpus"] == "8"
    assert started["actual_cpus_source"] == "eval_config.cpus"


# ---------------------------------------------------------------------------
# Stability
# ---------------------------------------------------------------------------


def test_export_is_byte_identical_when_called_twice(tmp_path: Path) -> None:
    run_dir = make_store(tmp_path / "run-stable")
    first = export_ledger_csv(run_dir, tmp_path / "a.csv")
    second = export_ledger_csv(run_dir, tmp_path / "b.csv")
    assert first.read_bytes() == second.read_bytes()
    assert first.read_bytes().endswith(b"\n")


def test_rows_have_a_stable_total_order_with_empty_numerics_last(tmp_path: Path) -> None:
    run_dir = make_store(tmp_path / "run-order")
    rows = build_ledger_rows(run_dir)
    probes = [(_numeric(row["outer_round"]), _numeric(row["inner_trial"])) for row in rows]
    assert probes == sorted(probes)

    # Round 2 holds both a screening row (inner_trial 1) and the static evaluation
    # (no inner_trial at all): the empty numeric must come last.
    round_two = [row for row in rows if row["outer_round"] == "2"]
    assert [row["inner_trial"] for row in round_two] == ["1", ""]


# ---------------------------------------------------------------------------
# query_set_sha256
# ---------------------------------------------------------------------------


def _baseline_query_set(run_dir: Path) -> str:
    rows = build_ledger_rows(run_dir)
    row = _row(rows, record_kind="evaluation", eval_id=BASELINE_EVAL_ID)
    assert row["query_set_sha256_source"] == "dataset_manifest.query_files"
    return row["query_set_sha256"]


def test_query_set_sha256_tracks_the_set_not_the_order(tmp_path: Path) -> None:
    base = _baseline_query_set(make_store(tmp_path / "run-qs-base"))
    reordered = _baseline_query_set(
        make_store(tmp_path / "run-qs-reordered", query_files=REORDERED_QUERY_FILES)
    )
    mutated = _baseline_query_set(
        make_store(tmp_path / "run-qs-mutated", query_files=MUTATED_QUERY_FILES)
    )
    assert base, "the fixture must produce a non-empty query-set digest"
    assert base == reordered
    assert base != mutated


def test_query_set_sha256_uses_query_completions_for_attempts(tmp_path: Path) -> None:
    run_dir = make_store(tmp_path / "run-qs-attempt")
    rows = build_ledger_rows(run_dir)
    complete = _row(rows, record_kind="attempt", attempt_id=COMPLETE_ATTEMPT_ID)
    started = _row(rows, record_kind="attempt", attempt_id=STARTED_ATTEMPT_ID)
    assert complete["query_set_sha256_source"] == "query_completions"
    assert complete["query_set_sha256"] == started["query_set_sha256"]
    # The per-query completions (q1/q2 hashes 1.., 2..) are a different set from the
    # dataset manifest (q1.fasta/q2.fasta hashes a.., b..).
    baseline = _row(rows, record_kind="evaluation", eval_id=BASELINE_EVAL_ID)
    assert complete["query_set_sha256"] != baseline["query_set_sha256"]


# ---------------------------------------------------------------------------
# Other column contracts
# ---------------------------------------------------------------------------


def test_patch_and_commit_columns_come_from_the_store(tmp_path: Path) -> None:
    run_dir = make_store(tmp_path / "run-columns")
    rows = build_ledger_rows(run_dir)

    candidate = _row(rows, record_kind="evaluation", eval_id=CANDIDATE_EVAL_ID)
    assert candidate["changed_components"] == "src/impl_sse/ssvfilter.c"
    assert candidate["patch_signature"]
    assert candidate["patch_diff_sha256"] == hashlib.sha256(
        CANDIDATE_DIFF.encode("utf-8")
    ).hexdigest()
    assert candidate["median_pair_speedup_fraction"] == "0.1"
    assert candidate["pair_speedup_fractions"] == "[0.08,0.12]"
    assert candidate["primary_candidate_cost_s"] == "91.5"
    assert candidate["compared_against_parent"] == "ps-baseline"
    assert candidate["patch_set_sha256"] == "p" * 64
    assert candidate["bundle_id"] == "b" * 64
    assert candidate["build_context"] == "host-cluster"
    assert candidate["hmmer_source_sha256"] == "s" * 64
    assert candidate["uniref_db_sha256"] == "u" * 64
    assert candidate["uniref_db_sha256_source"] == "dataset_manifest.uniref_db.sha256"
    assert candidate["astral_sha256"] == "a" * 64
    assert candidate["scop_sha256"] == "c" * 64
    assert candidate["inner_trial"] == "1"
    assert candidate["attempt_kind"] == "primary"
    assert candidate["decision"] == "accepted_speedup_paired"
    assert candidate["committed"] == "true"
    assert candidate["committed_source"] == "current_best"

    baseline = _row(rows, record_kind="evaluation", eval_id=BASELINE_EVAL_ID)
    assert baseline["committed"] == "true"
    assert baseline["committed_source"] == "accepted_lineage"

    static = _row(rows, record_kind="evaluation", eval_id=STATIC_EVAL_ID)
    assert static["committed"] == "false"
    assert static["committed_source"] == ""

    # attempts never invent a decision or a commit verdict
    complete = _row(rows, record_kind="attempt", attempt_id=COMPLETE_ATTEMPT_ID)
    assert complete["decision"] == ""
    assert complete["committed"] == ""
    assert complete["eval_id"] == CANDIDATE_EVAL_ID
    assert complete["compared_against_parent"] == "ps-baseline"
    assert complete["n_queries"] == "2"
    assert complete["job_concurrency"] == "1"


# ---------------------------------------------------------------------------
# Malformed store files
# ---------------------------------------------------------------------------


def test_malformed_json_is_skipped_without_losing_good_rows(tmp_path: Path) -> None:
    run_dir = make_store(tmp_path / "run-malformed")
    (run_dir / "c_patch_loop" / "evaluations" / "broken.json").write_text(
        "{ this is not json", encoding="utf-8"
    )
    (run_dir / "c_patch_loop" / "evaluation_attempts" / "also-broken.json").write_text(
        "[1, 2, 3]", encoding="utf-8"  # valid JSON, but not a record object
    )

    stats: dict = {}
    rows = build_ledger_rows(run_dir, stats=stats)  # must not raise

    assert stats["skipped_malformed_json"] == 2
    assert stats["warnings"] >= 2
    assert _row(rows, record_kind="evaluation", eval_id=BASELINE_EVAL_ID)
    assert _row(rows, record_kind="evaluation", eval_id=CANDIDATE_EVAL_ID)
    assert _row(rows, record_kind="attempt", attempt_id=COMPLETE_ATTEMPT_ID)
    assert len(rows) == 7
    # and the export still writes a complete CSV
    out = export_ledger_csv(run_dir, tmp_path / "malformed.csv")
    assert len(_read_rows(out)) == 7


def test_run_dir_without_a_store_exports_nothing(tmp_path: Path) -> None:
    stats: dict = {}
    assert build_ledger_rows(tmp_path / "empty-run", stats=stats) == []
    assert stats["rows"] == 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def test_cli_exports_only_the_named_csv(tmp_path: Path) -> None:
    run_dir = make_store(tmp_path / "run-cli")
    before = sorted(str(path.relative_to(run_dir)) for path in run_dir.rglob("*"))
    out = tmp_path / "cli-ledger.csv"

    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(WORKSPACE)] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else [])
    )
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "seq_align.c_patch_loop.ledger",
            "--run-dir",
            str(run_dir),
            "--out",
            str(out),
        ],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        env=env,
    )
    assert completed.returncode == 0, completed.stderr
    assert out.is_file()
    assert "wrote 7 rows" in completed.stderr

    after = sorted(str(path.relative_to(run_dir)) for path in run_dir.rglob("*"))
    assert before == after, "the exporter must not write anything into the run directory"

    assert _header(out) == list(LEDGER_HEADER)
    assert len(_read_rows(out)) == 7


def test_cli_main_is_importable_and_reports_row_count(tmp_path: Path, capsys) -> None:
    run_dir = make_store(tmp_path / "run-main")
    out = tmp_path / "main.csv"
    assert ledger.main(["--run-dir", str(run_dir), "--out", str(out)]) == 0
    assert len(_read_rows(out)) == 7
    assert "wrote 7 rows" in capsys.readouterr().err


@pytest.mark.parametrize(
    "eval_id, expected_rows",
    [(BASELINE_EVAL_ID, 1), (CANDIDATE_EVAL_ID, 1), (STATIC_EVAL_ID, 1)],
)
def test_every_evaluation_has_exactly_one_evaluation_row(
    tmp_path: Path, eval_id: str, expected_rows: int
) -> None:
    """Attempt rows carry their linked eval_id too, so count ``evaluation`` rows only."""
    run_dir = make_store(tmp_path / f"run-eval-{eval_id}")
    rows = build_ledger_rows(run_dir)
    evaluation_rows = [
        row for row in rows if row["record_kind"] == "evaluation" and row["eval_id"] == eval_id
    ]
    assert len(evaluation_rows) == expected_rows


def test_every_attempt_record_gets_exactly_one_row(tmp_path: Path) -> None:
    """started/complete pairs are NOT collapsed: two records, two rows."""
    run_dir = make_store(tmp_path / "run-attempts")
    rows = build_ledger_rows(run_dir)
    attempts = [row for row in rows if row["record_kind"] == "attempt"]
    assert sorted(row["attempt_id"] for row in attempts) == sorted(
        [STARTED_ATTEMPT_ID, COMPLETE_ATTEMPT_ID]
    )
    statuses = {row["attempt_id"]: row["status"] for row in attempts}
    assert statuses[STARTED_ATTEMPT_ID] == "started"
    assert statuses[COMPLETE_ATTEMPT_ID] == "complete"
    # the terminal row points back at its started parent, so the pair is auditable
    complete = _row(rows, record_kind="attempt", attempt_id=COMPLETE_ATTEMPT_ID)
    assert complete["parent_attempt_id"] == STARTED_ATTEMPT_ID
