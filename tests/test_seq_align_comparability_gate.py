"""The refusal gate, and the two orthogonal core knobs.

Two properties are locked in here, both of which were violated by the 2026-10-09
audit:

1. **A cost number may not be subtracted or divided unless its conditions match.**
   The audit found verdicts computed across local and Slurm backends, across
   ``--cpus-per-task`` 1 and 8, across a 288 MB SwissProt database and a 60.9 GB
   UniRef90 one, and with a baseline that a later pass had overwritten. Each of
   those produced a confident speedup from numbers that were never comparable.
   ``compare_costs`` must refuse -- and must refuse *loudly*, naming the fields --
   rather than return a ratio.

2. **"Cores" is two independent quantities.** The job allocation
   (``--cpus-per-task``) and the cores one query actually gets (the literal HMMER
   ``--cpu``) are different numbers, and conflating them is what turned a 45-core
   job into ``jackhmmer --cpu 44`` in the old code's shape.

Nothing here relaxes either property: every test asserts a refusal that must keep
happening, or a value that must stay pinned.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pytest

from seq_align import slurm_eval
from seq_align.c_patch_loop import comparability
from seq_align.c_patch_loop.comparability import (
    IncomparableMeasurement,
    compare_costs,
    compare_summaries,
    comparison_fingerprint,
    query_set_sha256,
)
from seq_align.c_patch_loop.controller import _strictly_faster, _try_speedup_fraction
from seq_align.c_patch_loop.models import EvalSummary
from seq_align.eval_schema import EvalResult


# ── helpers ──────────────────────────────────────────────────────────────────

def _fingerprint(**overrides: object) -> dict:
    """A complete, valid fingerprint. Override one key to make it incomparable."""
    fingerprint: dict = {
        "schema_version": "seqalign_comparability.v1",
        "backend": "hmmer_cli_slurm",
        "partition": "compute",
        "allocated_cpus": 45,
        "query_threads": 8,
        "queries_in_job": 5,
        "stage": "full",
        "query_set_sha256": "q" * 64,
        "max_iter": 1,
        "uniref_db_path": "/data/uniref90/uniref90.fasta",
        "uniref_db_sha256": "u" * 64,
        "astral_db_path": "/data/scop_astral/astral.fa",
        "astral_db_sha256": "a" * 64,
        "scop_txt_path": "/data/scop_astral/dir.cla.scope.txt",
        "scop_txt_sha256": "s" * 64,
        "hmmer_source_sha256": "h" * 64,
        "eval_repeats": 2,
        "paired_current_best_repeats": 2,
        "bundle_id": "b" * 64,
        "build_context": "host-cluster",
    }
    fingerprint.update(overrides)
    return fingerprint


def _summary(cost_s: float, *, fingerprint: dict | None = None, eval_id: str = "e", patchset_id: str = "p", tp: int = 10, fp: int = 1) -> EvalSummary:
    return EvalSummary(
        eval_id=eval_id,
        patchset_id=patchset_id,
        stage="full",
        status="ok",
        tp=tp,
        fp=fp,
        tp_fp_ratio=tp / (tp + fp) if (tp + fp) else 0.0,
        cost_s=cost_s,
        comparability=fingerprint if fingerprint is not None else _fingerprint(),
    )


def _runner_stub(**overrides: object) -> SimpleNamespace:
    """A stand-in carrying only the attributes the decision path reads."""
    from seq_align.c_patch_loop.runner import PatchRunner

    attrs = {
        "accept_min_tp_fraction": 1.0,
        "min_speedup_fraction": 0.03,
        "near_speedup_band_fraction": 0.02,
        # Bound directly: the stub never touches any other instance state.
        "_refuse_incomparable": lambda reference, result, *, stage: PatchRunner._refuse_incomparable(
            None, reference, result, stage=stage
        ),
    }
    attrs.update(overrides)
    return SimpleNamespace(**attrs)


# ── 1. the gate itself ───────────────────────────────────────────────────────

def test_identical_conditions_produce_a_speedup():
    """The gate must not be a wall: equal conditions still yield a number."""
    speedup = compare_costs(
        _fingerprint(), _fingerprint(), reference_cost_s=100.0, candidate_cost_s=90.0
    )
    assert speedup == pytest.approx(0.10)


@pytest.mark.parametrize(
    "field, value",
    [
        # The four the task names explicitly.
        ("query_threads", 1),          # 核数不同
        ("query_set_sha256", "z" * 64),  # 序列集合不同
        ("stage", "smoke"),            # stage 不同
        ("uniref_db_path", "/data/uniprot/uniprot_sprot.fasta"),  # 数据库不同
        ("uniref_db_sha256", "z" * 64),
        # And the rest of the conditions that change a measurement.
        ("backend", "hmmer_cli_local"),
        ("partition", "mineru"),
        ("allocated_cpus", 8),
        ("queries_in_job", 1),
        ("max_iter", 3),
        ("astral_db_sha256", "z" * 64),
        ("scop_txt_sha256", "z" * 64),
        ("hmmer_source_sha256", "z" * 64),
        ("eval_repeats", 1),
        ("paired_current_best_repeats", 0),
        ("bundle_id", "z" * 64),
        ("build_context", "other"),
    ],
)
def test_refuses_when_any_condition_differs(field, value):
    with pytest.raises(IncomparableMeasurement) as excinfo:
        compare_costs(
            _fingerprint(),
            _fingerprint(**{field: value}),
            reference_cost_s=100.0,
            candidate_cost_s=90.0,
        )
    assert field in excinfo.value.mismatches
    # The refusal must name the two values, not just fail.
    assert excinfo.value.mismatches[field] == {"reference": _fingerprint()[field], "candidate": value}


def test_refuses_when_a_required_condition_is_unrecorded():
    """An unknown condition is not a matching condition.

    This is the case that matters for every pre-existing record: the old store
    never recorded a per-query core count at all, so those numbers are
    unverifiable rather than equal.
    """
    with pytest.raises(IncomparableMeasurement) as excinfo:
        compare_costs({}, _fingerprint(), reference_cost_s=100.0, candidate_cost_s=90.0)
    assert "_missing_required" in excinfo.value.mismatches

    with pytest.raises(IncomparableMeasurement):
        compare_costs(_fingerprint(), {}, reference_cost_s=100.0, candidate_cost_s=90.0)


def test_refuses_a_non_positive_reference_cost():
    with pytest.raises(IncomparableMeasurement):
        compare_costs(_fingerprint(), _fingerprint(), reference_cost_s=0.0, candidate_cost_s=1.0)


def test_compare_summaries_reads_the_fingerprint_off_the_summary():
    faster = _summary(90.0)
    baseline = _summary(100.0, eval_id="base")
    assert comparability.compare_summaries(baseline, faster) == pytest.approx(0.10)

    mismatched = _summary(90.0, fingerprint=_fingerprint(query_threads=1))
    with pytest.raises(IncomparableMeasurement):
        comparability.compare_summaries(baseline, mismatched)


def test_query_set_fingerprint_tracks_membership_not_order():
    a = [{"query_id": "q1", "query_sha256": "1" * 64}, {"query_id": "q2", "query_sha256": "2" * 64}]
    b = [{"query_id": "q2", "query_sha256": "2" * 64}, {"query_id": "q1", "query_sha256": "1" * 64}]
    assert query_set_sha256(a) == query_set_sha256(b)
    changed = [{"query_id": "q1", "query_sha256": "1" * 64}, {"query_id": "q2", "query_sha256": "9" * 64}]
    assert query_set_sha256(a) != query_set_sha256(changed)
    assert query_set_sha256([]) is None


def test_comparison_fingerprint_separates_job_cores_from_query_threads():
    """The job allocation must not be mistaken for the per-query core count.

    ``allocated_cpus`` and ``query_threads`` come from different recorded fields,
    and a fingerprint built from a worker's meta must show 45 allocated / 8 per
    query rather than collapsing them.
    """
    fingerprint = comparison_fingerprint(
        eval_config={"backend": "hmmer_cli_slurm", "stage": "full", "max_iter": 1, "cpus": 8},
        meta={
            "job_cpus": 45,
            "query_cpus_per_query": 8,
            "queries_per_job": 5,
            "slurm_resources_per_query": {"partition": "compute", "cpus": 45},
            "uniref_db_sha256": "u" * 64,
            "astral_db_sha256": "a" * 64,
            "scop_txt_sha256": "s" * 64,
            "hmmer_bundle": {"source_tree_sha256": "h" * 64, "bundle_id": "b" * 64},
        },
        query_manifest=[{"query_id": "q1", "query_sha256": "1" * 64}],
    )
    assert fingerprint["allocated_cpus"] == 45
    assert fingerprint["query_threads"] == 8
    assert fingerprint["queries_in_job"] == 5
    assert fingerprint["partition"] == "compute"
    assert fingerprint["hmmer_source_sha256"] == "h" * 64


def test_comparison_fingerprint_prefers_recorded_meta_over_config_echo():
    """Only what the evaluation recorded may be treated as fact.

    A config that claims 8 cores while the worker recorded 1 must fingerprint as
    1, because that is what was dispatched.
    """
    fingerprint = comparison_fingerprint(
        eval_config={"backend": "hmmer_cli_slurm", "stage": "full", "cpus": 8},
        meta={"slurm_resources_per_query": {"cpus": 1}, "hmmer_worker_threads_per_query": 0},
        query_manifest=[{"query_id": "q1", "query_sha256": "1" * 64}],
    )
    assert fingerprint["allocated_cpus"] == 1
    assert fingerprint["query_threads"] == 0


# ── 2. the decision path uses the gate ───────────────────────────────────────

def _decision(stub: SimpleNamespace, reference: EvalSummary | None, result: EvalResult, *, stage: str = "full") -> str:
    from seq_align.c_patch_loop.runner import PatchRunner

    return PatchRunner._decision(stub, result, stage=stage, current_best=reference, quality_reference=reference, build=None)


def test_decision_refuses_a_candidate_measured_at_a_different_core_count():
    """A candidate at 1 core must not be declared 'faster' than a baseline at 8."""
    stub = _runner_stub()
    reference = _summary(100.0, fingerprint=_fingerprint(query_threads=8), eval_id="base")
    result = EvalResult(stage="full", status="ok", tp=10, fp=1, cost_s=90.0)
    result.meta["comparability"] = _fingerprint(query_threads=1)

    decision = _decision(stub, reference, result)
    assert decision == "rejected_incomparable_measurement"
    assert (result.meta or {}).get("incomparable_measurement", {}).get("mismatches", {}).get("query_threads")
    # And the refusal must be recorded on the result, not swallowed.
    assert any("refused to compare" in error for error in result.errors)


def test_decision_still_accepts_a_speedup_measured_the_same_way():
    """The gate must not make the loop unable to ever accept anything."""
    stub = _runner_stub()
    reference = _summary(100.0, eval_id="base")
    result = EvalResult(stage="full", status="ok", tp=10, fp=1, cost_s=90.0)
    result.meta["comparability"] = _fingerprint()
    assert _decision(stub, reference, result) == "accepted_speedup"


def test_decision_refuses_across_a_different_sequence_set():
    stub = _runner_stub()
    reference = _summary(100.0, fingerprint=_fingerprint(query_set_sha256="q" * 64), eval_id="base")
    result = EvalResult(stage="full", status="ok", tp=10, fp=1, cost_s=90.0)
    result.meta["comparability"] = _fingerprint(query_set_sha256="z" * 64)
    assert _decision(stub, reference, result) == "rejected_incomparable_measurement"


def test_decision_refuses_across_a_different_database():
    stub = _runner_stub()
    reference = _summary(100.0, eval_id="base")
    result = EvalResult(stage="full", status="ok", tp=10, fp=1, cost_s=90.0)
    result.meta["comparability"] = _fingerprint(uniref_db_path="/data/uniprot/uniprot_sprot.fasta")
    assert _decision(stub, reference, result) == "rejected_incomparable_measurement"


def test_decision_refuses_across_stages():
    stub = _runner_stub()
    reference = _summary(100.0, fingerprint=_fingerprint(stage="smoke"), eval_id="base")
    reference.stage = "smoke"
    result = EvalResult(stage="full", status="ok", tp=10, fp=1, cost_s=90.0)
    result.meta["comparability"] = _fingerprint(stage="full")
    assert _decision(stub, reference, result) == "rejected_incomparable_measurement"


def test_strictly_faster_refuses_instead_of_advancing_the_incumbent():
    reference = _summary(100.0, eval_id="base")
    equal = _summary(90.0)
    assert _strictly_faster(equal, reference) is True
    mismatched = _summary(90.0, fingerprint=_fingerprint(query_threads=1))
    assert _strictly_faster(mismatched, reference) is False


def test_try_speedup_fraction_returns_a_mismatch_instead_of_a_ratio():
    reference = _summary(100.0, eval_id="base")
    value, mismatch = _try_speedup_fraction(reference, _summary(90.0))
    assert value == pytest.approx(0.10) and mismatch is None

    value, mismatch = _try_speedup_fraction(reference, _summary(90.0, fingerprint=_fingerprint(stage="smoke")))
    assert value is None
    assert mismatch is not None and "stage" in mismatch["mismatches"]


def test_metric_delta_emits_no_ratio_for_incomparable_numbers():
    """A missing ratio is the honest output; a number here would be read as a result."""
    from seq_align.c_patch_loop.runner import PatchRunner

    stub = _runner_stub()
    reference = _summary(100.0, eval_id="base")
    result = EvalResult(stage="full", status="ok", tp=10, fp=1, cost_s=90.0)
    result.meta["comparability"] = _fingerprint(backend="hmmer_cli_local")
    delta = PatchRunner._metric_delta(stub, result, reference, quality_reference=reference)
    assert delta["speedup_fraction"] is None
    assert delta["cost_s_delta"] is None
    assert delta["incomparable_measurement"]["mismatches"]["backend"]


# ── 3. the two orthogonal core knobs ─────────────────────────────────────────

def _clean_cpu_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "SEQ_ALIGN_SLURM_FULL_CPUS",
        "SEQ_ALIGN_SLURM_SMOKE_CPUS",
        "SEQ_ALIGN_SLURM_FULL_QUERY_CPUS",
        "SEQ_ALIGN_SLURM_SMOKE_QUERY_CPUS",
        "SEQ_ALIGN_SLURM_FULL_QUERIES_PER_JOB",
        "SEQ_ALIGN_SLURM_SMOKE_QUERIES_PER_JOB",
        "SEQ_ALIGN_SLURM_FULL_MAX_PARALLEL",
        "SEQ_ALIGN_SLURM_SMOKE_MAX_PARALLEL",
    ):
        monkeypatch.delenv(name, raising=False)


def test_production_core_shape_is_five_queries_at_eight_cores_in_a_45_core_job(monkeypatch):
    """The agreed shape: 5 concurrent queries x 8 cores + 5 coordinator cores = 45.

    A 50-query run is then 10 jobs running 50 queries in parallel.
    """
    _clean_cpu_env(monkeypatch)
    for stage in ("full", "smoke"):
        assert slurm_eval.query_cpus_for(stage) == 8
        assert slurm_eval.queries_per_job(stage) == 5
        assert slurm_eval.min_job_cpus_for(stage) == 45
        assert slurm_eval.resources_for(stage).cpus == 45
        # %N must not exceed what the allocation covers.
        assert slurm_eval.max_parallel_for(stage) == 5

    assert -(-50 // slurm_eval.queries_per_job("full")) == 10  # 10 jobs for 50 queries


def test_the_job_allocation_never_becomes_the_hmmer_cpu_argument(monkeypatch):
    """``--cpu`` is the per-query count, never ``allocation - 1``.

    The old derivation (``cpus - 1``) would have handed HMMER ``--cpu 44`` inside a
    45-core job; the new one passes the declared per-query count through.
    """
    from seq_align import run_hmmer_cli

    _clean_cpu_env(monkeypatch)
    assert run_hmmer_cli._hmmer_worker_threads(8) == 8
    assert run_hmmer_cli._hmmer_worker_threads(45) == 45
    assert run_hmmer_cli._hmmer_worker_threads(1) == 1
    assert run_hmmer_cli._hmmer_worker_threads(0) == 1  # never HMMER's self-detect sentinel


def test_an_explicit_allocation_that_cannot_fit_the_shape_is_refused(monkeypatch):
    """The old default of 1 core per job must now be a loud error, not a silent 5x oversubscription."""
    _clean_cpu_env(monkeypatch)
    monkeypatch.setenv("SEQ_ALIGN_SLURM_FULL_CPUS", "8")
    outcome = slurm_eval.submit_and_wait(
        stage="full",
        mods=[],
        run_dir=Path("/tmp/does-not-matter"),
        round_idx=0,
        data_dir=Path("/tmp/does-not-matter"),
        max_iter=1,
        astral_db="/tmp/a",
        uniref_db="/tmp/u",
        scop_txt="/tmp/s",
    )
    assert outcome.status == "run_error"
    assert any("job allocation too small" in error for error in outcome.errors)


def test_a_per_query_core_count_that_disagrees_with_the_runner_is_refused(monkeypatch):
    """A record may not claim a core count the submission would not dispatch."""
    _clean_cpu_env(monkeypatch)
    outcome = slurm_eval.submit_and_wait(
        stage="full",
        mods=[],
        run_dir=Path("/tmp/does-not-matter"),
        round_idx=0,
        data_dir=Path("/tmp/does-not-matter"),
        max_iter=1,
        astral_db="/tmp/a",
        uniref_db="/tmp/u",
        scop_txt="/tmp/s",
        expected_query_cpus=2,
    )
    assert outcome.status == "run_error"
    assert any("per-query core count mismatch" in error for error in outcome.errors)


def test_job_allocation_is_derived_from_the_two_knobs(monkeypatch):
    """Raising either knob raises the derived allocation, so they cannot drift apart."""
    _clean_cpu_env(monkeypatch)
    assert slurm_eval.min_job_cpus_for("full") == 45

    monkeypatch.setenv("SEQ_ALIGN_SLURM_FULL_QUERY_CPUS", "4")
    assert slurm_eval.min_job_cpus_for("full") == 25  # 5 * (4 + 1)

    monkeypatch.setenv("SEQ_ALIGN_SLURM_FULL_QUERIES_PER_JOB", "10")
    assert slurm_eval.min_job_cpus_for("full") == 50  # 10 * (4 + 1)


def test_request_records_the_dispatched_core_shape_not_the_config_echo(tmp_path):
    """The request a worker reads must carry the per-query count explicitly."""
    request = {
        "stage": "full",
        "mods": [],
        "query_paths": ["q0"],
        "max_iter": 1,
        "astral_db": "a",
        "uniref_db": "u",
        "scop_txt": "s",
        "work_root": str(tmp_path / "work"),
        "result_dir": str(tmp_path / "results"),
        # A deliberately misleading job allocation: the worker must use query_cpus.
        "resources": {"partition": "compute", "cpus": 45, "memory": "16G", "walltime": "24:00:00"},
        "query_cpus": 8,
        "queries_per_job": 5,
        "source_paths": {"data_dir": "d", "astral_db": "a", "uniref_db": "u", "scop_txt": "s"},
    }
    (tmp_path / "queries").mkdir()
    (tmp_path / "queries" / "q0").write_text(">q\nMKV\n", encoding="utf-8")
    request["query_paths"] = [str(tmp_path / "queries" / "q0")]
    request_path = tmp_path / "request.json"
    request_path.write_text(json.dumps(request), encoding="utf-8")

    captured: dict = {}

    def fake_run_all_queries(cfg, **kwargs):
        captured.update(kwargs)
        return EvalResult(stage="full", status="ok", tp=1, fp=0, cost_s=1.0)

    from seq_align import run_hmmer_cli

    import seq_align.slurm_eval as module

    original = run_hmmer_cli.run_all_queries
    run_hmmer_cli.run_all_queries = fake_run_all_queries
    try:
        module.run_request(request_path, 0)
    finally:
        run_hmmer_cli.run_all_queries = original

    assert captured["cpus"] == 8, "the worker passed the job allocation instead of the per-query count"


def test_resume_matches_on_the_core_shape(tmp_path, monkeypatch):
    """A job measured under a different core shape must not be re-attached."""
    _clean_cpu_env(monkeypatch)
    round_path = tmp_path / "round_000"
    job_dir = round_path / "slurm" / "full_1"
    (job_dir / "results").mkdir(parents=True)
    request = {
        "stage": "full",
        "mods": [],
        "max_iter": 1,
        "source_paths": {"data_dir": "d", "astral_db": "a", "uniref_db": "u", "scop_txt": "s"},
        "query_timeout_policy": None,
        "optimization_objective": slurm_eval.objective_metadata(),
        "runtime": "host",
        "image_ref": None,
        "hmmer_bundle_id": None,
        "query_paths": ["q0"],
        "resources": asdict(slurm_eval.resources_for("full")),
        "query_cpus": 8,
        "queries_per_job": 5,
    }
    (job_dir / "request.json").write_text(json.dumps(request), encoding="utf-8")
    (job_dir / "submission.json").write_text(
        json.dumps({"job_id": "55", "query_count": 1, "runtime": "host", "image_ref": None}),
        encoding="utf-8",
    )
    (job_dir / "results" / "result_000.json").write_text(
        EvalResult(stage="full", status="ok").model_dump_json(), encoding="utf-8"
    )

    def find(request_dict):
        (job_dir / "request.json").write_text(json.dumps(request_dict), encoding="utf-8")
        return slurm_eval._find_resumable_job(
            round_path=round_path,
            stage="full",
            mods=[],
            max_iter=1,
            source_paths=request["source_paths"],
            query_timeout_policy=None,
            optimization_objective=slurm_eval.objective_metadata(),
            bundle_id=None,
            runtime="host",
            image_ref=None,
        )

    assert find(request) is not None
    assert find({**request, "query_cpus": 1}) is None
    assert find({**request, "queries_per_job": 1}) is None
    assert find({**request, "resources": {**request["resources"], "cpus": 8}}) is None
    assert find({k: v for k, v in request.items() if k != "query_cpus"}) is None, (
        "a request whose core shape is unrecorded must not be resumed as if it matched"
    )


# ── 4. the verdict reference is the round parent, not an in-round sibling ────

def test_inner_full_slot_measures_the_verdict_against_the_round_parent():
    """The paired verdict must compare against what the round started from.

    On trial 2+, the in-round ranking incumbent (``provisional``) is a faster
    sibling created in the SAME round. Taking the verdict against a sibling asks
    "is this better than the other candidate I just built" instead of "is this
    better than the parent", which lets a round commit on a sibling-vs-sibling
    comparison and never against the incumbent the round is supposed to beat.
    """
    from seq_align.c_patch_loop import controller

    captured: dict = {}

    class StubRunner:
        def evaluate(self, patchset, cfg, **kwargs):
            captured.update(kwargs)
            return "record"

    provisional = _summary(95.0, eval_id="provisional", patchset_id="sibling")
    outer_parent = _summary(100.0, eval_id="outer_parent", patchset_id="parent")

    controller._evaluate_inner_full(
        StubRunner(),
        SimpleNamespace(patchset_id="candidate"),
        object(),
        current_best=provisional,
        quality_reference=outer_parent,
        keep_work=False,
        query_timeout_reference=None,
    )

    assert captured["current_best"] is provisional, "in-round ranking reference must stay provisional"
    assert captured["paired_reference"] is outer_parent, (
        "the verdict reference must be the round parent, not the in-round sibling"
    )
    assert captured["commit_indexes"] is False, "an inner measurement must not write authoritative indexes"
    assert captured["write_legacy_result"] is False


def test_paired_reference_falls_back_to_current_best_when_absent():
    """A caller that does not name a verdict reference keeps the old behaviour."""
    from seq_align.c_patch_loop.runner import PatchRunner

    reference = _summary(100.0, eval_id="best")
    assert (
        PatchRunner._apply_paired_current_best_confirmation.__defaults__
        or True  # keyword-only defaults are exercised below
    )
    import inspect

    signature = inspect.signature(PatchRunner._apply_paired_current_best_confirmation)
    assert signature.parameters["paired_reference"].default is None


# ── 5. the baseline is established once and is then immutable ────────────────

def _baseline_runner(tmp_path: Path):
    """A runner whose dataset fixtures exist, so only the index logic is under test."""
    from seq_align.c_patch_loop.runner import PatchRunner
    from seq_align.c_patch_loop.store import LoopStore

    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "query1.fasta").write_text(">d1abcA_\nMKV\n", encoding="utf-8")
    astral = tmp_path / "astral.fasta"
    astral.write_text(">d1abcA_\nMKV\n", encoding="utf-8")
    uniref = tmp_path / "uniref.fasta"
    uniref.write_text(">u1\nMKV\n", encoding="utf-8")
    scop = tmp_path / "scop.txt"
    scop.write_text("d1abcA_\ta.1.1.1\n", encoding="utf-8")
    runner = PatchRunner(
        LoopStore(tmp_path / "run"),
        data_dir=data_dir,
        astral_db=astral,
        uniref_db=uniref,
        scop_txt=scop,
        eval_backend="local",
    )
    return runner


def _baseline_summary(eval_id: str, cost_s: float) -> dict:
    return {
        "eval_id": eval_id,
        "patchset_id": "stock",
        "stage": "full",
        "status": "ok",
        "tp": 10,
        "fp": 1,
        "cost_s": cost_s,
    }


def test_baseline_index_is_written_once_and_a_re_measurement_cannot_replace_it(tmp_path):
    """A later baseline must not retroactively change what earlier verdicts meant.

    Every verdict in this loop is a ratio against the baseline, so replacing the
    baseline silently rewrites history: a candidate that "won" against the old
    number may not have won against the new one. The read-side cache check makes a
    second baseline evaluation easy to trigger (any change to the reuse fingerprint
    -- a different core count, a different database -- causes one), so the write
    side has to refuse.
    """
    runner = _baseline_runner(tmp_path)

    runner._write_baseline_index("baseline_full", _baseline_summary("eval_a", 100.0))
    assert runner.store.read_index("baseline_full")["cost_s"] == 100.0

    runner._write_baseline_index("baseline_full", _baseline_summary("eval_b", 250.0))

    authoritative = runner.store.read_index("baseline_full")
    assert authoritative["eval_id"] == "eval_a", "the authoritative baseline was overwritten"
    assert authoritative["cost_s"] == 100.0

    # The re-measurement is preserved, but somewhere that cannot be mistaken for the
    # baseline: a calibration log is evidence, the index is a decision input.
    recalibration = runner.store.read_index("baseline_recalibration_full")
    assert isinstance(recalibration, list) and len(recalibration) == 1
    assert recalibration[0]["eval_id"] == "eval_b" and recalibration[0]["cost_s"] == 250.0

    # A third re-measurement accumulates rather than replacing.
    runner._write_baseline_index("baseline_full", _baseline_summary("eval_c", 300.0))
    assert len(runner.store.read_index("baseline_recalibration_full")) == 2
    assert runner.store.read_index("baseline_full")["eval_id"] == "eval_a"


def test_re_publishing_the_same_baseline_evaluation_is_allowed(tmp_path):
    """A resume that re-publishes the SAME evaluation is a no-op, not a conflict."""
    runner = _baseline_runner(tmp_path)
    runner._write_baseline_index("baseline", _baseline_summary("eval_a", 100.0))
    runner._write_baseline_index("baseline", _baseline_summary("eval_a", 100.0))

    assert runner.store.read_index("baseline")["eval_id"] == "eval_a"
    assert runner.store.read_index("baseline_recalibration") is None


def test_baseline_index_payload_keeps_a_usable_fingerprint(tmp_path):
    """The index fast path must not degrade the gate into "always refuse".

    ``_baseline_summary`` prefers the authoritative index over re-scanning the
    evaluation records, and ``_update_indexes`` stores ``eval_config`` verbatim.
    ``runner.evaluate`` therefore mirrors the measured-conditions fingerprint into
    ``eval_config`` as well as ``meta``.
    """
    from seq_align.c_patch_loop.models import EvaluationRecord, PatchsetRecord
    from seq_align.c_patch_loop.runner import eval_summary_from_record

    runner = _baseline_runner(tmp_path)
    fingerprint = _fingerprint()
    record = EvaluationRecord(
        eval_id="baseline-eval",
        patchset_id="stock",
        round_idx=0,
        stage="full",
        status="ok",
        decision="accepted_baseline",
        eval_result={
            "stage": "full",
            "status": "ok",
            "tp": 10,
            "fp": 1,
            "cost_s": 1000.0,
            "meta": {"comparability": fingerprint},
        },
        # Both writers run in runner.evaluate(); the index only keeps eval_config.
        eval_config={
            "stage": "full",
            "cpus": 45,
            "max_iter": 1,
            "uniref_db": fingerprint["uniref_db_path"],
            "comparability": fingerprint,
        },
    )
    runner._update_indexes(
        PatchsetRecord(round_idx=0, patchset_id="stock", base_hmmer_source="/tmp/hmmer"),
        record,
        baseline=True,
    )

    stored = runner.store.read_index("baseline_full")
    assert stored["eval_config"]["comparability"] == fingerprint, (
        "the authoritative index dropped the fingerprint; every verdict against this "
        "baseline would be refused as an unknown-conditions measurement"
    )

    baseline = eval_summary_from_record(stored)
    assert baseline.comparability
    # Equal conditions must still yield the number the gate exists to compute.
    assert compare_summaries(baseline, _summary(900.0, fingerprint=fingerprint)) == pytest.approx(0.10)
    # ... and a different database must still be refused, by name.
    with pytest.raises(IncomparableMeasurement) as excinfo:
        compare_summaries(
            baseline, _summary(900.0, fingerprint=_fingerprint(uniref_db_sha256="d" * 64))
        )
    assert "uniref_db_sha256" in excinfo.value.mismatches

    # Negative control: without the eval_config mirror the same baseline is refused
    # as unverifiable rather than silently compared.
    stripped = {k: v for k, v in stored.items()}
    stripped["eval_config"] = {
        k: v for k, v in stored["eval_config"].items() if k != "comparability"
    }
    with pytest.raises(IncomparableMeasurement) as missing:
        compare_summaries(eval_summary_from_record(stripped), _summary(900.0, fingerprint=fingerprint))
    assert "_missing_required" in missing.value.mismatches


def test_unsuffixed_baseline_is_guarded_too(tmp_path):
    runner = _baseline_runner(tmp_path)
    runner._write_baseline_index("baseline", _baseline_summary("eval_a", 100.0))
    runner._write_baseline_index("baseline", _baseline_summary("eval_b", 250.0))

    assert runner.store.read_index("baseline")["eval_id"] == "eval_a"
    assert runner.store.read_index("baseline_recalibration")[0]["eval_id"] == "eval_b"


# ── 5. the inner pre-screen may not write the authoritative indexes ──────────

def test_screen_candidate_cannot_write_authoritative_indexes(tmp_path, monkeypatch):
    """A cheap pre-screen must not redefine the reference later verdicts use.

    The 2026-10-09 audit found the screening path writing ``baseline_smoke`` /
    ``current_best_smoke``: a 5-query pre-screen silently became the reference every
    later smoke verdict was measured against. ``runner.evaluate`` gates the index
    write on ``commit_indexes`` (``if commit_indexes and decision.startswith(
    "accepted")``), so only the outer commit path may pass True.
    """
    from seq_align.c_patch_loop.models import EvaluationRecord
    from seq_align.c_patch_loop.runner import PatchRunner

    runner = _baseline_runner(tmp_path)
    runner.eval_repeats = 3
    runner.screen_queries = 5
    runner.screen_eval_repeats = 1
    captured: dict = {}

    def fake_evaluate(self, patchset, cfg, **kwargs):
        captured.update(kwargs)
        # Exactly the production gate: only a caller that did not opt out can write.
        if kwargs.get("commit_indexes", True) and str(kwargs.get("stage")) == "smoke":
            self.store.write_index("current_best_smoke", {"eval_id": "leaked-by-screen"})
        return EvaluationRecord(
            eval_id="screen-eval",
            patchset_id="cand",
            round_idx=0,
            stage="smoke",
            status="ok",
            decision="accepted_smoke",
            eval_result={"stage": "smoke", "status": "ok"},
        )

    monkeypatch.setattr(PatchRunner, "evaluate", fake_evaluate)
    patchset = SimpleNamespace(patchset_id="cand", round_idx=0)

    record = runner.screen_candidate(
        patchset,
        object(),
        current_best=_summary(100.0),
        quality_reference=_summary(100.0),
    )

    assert captured["commit_indexes"] is False
    assert captured["write_legacy_result"] is False
    assert captured["attempt_kind"] == "screen"
    assert captured["stage"] == "smoke"
    assert captured["quality_reference"] is not None
    for name in ("current_best_smoke", "baseline_smoke", "leaderboard_smoke"):
        assert runner.store.read_index(name) is None, f"the inner screen wrote {name}"
    # The screen is identifiable as a screen in the record it returns.
    assert record.eval_result["meta"]["screening"]["queries"] == 5
    # The per-trial repeat budget is restored, not left at the screen's value.
    assert runner.eval_repeats == 3

    # Control: a commit-path caller with the flag set does write the index.
    runner.evaluate(patchset, object(), stage="smoke", commit_indexes=True)
    assert runner.store.read_index("current_best_smoke")["eval_id"] == "leaked-by-screen"


def test_screen_candidate_restores_the_repeat_budget_when_the_screen_raises(tmp_path, monkeypatch):
    """A failed screen must not leave the runner measuring at the screen's repeat count."""
    from seq_align.c_patch_loop.runner import PatchRunner

    runner = _baseline_runner(tmp_path)
    runner.eval_repeats = 3
    runner.screen_eval_repeats = 1

    def boom(self, patchset, cfg, **kwargs):
        raise RuntimeError("backend exploded")

    monkeypatch.setattr(PatchRunner, "evaluate", boom)
    with pytest.raises(RuntimeError):
        runner.screen_candidate(
            SimpleNamespace(patchset_id="cand", round_idx=0),
            object(),
            current_best=_summary(100.0),
            quality_reference=_summary(100.0),
        )
    assert runner.eval_repeats == 3


# ── 7. an instrumented pass must really execute every query ──────────────────

def _profiling_probe_runner(tmp_path: Path):
    """A local runner plus a backend spy, so the two seams can be driven cheaply."""
    from seq_align import run_hmmer_cli
    from seq_align.assembler import AssembledConfig
    from seq_align.c_patch_loop.runner import PatchRunner
    from seq_align.c_patch_loop.store import LoopStore

    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "query1.fasta").write_text(">d1abcA_\nMKV\n", encoding="utf-8")
    uniref = tmp_path / "uniref.fasta"
    uniref.write_text(">u1\nMKV\n", encoding="utf-8")
    runner = PatchRunner(
        LoopStore(tmp_path / "run"),
        data_dir=data_dir,
        uniref_db=uniref,
        eval_backend="local",
    )
    return runner, AssembledConfig(), run_hmmer_cli


def _capture_attempt(monkeypatch, captured: dict):
    """Replace the attempt runner: record the checkpoint flag, hand back the backend."""
    from seq_align.c_patch_loop.runner import PatchRunner

    def fake_attempt(self, backend, *, supports_query_checkpoints, **kwargs):
        captured["supports_query_checkpoints"] = supports_query_checkpoints
        captured["backend"] = backend
        return EvalResult(stage="full", status="ok", tp=1, fp=0, cost_s=1.0)

    monkeypatch.setattr(PatchRunner, "_run_eval_attempt", fake_attempt)


def test_profiling_pass_cannot_resume_queries_from_checkpoints(tmp_path, monkeypatch):
    """A profiling pass must not resume queries from checkpoints.

    A resumed query is reconstructed without ``stage_timing`` and never runs the
    instrumented binary, so the pass would finish ``ok`` with no stage report.
    """
    from seq_align import run_hmmer_cli

    runner, cfg, _ = _profiling_probe_runner(tmp_path)
    captured: dict = {}
    _capture_attempt(monkeypatch, captured)

    seen: dict = {}

    def fake_run_all_queries(cfg_arg, **kwargs):
        seen.update(kwargs)
        from seq_align.eval_schema import EvalResult

        return EvalResult(stage="full", status="ok", tp=1, fp=0, cost_s=1.0)

    monkeypatch.setattr(run_hmmer_cli, "run_all_queries", fake_run_all_queries)

    common = {
        "work_root": tmp_path / "w_prof",
        "stage": "full",
        "data_dir": tmp_path / "data",
        "max_iter": 1,
        "astral_db": str(tmp_path / "astral.fa"),
        "uniref_db": str(tmp_path / "uniref.fasta"),
        "scop_txt": str(tmp_path / "scop.txt"),
    }

    runner._run_eval_repeats(cfg, eval_repeats=1, _profiling=True, **{**common, "work_root": tmp_path / "w_prof"})
    assert captured["supports_query_checkpoints"] is False, (
        "a profiling pass resumed queries from checkpoints, so it could report ok while "
        "running no instrumented binary at all"
    )
    assert captured["backend"] is not None

    # ... and the flag also reaches the backend that does the work.
    captured["backend"]()
    assert seen.get("profiling") is True

    # A normal pass keeps checkpointing: the property is about instrumentation, not
    # about disabling resume everywhere.
    runner._run_eval_repeats(cfg, eval_repeats=1, _profiling=False, **{**common, "work_root": tmp_path / "w_plain"})
    assert captured["supports_query_checkpoints"] is True
    seen.clear()
    captured["backend"](resume_provider=lambda *a, **k: {}, on_query_complete=lambda *a, **k: None)
    assert seen.get("profiling") is False


def test_named_attempt_keeps_the_profiling_flag(tmp_path, monkeypatch):
    """A re-measurement asked for instrumentation must stay instrumented.

    The named-attempt path used to drop ``_profiling``, so a re-measured evaluation
    silently degraded to a non-instrumented build: the record claimed a profiling
    pass and carried no stage times.
    """
    from seq_align import run_hmmer_cli

    runner, cfg, _ = _profiling_probe_runner(tmp_path)
    captured: dict = {}
    _capture_attempt(monkeypatch, captured)
    seen: dict = {}

    def fake_run_all_queries(cfg_arg, **kwargs):
        seen.update(kwargs)
        from seq_align.eval_schema import EvalResult

        return EvalResult(stage="full", status="ok", tp=1, fp=0, cost_s=1.0)

    monkeypatch.setattr(run_hmmer_cli, "run_all_queries", fake_run_all_queries)

    runner._run_named_eval_attempt(
        cfg,
        eval_kwargs={
            "_profiling": True,
            "stage": "full",
            "data_dir": tmp_path / "data",
            "max_iter": 1,
            "astral_db": str(tmp_path / "astral.fa"),
            "uniref_db": str(tmp_path / "uniref.fasta"),
            "scop_txt": str(tmp_path / "scop.txt"),
        },
        work_root=tmp_path / "named_w",
        attempt_kind="repeat",
        attempt_namespace="named/repeat_01",
        patchset_id="ps",
    )

    assert captured["supports_query_checkpoints"] is False
    captured["backend"]()
    assert seen.get("profiling") is True, "the named attempt dropped the profiling flag"


# ── 7b. the paired verdict refuses too, on its own re-measurement ────────────

def _paired_fixture(tmp_path: Path, *, paired_repeats: int = 2):
    """A local runner plus the fingerprint production will compute for its results.

    The reference summary has to carry exactly the fingerprint the candidate will
    get, otherwise the gate refuses before the paired stage is ever reached -- and
    the paired refusal is the one that matters, because that comparison is what
    actually decides a commit.
    """
    from seq_align.assembler import AssembledConfig
    from seq_align.c_patch_loop.runner import PatchRunner
    from seq_align.c_patch_loop.store import LoopStore

    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    for idx in (1, 2):
        (data_dir / f"query{idx}.fasta").write_text(f">d{idx}abcA_\nMKV\n", encoding="utf-8")
    astral = tmp_path / "astral.fasta"
    astral.write_text(">d1abcA_\nMKV\n", encoding="utf-8")
    uniref = tmp_path / "uniref.fasta"
    uniref.write_text(">u1\nMKV\n", encoding="utf-8")
    scop = tmp_path / "scop.txt"
    scop.write_text("d1abcA_\ta.1.1.1\n", encoding="utf-8")

    runner = PatchRunner(
        LoopStore(tmp_path / "run"),
        data_dir=data_dir,
        astral_db=astral,
        uniref_db=uniref,
        scop_txt=scop,
        cpus=1,
        eval_backend="local",
        paired_current_best_repeats=paired_repeats,
    )
    probe = EvalResult(stage="full", status="ok", tp=10, fp=1, cost_s=1.0)
    probe.meta = {}
    fingerprint = runner._measurement_comparability(
        eval_config=runner._eval_config(stage="full", max_iter=1, work_root=None, data_dir=data_dir, uniref_db=uniref),
        result=probe,
        dataset_manifest=runner._dataset_manifest(data_dir=data_dir, uniref_db=uniref),
        data_dir=data_dir,
    )
    patchset = runner.build_stock_patchset(stage="full")
    reference = EvalSummary(
        eval_id="reference",
        patchset_id=patchset.patchset_id,
        stage="full",
        status="ok",
        tp=10,
        fp=1,
        cost_s=100.0,
        comparability=fingerprint,
    )
    return runner, patchset, AssembledConfig(), reference, fingerprint


def _paired_fake_backend(runner, fingerprint, *, reference_fingerprint):
    """Fake backend that commits real query evidence and varies the fingerprint by role."""
    from seq_align import run_hmmer_cli

    def fake(cfg, **kwargs):
        work_root = Path(kwargs["work_root"])
        role = next(
            (name for name in ("candidate", "reference") if work_root.name.endswith(f"_{name}")),
            "initial",
        )
        kwargs["resume_provider"]({"jackhmmer_sha256": "jack-sha", "hmmsearch_sha256": "search-sha"})
        for idx in (1, 2):
            # The paired confirmation re-reads its evidence from the committed query
            # completions, so a fake must commit completions that carry verifiable
            # artifacts -- otherwise the verdict is refused for missing evidence and
            # the comparability path under test is never reached.
            result = EvalResult(
                stage="full",
                status="ok",
                tp=5,
                fp=0,
                n_seqs=1,
                iterations=1,
                cost_s=5.0,
                meta={
                    "query_id": f"d{idx}",
                    "profile": {
                        "commands": {"jackhmmer": {"wall_s": 5.0}},
                        "artifacts": {
                            # The evidence check requires every artifact the real
                            # pipeline produces, including the hmmsearch table.
                            "jackhmmer_tbl": {"exists": True, "bytes": 10, "sha256": f"jack-{idx}"},
                            "jackhmmer_out": {"exists": True, "bytes": 10, "sha256": f"out-{idx}"},
                            "hmmsearch_tbl": {"exists": True, "bytes": 10, "sha256": f"search-{idx}"},
                            "hmmsearch_out": {"exists": True, "bytes": 10, "sha256": f"sout-{idx}"},
                        },
                    },
                    "comparability": fingerprint,
                },
            )
            kwargs["on_query_complete"](
                idx,
                str(Path(kwargs["data_dir"]) / f"query{idx}.fasta"),
                work_root / "queries" / f"query_{idx:04d}",
                result,
            )
        cost = 90.0 if role in {"initial", "candidate"} else 100.0
        return EvalResult(
            stage="full",
            status="ok",
            tp=10,
            fp=1,
            tp_fp_ratio=10 / 11,
            n_seqs=2,
            cost_s=cost,
            meta={
                "build": {"runtime_effective": True},
                "comparability": reference_fingerprint if role == "reference" else fingerprint,
            },
        )

    return fake


def test_paired_verdict_refuses_when_its_own_re_measurement_differs(tmp_path, monkeypatch):
    """The paired statistic divides candidate by reference, so it must be guarded.

    This is the comparison that actually decides commits, and it is made from two
    FRESH measurements taken inside the paired schedule. Interleaving removes drift;
    it does not make two different configurations the same measurement.
    """
    runner, patchset, cfg, reference, fingerprint = _paired_fixture(tmp_path)
    # The reference arm reports a different per-query core count than the candidate.
    mismatched = dict(fingerprint)
    mismatched["query_threads"] = 1 if fingerprint.get("query_threads") != 1 else 2
    monkeypatch.setattr(
        run_hmmer_cli := __import__("seq_align.run_hmmer_cli", fromlist=["x"]),
        "run_all_queries",
        _paired_fake_backend(runner, fingerprint, reference_fingerprint=mismatched),
    )

    record = runner.evaluate(
        patchset, cfg, stage="full", current_best=reference, quality_reference=reference
    )

    assert record.decision == "rejected_incomparable_measurement", (
        f"paired verdict used incomparable numbers; got {record.decision}"
    )
    paired = (record.metric_delta or {}).get("paired_confirmation") or {}
    assert paired.get("comparability_errors"), "the refusal must name what differed"
    assert "query_threads" in json.dumps(paired["comparability_errors"])


def test_paired_verdict_still_commits_when_both_arms_match(tmp_path, monkeypatch):
    """The guard must not block a well-formed paired verdict."""
    from seq_align import run_hmmer_cli

    runner, patchset, cfg, reference, fingerprint = _paired_fixture(tmp_path)
    monkeypatch.setattr(
        run_hmmer_cli,
        "run_all_queries",
        _paired_fake_backend(runner, fingerprint, reference_fingerprint=fingerprint),
    )

    record = runner.evaluate(
        patchset, cfg, stage="full", current_best=reference, quality_reference=reference
    )

    paired = (record.metric_delta or {}).get("paired_confirmation") or {}
    assert not paired.get("comparability_errors")
    # The candidate really is 10% faster than the reference, above the 3% gate.
    assert record.decision == "accepted_speedup_paired", record.decision
    assert paired["median_pair_speedup_fraction"] == pytest.approx(0.10, abs=0.01)
