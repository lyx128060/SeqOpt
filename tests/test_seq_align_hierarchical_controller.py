from __future__ import annotations

import pytest

from pathlib import Path
import json

from seq_align.c_patch_loop import controller
from seq_align.c_patch_loop.models import (
    AlgorithmPlanProposal,
    EvaluationRecord,
    PatchsetRecord,
)
from seq_align.c_patch_loop.proposer import BaseProposer, FixtureProposer
from seq_align.c_patch_loop.runner import PatchRunner
from seq_align.c_patch_loop.store import LoopStore, record_hash, utc_now


# Canonical measurement conditions for the synthetic summaries/results these
# tests build. A cost may only be subtracted/divided from another when every one
# of these fields matches, so both sides of a comparison use the same
# fingerprint; a test that wants a refusal overrides exactly one key.
_COMPARABILITY: dict = {
    "schema_version": "test_comparability",
    "backend": "hmmer_cli_local",
    "partition": "compute",
    "allocated_cpus": 45,
    "query_threads": 8,
    "queries_in_job": 5,
    "stage": "full",
    "query_set_sha256": "q" * 64,
    "max_iter": 1,
    "uniref_db_path": "/db/uniref90.fasta",
    "uniref_db_sha256": "u" * 64,
    "astral_db_path": "/db/astral.fa",
    "astral_db_sha256": "a" * 64,
    "scop_txt_path": "/db/scop.txt",
    "scop_txt_sha256": "s" * 64,
    "hmmer_source_sha256": "h" * 64,
    "eval_repeats": 1,
    "paired_current_best_repeats": 0,
    "bundle_id": "b" * 64,
    "build_context": "test",
}


def _comparable(**overrides: object) -> dict:
    """A fingerprint whose conditions match, so the decision logic is exercised.

    Override a key to construct a deliberate mismatch (cpus, query set, stage,
    database). ``stage`` must be the stage actually evaluated; derive it from the
    helper's argument when one helper serves both ``smoke`` and ``full``.
    """
    return {**_COMPARABILITY, **overrides}


class _OuterOnlyProposer(BaseProposer):
    def propose_algorithm_plan(self, pack, **kwargs):
        proposal = AlgorithmPlanProposal(
            algorithm_family="pipeline_flow_strategy",
            baseline_algorithm_summary="Stock HMMER runs the exact comparison pipeline.",
            proposed_algorithm_change=(
                "Reuse exact comparison decisions in a worker-local decision surface, "
                "with stock fallback for every uncertain target."
            ),
            speed_mechanism="Avoid repeated comparison work for already proven exact states.",
            quality_preservation_argument=(
                "TP and FP remain identical because every uncertain case runs the exact stock path."
            ),
            target_contracts=["pipeline_flow"],
            target_functions=["p7_evo_align_Run"],
            exact_fallback_strategy="Call p7_evo_align_stock_Run for every uncertain target.",
            profile_evidence_citation="No measured profile evidence is available.",
        )
        return proposal, {"provider": "test"}

    def propose(self, pack, **kwargs):  # pragma: no cover - inner slot is stubbed
        raise AssertionError("inner proposer should be stubbed")


class _LedgerRunner(PatchRunner):
    def __init__(self, store: LoopStore):
        super().__init__(store, eval_backend="local")
        self.commits: list[str] = []

    def evaluate_stock_baseline(self, *, stage: str, keep_work: bool = False):
        patchset = self.build_stock_patchset(stage="full")
        record = _evaluation(patchset, cost_s=100.0, tp=10, fp=1, baseline=True)
        self.store.put_evaluation(record)
        self._update_indexes(patchset, record, baseline=True)
        return record

    def commit_evaluation(self, patchset, record):
        self.commits.append(record.eval_id)
        self._update_indexes(patchset, record, baseline=False)


class _RepairingFixtureProposer(FixtureProposer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.repair_feedback: list[str] = []

    def repair(self, pack, *, feedback: str, **kwargs):
        self.repair_feedback.append(feedback)
        return super().repair(pack, feedback=feedback, **kwargs)


def _evaluation(
    patchset: PatchsetRecord,
    *,
    cost_s: float,
    tp: int,
    fp: int,
    baseline: bool = False,
) -> EvaluationRecord:
    record = EvaluationRecord(
        patchset_id=patchset.patchset_id or "",
        round_idx=patchset.round_idx,
        stage="full",
        status="ok",
        decision="accepted_baseline" if baseline else "rejected_not_faster",
        eval_result={
            "stage": "full",
            "status": "ok",
            "tp": tp,
            "fp": fp,
            "cost_s": cost_s,
            "meta": {
                "query_results": [
                    {
                        "query_index": 0,
                        "query_id": "query-0",
                        "status": "ok",
                        "cost_s": cost_s,
                    }
                ],
                "comparability": _comparable(stage="full"),
            },
        },
        created_at=utc_now(),
    )
    record.eval_id = record_hash(record, id_field="eval_id")
    return record


def test_hierarchical_inner_trials_are_sequential_and_commit_only_at_outer_end(
    tmp_path: Path,
    monkeypatch,
):
    store = LoopStore(tmp_path / "run")
    runner = _LedgerRunner(store)
    candidates = iter([(99.0, 10, 1), (50.0, 9, 1), (96.0, 10, 1)])
    observed_parents: list[float] = []
    timeout_references: list[Path | None] = []

    def fake_inner_slot(**kwargs):
        provisional = kwargs["provisional"]
        observed_parents.append(provisional.cost_s)
        timeout_references.append(kwargs["query_timeout_reference"])
        cost_s, tp, fp = next(candidates)
        patchset = PatchsetRecord(
            round_idx=kwargs["outer_idx"],
            parent_patchset_id=provisional.patchset_id,
            parent_version_id=provisional.patchset_id,
            algorithm_plan_id=kwargs["algorithm_plan"].algorithm_plan_id,
            base_hmmer_source=str(runner.hmmer_src_root),
            proposal_metadata={"inner_idx": kwargs["inner_idx"]},
            created_at=utc_now(),
        )
        patchset.patchset_id = record_hash(patchset, id_field="patchset_id")
        store.put_patchset(patchset)
        record = _evaluation(patchset, cost_s=cost_s, tp=tp, fp=fp)
        store.put_evaluation(record)
        row = {
            "outer_idx": kwargs["outer_idx"],
            "inner_idx": kwargs["inner_idx"],
            "status": "ok",
            "patchset_id": patchset.patchset_id,
            "final_eval_id": record.eval_id,
            "cost_s": cost_s,
            "tp": tp,
            "fp": fp,
        }
        controller._transition_slot(
            store,
            kwargs["invocation"],
            outer_idx=kwargs["outer_idx"],
            inner_idx=kwargs["inner_idx"],
            state="evaluated",
            result=row,
        )
        return row, record

    monkeypatch.setattr(controller, "_run_inner_full_slot", fake_inner_slot)
    rows = controller.run_hierarchical_loop(
        run_dir=store.run_dir,
        outer_rounds=1,
        inner_trials=3,
        proposer=_OuterOnlyProposer(),
        runner=runner,
    )

    # Trial 2 is faster but loses a TP, so trial 3 must still start from trial 1.
    assert observed_parents == [100.0, 99.0, 99.0]
    assert len(timeout_references) == 3
    assert timeout_references[0] is not None
    assert len({str(path) for path in timeout_references}) == 1
    assert runner.commits == [rows[-1]["winner_eval_id"]]
    assert rows[-1]["status"] == "committed"
    assert rows[-1]["speedup_fraction"] == 0.04
    assert store.read_index("current_speed_best_full")["cost_s"] == 96.0


def test_hierarchical_outer_does_not_commit_subthreshold_provisional(
    tmp_path: Path,
    monkeypatch,
):
    store = LoopStore(tmp_path / "run")
    runner = _LedgerRunner(store)

    def fake_inner_slot(**kwargs):
        provisional = kwargs["provisional"]
        cost_s = 99.0 if kwargs["inner_idx"] == 1 else 98.0
        patchset = PatchsetRecord(
            round_idx=1,
            parent_patchset_id=provisional.patchset_id,
            parent_version_id=provisional.patchset_id,
            algorithm_plan_id=kwargs["algorithm_plan"].algorithm_plan_id,
            base_hmmer_source=str(runner.hmmer_src_root),
            proposal_metadata={"inner_idx": kwargs["inner_idx"]},
            created_at=utc_now(),
        )
        patchset.patchset_id = record_hash(patchset, id_field="patchset_id")
        store.put_patchset(patchset)
        record = _evaluation(patchset, cost_s=cost_s, tp=10, fp=1)
        store.put_evaluation(record)
        row = {
            "outer_idx": 1,
            "inner_idx": kwargs["inner_idx"],
            "status": "ok",
            "patchset_id": patchset.patchset_id,
            "final_eval_id": record.eval_id,
        }
        controller._transition_slot(
            store,
            kwargs["invocation"],
            outer_idx=1,
            inner_idx=kwargs["inner_idx"],
            state="evaluated",
            result=row,
        )
        return row, record

    monkeypatch.setattr(controller, "_run_inner_full_slot", fake_inner_slot)
    rows = controller.run_hierarchical_loop(
        run_dir=store.run_dir,
        outer_rounds=1,
        inner_trials=2,
        proposer=_OuterOnlyProposer(),
        runner=runner,
    )

    assert runner.commits == []
    assert rows[-1]["status"] == "not_committed"
    assert rows[-1]["speedup_fraction"] == 0.02
    assert store.read_index("current_speed_best_full")["cost_s"] == 100.0


def test_base_runner_outer_commit_publishes_canonical_and_legacy_indexes(tmp_path: Path):
    store = LoopStore(tmp_path / "run")
    runner = PatchRunner(store, eval_backend="local")
    patchset = runner.build_stock_patchset(stage="full")
    record = _evaluation(patchset, cost_s=90.0, tp=10, fp=1)
    store.put_evaluation(record)

    runner.commit_evaluation(patchset, record)

    assert store.read_index("current_speed_best_full")["eval_id"] == record.eval_id
    accepted = json.loads((store.run_dir / "accepted_eval.json").read_text(encoding="utf-8"))
    assert accepted["meta"]["hierarchical_outer_commit"] is True
    assert accepted["meta"]["c_patch_loop_eval_id"] == record.eval_id
    assert json.loads((store.run_dir / "accepted_mods.json").read_text(encoding="utf-8")) == []


@pytest.mark.requires_hmmer_source
def test_hierarchical_fixture_builds_complete_inner_lineage(
    tmp_path: Path,
    monkeypatch,
):
    store = LoopStore(tmp_path / "run")
    runner = _LedgerRunner(store)
    runner.require_algorithmic_patch = False

    def fake_evaluate(patchset, cfg, **kwargs):
        assert kwargs["stage"] == "full"
        assert kwargs["commit_indexes"] is False
        assert kwargs["write_legacy_result"] is False
        record = _evaluation(patchset, cost_s=95.0, tp=10, fp=1)
        record.build = {"runtime_effective": True}
        record.decision = "accepted_speedup"
        record.eval_id = record_hash(record, id_field="eval_id")
        store.put_evaluation(record)
        return record

    monkeypatch.setattr(runner, "evaluate", fake_evaluate)
    rows = controller.run_hierarchical_loop(
        run_dir=store.run_dir,
        outer_rounds=1,
        inner_trials=1,
        proposer=FixtureProposer(),
        runner=runner,
        require_algorithmic_patch=False,
    )

    assert rows[-2]["status"] == "ok"
    assert rows[-1]["status"] == "committed"
    assert len(store.iter_algorithm_plans()) == 1
    assert len(store.iter_patch_plans()) == 1
    assert len(store.iter_implementation_trials()) == 1
    patchset = store.get_patchset(rows[-2]["patchset_id"])
    assert patchset.parent_version_id
    assert patchset.algorithm_plan_id
    assert patchset.implementation_trial_id
    assert sorted(head.state for head in store.slot_heads().values()) == [
        "committed",
        "evaluated",
    ]


@pytest.mark.requires_hmmer_source
def test_hierarchical_full_compile_error_repairs_in_same_inner_slot(
    tmp_path: Path,
    monkeypatch,
):
    store = LoopStore(tmp_path / "run")
    runner = _LedgerRunner(store)
    runner.require_algorithmic_patch = False
    proposer = _RepairingFixtureProposer()
    evaluated_patchsets: list[str] = []

    def fake_evaluate(patchset, cfg, **kwargs):
        evaluated_patchsets.append(patchset.patchset_id)
        if len(evaluated_patchsets) == 1:
            record = EvaluationRecord(
                patchset_id=patchset.patchset_id,
                round_idx=patchset.round_idx,
                stage="full",
                status="compile_error",
                decision="rejected_compile_error",
                errors=["p7_evo_iter.c:78: error: too many arguments to function p7_oprofile_Clone"],
                eval_result={
                    "stage": "full",
                    "status": "compile_error",
                    "tp": 0,
                    "fp": 0,
                    "cost_s": 0.0,
                },
                build={"status": "compile_error"},
                created_at=utc_now(),
            )
        else:
            record = _evaluation(patchset, cost_s=95.0, tp=10, fp=1)
            record.build = {"runtime_effective": True}
            record.decision = "accepted_speedup"
        record.eval_id = record_hash(record, id_field="eval_id")
        store.put_evaluation(record)
        return record

    monkeypatch.setattr(runner, "evaluate", fake_evaluate)
    rows = controller.run_hierarchical_loop(
        run_dir=store.run_dir,
        outer_rounds=1,
        inner_trials=1,
        proposer=proposer,
        runner=runner,
        require_algorithmic_patch=False,
        max_repair_attempts=2,
    )

    assert len(evaluated_patchsets) == 2
    assert evaluated_patchsets[0] != evaluated_patchsets[1]
    assert len(proposer.repair_feedback) == 1
    assert "too many arguments to function p7_oprofile_Clone" in proposer.repair_feedback[0]
    assert rows[-2]["inner_idx"] == 1
    assert rows[-2]["status"] == "ok"
    states = [event["state"] for event in store.iter_slot_events() if event.get("inner_idx") == 1]
    assert states.count("repairing") == 1
    assert states.count("full_running") == 2
    inner_head = next(head for head in store.slot_heads().values() if head.inner_idx == 1)
    assert inner_head.state == "evaluated"


def _empty_context_pack(store: LoopStore):
    """Minimal context pack for tests that only need the algorithm-plan binding."""
    from seq_align.c_patch_loop.models import ContextPackRecord

    return ContextPackRecord(
        round_idx=1,
        run_dir=str(store.run_dir),
        objective={"primary_metric": "cost_s"},
        allowed_c_contracts=[],
        source_excerpts=[],
    )


def _stage_evaluation(patchset: PatchsetRecord, *, cost_s: float, tp: int, fp: int, stage: str, baseline: bool = False):
    """An evaluation record hashed *after* its stage is set, so the store accepts it."""
    record = _evaluation(patchset, cost_s=cost_s, tp=tp, fp=fp, baseline=baseline)
    record.stage = stage
    record.eval_result["stage"] = stage
    # The fingerprint has to name the stage that was actually measured, or the
    # smoke screen and the full measurement would look comparable.
    record.eval_result["meta"]["comparability"] = _comparable(stage=stage)
    record.eval_id = record_hash(record, id_field="eval_id")
    return record


def test_inner_screen_rejects_without_paying_for_the_full_measurement(tmp_path: Path, monkeypatch):
    """The cheap pre-screen hard-rejects only against an explicit wide band.

    The screen is ordering-only by default: its tiny query subset is far too
    noisy to be a verdict. A candidate may only be rejected before the full
    measurement when an out-of-band ``screen_reject_speedup_fraction`` says the
    screen result is slower by more than that wide margin. Here 100s -> 140s is
    -0.40, beyond the explicit 0.30 band, so the full repeats+paired pass is
    skipped entirely.
    """
    store = LoopStore(tmp_path / "run")
    runner = PatchRunner(store, eval_backend="local", screen_inner_trials=True, screen_eval_repeats=1)
    invocation = controller._open_invocation(store, runner, {"mode": "screening_test", "rounds": 1})
    patchset = PatchsetRecord(
        round_idx=1,
        base_hmmer_source=str(runner.hmmer_src_root),
        created_at=utc_now(),
    )
    patchset.patchset_id = record_hash(patchset, id_field="patchset_id")
    store.put_patchset(patchset)
    # Enter the slot past authoring/static verification (its own unit under test),
    # so the trial reaches exactly the screen gate with this patchset.
    static_record = _stage_evaluation(patchset, cost_s=0.0, tp=10, fp=1, stage="static_verify")
    store.put_evaluation(static_record)
    controller._transition_slot(
        store,
        invocation,
        outer_idx=1,
        inner_idx=1,
        state="static_verified",
        result={
            "status": "static_verified",
            "patchset_id": patchset.patchset_id,
            "static_eval_id": static_record.eval_id,
        },
    )

    full_calls: list[str] = []

    def fail_full(*args, **kwargs):  # pragma: no cover - must not be reached
        full_calls.append("full")
        raise AssertionError("the full measurement ran despite a screen rejection")

    monkeypatch.setattr(controller, "_evaluate_inner_full", fail_full)

    smoke_record = _stage_evaluation(patchset, cost_s=100.0, tp=10, fp=1, stage="smoke", baseline=True)
    slow_screen = _stage_evaluation(patchset, cost_s=140.0, tp=10, fp=1, stage="smoke")
    store.put_evaluation(slow_screen)

    monkeypatch.setattr(
        controller,
        "_screen_inner_candidate",
        lambda **kwargs: (slow_screen, controller.eval_summary_from_record(smoke_record)),
    )

    proposal = AlgorithmPlanProposal(
        algorithm_family="pipeline_flow_strategy",
        baseline_algorithm_summary="stock",
        proposed_algorithm_change="worker-local exact comparison reuse with stock fallback",
        speed_mechanism="avoid repeated comparison work",
        quality_preservation_argument="exact stock fallback keeps TP/FP identical",
        target_contracts=["pipeline_flow"],
        target_functions=["p7_evo_align_Run"],
        exact_fallback_strategy="stock fallback",
        profile_evidence_citation="test",
    )
    plan = runner.build_algorithm_plan(proposal, _empty_context_pack(store), {})
    row, record = controller._run_inner_full_slot(
        store=store,
        runner=runner,
        invocation=invocation,
        proposer=object(),
        algorithm_proposal=proposal,
        algorithm_plan=plan,
        outer_idx=1,
        inner_idx=1,
        outer_parent=controller.eval_summary_from_record(smoke_record),
        provisional=controller.eval_summary_from_record(smoke_record),
        hmmer_src_root=None,
        keep_work=False,
        max_repair_attempts=0,
        min_commit_speedup_fraction=0.10,
        eval_repeats=2,
        query_timeout_reference=None,
        screen_inner_trials=True,
        screen_min_speedup_fraction=0.5,
        screen_reject_speedup_fraction=0.3,
        screen_eval_repeats=1,
    )

    assert full_calls == []
    assert row["decision"] == "rejected_screening_not_faster"
    assert row["screening"] is True
    assert row["screen_hard_reject_fraction"] == 0.3
    assert row["screen_required_speedup_fraction"] == 0.05
    assert row["screen_speedup_fraction"] < 0  # 100 -> 140 is slower
    assert row["implementation_valid"] is False
    smoke_summary = controller.eval_summary_from_record(smoke_record)
    record_summary = controller.eval_summary_from_record(record)
    # A screened candidate is never committable on the screen's own numbers.
    assert controller._candidate_eligible(record, record_summary, smoke_summary) is False


def test_inner_screen_only_orders_by_default_and_still_pays_for_the_full_measurement(
    tmp_path: Path, monkeypatch
):
    """With the default threshold (0.0) the screen must not reject anything.

    The screen's small subset is ordering evidence, not a verdict: a candidate
    whose screen looks slower still gets the authoritative full measurement, and
    the trial must never be labelled ``rejected_screening_not_faster``.
    """
    store = LoopStore(tmp_path / "run")
    runner = PatchRunner(store, eval_backend="local", screen_inner_trials=True, screen_eval_repeats=1)
    invocation = controller._open_invocation(store, runner, {"mode": "screening_default_test", "rounds": 1})
    patchset = PatchsetRecord(
        round_idx=1,
        base_hmmer_source=str(runner.hmmer_src_root),
        created_at=utc_now(),
    )
    patchset.patchset_id = record_hash(patchset, id_field="patchset_id")
    store.put_patchset(patchset)
    static_record = _stage_evaluation(patchset, cost_s=0.0, tp=10, fp=1, stage="static_verify")
    store.put_evaluation(static_record)
    controller._transition_slot(
        store,
        invocation,
        outer_idx=1,
        inner_idx=1,
        state="static_verified",
        result={
            "status": "static_verified",
            "patchset_id": patchset.patchset_id,
            "static_eval_id": static_record.eval_id,
        },
    )

    full_calls: list[str] = []

    def fake_full(*args, **kwargs):
        full_calls.append("full")
        record = _stage_evaluation(patchset, cost_s=95.0, tp=10, fp=1, stage="full")
        record.decision = "accepted_speedup"
        record.eval_id = record_hash(record, id_field="eval_id")
        store.put_evaluation(record)
        return record

    monkeypatch.setattr(controller, "_evaluate_inner_full", fake_full)

    smoke_record = _stage_evaluation(patchset, cost_s=100.0, tp=10, fp=1, stage="smoke", baseline=True)
    slow_screen = _stage_evaluation(patchset, cost_s=140.0, tp=10, fp=1, stage="smoke")
    store.put_evaluation(slow_screen)

    monkeypatch.setattr(
        controller,
        "_screen_inner_candidate",
        lambda **kwargs: (slow_screen, controller.eval_summary_from_record(smoke_record)),
    )

    proposal = AlgorithmPlanProposal(
        algorithm_family="pipeline_flow_strategy",
        baseline_algorithm_summary="stock",
        proposed_algorithm_change="worker-local exact comparison reuse with stock fallback",
        speed_mechanism="avoid repeated comparison work",
        quality_preservation_argument="exact stock fallback keeps TP/FP identical",
        target_contracts=["pipeline_flow"],
        target_functions=["p7_evo_align_Run"],
        exact_fallback_strategy="stock fallback",
        profile_evidence_citation="test",
    )
    plan = runner.build_algorithm_plan(proposal, _empty_context_pack(store), {})
    row, record = controller._run_inner_full_slot(
        store=store,
        runner=runner,
        invocation=invocation,
        proposer=object(),
        algorithm_proposal=proposal,
        algorithm_plan=plan,
        outer_idx=1,
        inner_idx=1,
        outer_parent=controller.eval_summary_from_record(smoke_record),
        provisional=controller.eval_summary_from_record(smoke_record),
        hmmer_src_root=None,
        keep_work=False,
        max_repair_attempts=0,
        min_commit_speedup_fraction=0.10,
        eval_repeats=2,
        query_timeout_reference=None,
        screen_inner_trials=True,
        screen_min_speedup_fraction=0.5,
        screen_eval_repeats=1,
    )

    assert full_calls == ["full"]
    assert row["decision"] != "rejected_screening_not_faster"
    assert row.get("screening") is not True
    assert record is not None and record.decision == "accepted_speedup"


def test_screened_query_dir_caps_the_subset_and_keeps_the_reference_db(tmp_path: Path):
    """The screen must use a bounded, reproducible query subset."""
    store = LoopStore(tmp_path / "run")
    data_dir = tmp_path / "data"
    smoke = data_dir / "smoke"
    smoke.mkdir(parents=True)
    for idx in range(1, 9):
        (smoke / f"all_samples.part_{idx:03d}.fasta").write_text(f">q{idx}\nMKV\n", encoding="utf-8")
    (smoke / "uniref_smoke.fasta").write_text(">u1\nMKV\n", encoding="utf-8")
    (smoke / "query_manifest.json").write_text("{}", encoding="utf-8")

    runner = PatchRunner(store, data_dir=data_dir, screen_queries=3)
    subset = runner._data_dir_for_stage("smoke")

    assert subset.name == "smoke_screen_3"
    assert len(sorted(subset.glob("all_samples.part_*.fasta"))) == 3
    assert (subset / "uniref_smoke.fasta").is_file()
    # Deterministic on a second call: same directory, same first N queries.
    assert runner._data_dir_for_stage("smoke") == subset
