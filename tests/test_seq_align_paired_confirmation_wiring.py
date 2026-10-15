"""The paired-confirmation knob must be reachable AND binding (2026-10-08).

Context: every comparison the hierarchical loop makes is "freshly measured
candidate" vs "a cost stored by an earlier evaluation", so run-to-run timing
drift is charged to the patch. A deliberately score-neutral probe patch "won"
76% that way and was committed with `decision=accepted_speedup`.

`PatchRunner` already implements the fix (re-measure the current best
interleaved with the candidate -> `accepted_speedup_paired`), but the
hierarchical path hard-reset the knob to 0, so the machinery was dead code.
These tests pin the two halves of making it work:

1. the value reaches the runner (it used to be overwritten with 0), and
2. a paired refusal actually blocks the commit -- the commit gate recomputes
   its speedup from stored costs, so without this the knob is decorative.
"""

from __future__ import annotations

import pytest

from seq_align.c_patch_loop import controller
from seq_align.c_patch_loop.models import EvaluationRecord
from seq_align.c_patch_loop.runner import PatchRunner
from seq_align.c_patch_loop.store import LoopStore

PAIRED_REFUSALS = (
    "rejected_not_faster_paired",
    "rejected_paired_confirmation_error",
    "rejected_paired_incomplete_evidence",
    "rejected_paired_quality_regression",
)


class _Recorder:
    """Stands in for PatchRunner; records whether the loop forwarded the knob."""

    def __init__(self, store):
        self.store = store
        self.paired_current_best_repeats = -1
        self.min_speedup_fraction = None

    def evaluate_stock_baseline(self, **kwargs):
        raise AssertionError("this test must not reach the evaluator")


def _record(decision: str) -> EvaluationRecord:
    # model_construct skips validation: this test is about the decision filter,
    # not about how a record gets built.
    return EvaluationRecord.model_construct(decision=decision, build={})


def _summary() -> object:
    return type("S", (), {"tp": 33, "fp": 0, "cost_s": 13.2, "patchset_id": "p", "eval_id": "e"})()


@pytest.mark.parametrize("decision", PAIRED_REFUSALS)
def test_paired_refusal_blocks_commit_eligibility(decision, monkeypatch):
    monkeypatch.setattr(controller, "_quality_valid", lambda *a, **k: True)
    assert controller._candidate_eligible(_record(decision), _summary(), _summary()) is False


def test_plain_speedup_stays_eligible(monkeypatch):
    monkeypatch.setattr(controller, "_quality_valid", lambda *a, **k: True)
    assert controller._candidate_eligible(_record("accepted_speedup"), _summary(), _summary()) is True


def test_paired_acceptance_stays_eligible(monkeypatch):
    monkeypatch.setattr(controller, "_quality_valid", lambda *a, **k: True)
    assert controller._candidate_eligible(_record("accepted_speedup_paired"), _summary(), _summary()) is True


def test_hierarchical_loop_forwards_paired_repeats(tmp_path, monkeypatch):
    """The exact regression: this used to be hard-reset to 0 on the way in."""
    monkeypatch.setattr(controller, "_run_hierarchical_invocation", lambda *a, **k: [])
    runner = _Recorder(controller.LoopStore(tmp_path))
    controller.run_hierarchical_loop(
        run_dir=tmp_path,
        outer_rounds=0,
        proposer=object(),
        paired_current_best_repeats=2,
        runner=runner,
    )
    assert runner.paired_current_best_repeats == 2


def test_hierarchical_loop_defaults_to_legacy_behaviour(tmp_path, monkeypatch):
    monkeypatch.setattr(controller, "_run_hierarchical_invocation", lambda *a, **k: [])
    runner = _Recorder(controller.LoopStore(tmp_path))
    controller.run_hierarchical_loop(run_dir=tmp_path, outer_rounds=0, proposer=object(), runner=runner)
    assert runner.paired_current_best_repeats == 0


def test_hierarchical_loop_rejects_negative_repeats(tmp_path):
    with pytest.raises(ValueError, match="paired_current_best_repeats must be non-negative"):
        controller.run_hierarchical_loop(
            run_dir=tmp_path, outer_rounds=0, proposer=object(), paired_current_best_repeats=-1
        )


def test_patch_runner_stores_the_knob_and_clamps_negatives(tmp_path):
    """Guard the other end: the runner must keep an explicit value (and only
    clamp it at 0, which is what 'off' means)."""
    store = LoopStore(tmp_path)
    assert PatchRunner(store, paired_current_best_repeats=3).paired_current_best_repeats == 3
    assert PatchRunner(store, paired_current_best_repeats=-1).paired_current_best_repeats == 0
