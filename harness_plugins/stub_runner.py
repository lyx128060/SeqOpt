"""Deterministic, cluster-free evaluation backend for exercising the callback loop.

This module exists ONLY so the agent-facing protocol (needs_proposal -> proposal ->
resume -> commit) can be driven end-to-end on a laptop, with no Slurm, no HMMER
build and no ASTRAL/UniRef databases. It is a development affordance and is never
used on the production path.

It mirrors the pattern already proven in
``tests/test_seq_align_hierarchical_controller.py`` (``_LedgerRunner`` plus a
monkeypatched ``controller._run_inner_full_slot``), but makes it installable from
a CLI so the loop can run in a *separate process* from the "agent", which is what
actually validates the stdout/stdin JSON contract.

Synthetic dynamics (per outer round, parent cost C, TP=10, FP=1):
  * inner trial 1..2: each is ~2% faster than the current provisional -> eligible
  * every 3rd inner trial: TP drops to 9 -> quality-invalid -> must NOT be inherited
  * so the committed winner is ~4% faster than the outer parent, clearing the 3% gate
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from seq_align.c_patch_loop import controller
from seq_align.c_patch_loop.models import EvaluationRecord, PatchsetRecord
from seq_align.c_patch_loop.runner import PatchRunner
from seq_align.c_patch_loop.store import LoopStore, record_hash, utc_now


def _synthetic_evaluation(
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
                        "query_id": "stub-query-0",
                        "status": "ok",
                        "cost_s": cost_s,
                    }
                ]
            },
        },
        created_at=utc_now(),
    )
    record.eval_id = record_hash(record, id_field="eval_id")
    return record


class StubRunner(PatchRunner):
    """PatchRunner with synthetic Full evaluations; no compiler, no Slurm."""

    baseline_cost_s = 100.0

    def __init__(self, store: LoopStore):
        super().__init__(store, eval_backend="local")
        self.commits: list[str] = []

    def evaluate_stock_baseline(self, *, stage: str, keep_work: bool = False):  # type: ignore[override]
        patchset = self.build_stock_patchset(stage="full")
        record = _synthetic_evaluation(
            patchset, cost_s=self.baseline_cost_s, tp=10, fp=1, baseline=True
        )
        self.store.put_evaluation(record)
        self._update_indexes(patchset, record, baseline=True)
        return record

    def commit_evaluation(self, patchset, record):  # type: ignore[override]
        self.commits.append(record.eval_id)
        self._update_indexes(patchset, record, baseline=False)


def make_inner_slot_stub():
    """Return a ``_run_inner_full_slot`` replacement producing synthetic trials."""

    def _stub(**kwargs: Any):
        store = kwargs["store"]
        runner = kwargs["runner"]
        invocation = kwargs["invocation"]
        outer_idx = kwargs["outer_idx"]
        inner_idx = kwargs["inner_idx"]
        provisional = kwargs["provisional"]
        algorithm_plan = kwargs["algorithm_plan"]

        # Every 3rd trial loses a TP: faster, but quality-invalid, so it must not
        # become the next trial's parent. This is the behaviour under test.
        quality_valid = inner_idx % 3 != 0
        cost_s = provisional.cost_s * 0.98
        tp = 10 if quality_valid else 9

        patchset = PatchsetRecord(
            round_idx=outer_idx,
            parent_patchset_id=provisional.patchset_id,
            parent_version_id=provisional.patchset_id,
            algorithm_plan_id=algorithm_plan.algorithm_plan_id,
            base_hmmer_source=str(runner.hmmer_src_root),
            proposal_metadata={"inner_idx": inner_idx, "stub": True},
            created_at=utc_now(),
        )
        patchset.patchset_id = record_hash(patchset, id_field="patchset_id")
        store.put_patchset(patchset)
        record = _synthetic_evaluation(patchset, cost_s=cost_s, tp=tp, fp=1)
        store.put_evaluation(record)

        row = {
            "outer_idx": outer_idx,
            "inner_idx": inner_idx,
            "status": "ok",
            "patchset_id": patchset.patchset_id,
            "final_eval_id": record.eval_id,
            "cost_s": cost_s,
            "tp": tp,
            "fp": 1,
            "stub": True,
        }
        controller._transition_slot(
            store,
            invocation,
            outer_idx=outer_idx,
            inner_idx=inner_idx,
            state="evaluated",
            result=row,
        )
        return row, record

    return _stub


# Captured at import so the patch below is reversible. Without this, a test that
# installs the synthetic inner slot leaks the patch into every later test in the
# same process (the real inner slot silently never runs again).
_ORIGINAL_INNER_SLOT = controller._run_inner_full_slot


def restore_inner_slot() -> None:
    """Undo ``install_stub_runner``'s patch to the controller inner slot."""
    controller._run_inner_full_slot = _ORIGINAL_INNER_SLOT  # type: ignore[assignment]


def install_stub_runner(run_dir: str | Path, *, patch_inner_slot: bool = True) -> StubRunner:
    """Return a synthetic-evaluation runner for ``run_dir``.

    ``patch_inner_slot=True`` (default) additionally replaces the whole inner slot
    with synthetic trials, which lets a full N-round loop run with no toolchain at
    all. Pass ``False`` to keep the *real* inner-slot control flow: the loop then
    stops at the first inner proposal request, which is how the callback yield
    behaviour is exercised without a compiler or Slurm.

    This mutates controller-module state. Callers that do not control the process
    lifetime (i.e. tests) must call :func:`restore_inner_slot` afterwards.
    """
    store = LoopStore(run_dir)
    if patch_inner_slot:
        controller._run_inner_full_slot = make_inner_slot_stub()  # type: ignore[assignment]
    else:
        restore_inner_slot()
    return StubRunner(store)
