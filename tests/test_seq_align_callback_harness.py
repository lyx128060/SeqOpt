"""Regression tests for the callback ("agent as proposer") bridge.

These run with NO cluster, NO Slurm, NO HMMER build and NO databases: the
evaluation backend is synthetic (``harness_plugins/stub_runner.py``). What they
assert is the *control-flow inversion*, which is the whole point of the migration:

1. The engine yields a proposal request instead of calling an in-process LLM.
2. ``NeedProposal`` is not swallowed by the ``except ProposalError`` handlers, so a
   yielded slot stays retryable (``interrupted``) rather than permanently blocked.
3. The engine re-attaches to the SAME invocation across yields, so the frozen outer
   algorithm plan and completed inner trials survive.
4. The repair path currently cannot resume authoring — documented and pinned here so
   a future engine fix flips this test rather than silently regressing.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_HARNESS = Path(__file__).resolve().parent.parent / "harness_plugins"
if str(_HARNESS) not in sys.path:
    sys.path.insert(0, str(_HARNESS))

from seq_align.c_patch_loop import controller  # noqa: E402
from seq_align.c_patch_loop.callback_proposer import (  # noqa: E402
    KIND_ALGORITHM_PLAN,
    KIND_INNER,
    CallbackProposer,
    NeedProposal,
)
from seq_align.c_patch_loop.runner import PatchRunner  # noqa: E402
from seq_align.c_patch_loop.store import LoopStore, record_hash  # noqa: E402
import stub_runner  # noqa: E402
from stub_runner import install_stub_runner, restore_inner_slot  # noqa: E402


# Canonical measurement conditions for the synthetic stub evaluations. The
# acceptance gate refuses to subtract two costs measured under different
# conditions, and the stub backend (``harness_plugins/stub_runner.py``) records no
# fingerprint of its own, so the fixture supplies one for every synthetic record.
# Without it the outer commit gate refuses the (identical) synthetic numbers as
# incomparable and no round can ever commit.
_STUB_COMPARABILITY: dict = {
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


def _stub_records_are_comparable(monkeypatch: pytest.MonkeyPatch) -> None:
    """Give every synthetic stub evaluation the same comparability fingerprint.

    ``stub_runner._synthetic_evaluation`` is the one place the stub builds an
    ``EvaluationRecord``; wrapping it here keeps the harness module untouched
    (it lives outside ``tests/``) while making the numbers the loop compares
    comparable by construction. Both the baseline and every inner trial get the
    identical fingerprint, which is what the gate requires.
    """
    original = stub_runner._synthetic_evaluation

    def fingerprinted(*args, **kwargs):
        record = original(*args, **kwargs)
        record.eval_result.setdefault("meta", {})["comparability"] = dict(_STUB_COMPARABILITY)
        # eval_id is a content hash, so it must be recomputed after the meta
        # change or the stored record no longer matches its own id.
        record.eval_id = record_hash(record, id_field="eval_id")
        return record

    monkeypatch.setattr(stub_runner, "_synthetic_evaluation", fingerprinted)


@pytest.fixture(autouse=True)
def _isolate_inner_slot():
    """``install_stub_runner`` patches controller module state; undo it per test."""
    yield
    restore_inner_slot()

PLAN = {
    "algorithm_family": "pipeline_flow_strategy",
    "component_name": "pipeline_flow",
    "target_function": "p7_evo_align_Run",
    "algorithm_change": (
        "Reuse exact comparison decisions in a worker-local decision surface, "
        "with stock fallback for every uncertain target."
    ),
    "speed_mechanism": "Avoid repeated comparison work for already proven exact states.",
    "quality_preservation_argument": (
        "TP and FP remain identical because every uncertain case runs the exact stock path."
    ),
    "exact_fallback_strategy": "Call the stock exact path for every uncertain target.",
    "profile_evidence_citation": "Measured dominant command 'hmmsearch'.",
}


def _write(tmp_path: Path, name: str, data: dict) -> Path:
    path = tmp_path / name
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def _inner_slot_states(store: LoopStore) -> dict[str, str]:
    return {k: h.state for k, h in store.slot_heads().items()}


def test_outer_yield_then_inner_yield_stays_retryable(tmp_path: Path):
    """The core invariant: yielding must not permanently block a slot."""
    run_dir = tmp_path / "run"
    runner = install_stub_runner(run_dir, patch_inner_slot=False)

    # --- call 1: the engine has nothing to work with, so it must yield. -------
    with pytest.raises(NeedProposal) as first:
        controller.run_hierarchical_loop(
            run_dir=run_dir,
            outer_rounds=2,
            inner_trials=3,
            proposer=CallbackProposer(kind=KIND_ALGORITHM_PLAN),
            runner=runner,
        )
    assert first.value.kind == KIND_ALGORITHM_PLAN

    # --- call 2: answer the outer plan; the engine then wants an inner trial. --
    plan_path = _write(tmp_path, "plan.json", PLAN)
    with pytest.raises(NeedProposal) as second:
        controller.run_hierarchical_loop(
            run_dir=run_dir,
            outer_rounds=2,
            inner_trials=3,
            proposer=CallbackProposer(plan_path, kind=KIND_ALGORITHM_PLAN, round_idx=1),
            runner=runner,
        )
    assert second.value.kind == KIND_INNER

    store = LoopStore(run_dir)
    states = _inner_slot_states(store)

    # The frozen outer plan survived the yield...
    outer_slots = {k: v for k, v in states.items() if k.endswith(":outer")}
    assert states[[k for k in outer_slots if ":001:outer" in k][0]] == "validated"
    # ...and the yielded inner slot is retryable, NOT blocked. It is
    # ``awaiting_proposal`` rather than ``interrupted`` precisely because the
    # deferral is recoverable waiting -- ``interrupted`` carrying a patchset would
    # be replayed instead of re-authored (see the G4 regression test).
    inner1 = [k for k in states if ":001:inner-001" in k][0]
    assert states[inner1] == "awaiting_proposal"

    # Exactly one invocation: the engine re-attached instead of restarting, so the
    # validated plan and the slot grid were not thrown away.
    assert len(list(store.iter_invocations())) == 1


def test_full_loop_commits_and_keeps_provenance(tmp_path: Path, monkeypatch):
    """N rounds driven one proposal at a time, with a clean provenance trail."""
    run_dir = tmp_path / "run"
    _stub_records_are_comparable(monkeypatch)
    runner = install_stub_runner(run_dir)  # synthetic inner trials
    plan_path = _write(tmp_path, "plan.json", PLAN)

    results = controller.run_hierarchical_loop(
        run_dir=run_dir,
        outer_rounds=2,
        inner_trials=3,
        # Unbound proposer: this test is about bookkeeping, so one file answers
        # every round. A real agent supplies a fresh file per round (see
        # harness_plugins/local_agent_driver.py, which binds --proposal-round).
        proposer=CallbackProposer(plan_path, kind=KIND_ALGORITHM_PLAN),
        runner=runner,
    )

    committed = [r for r in results if r.get("status") == "committed"]
    assert committed, "synthetic dynamics should clear the 3% gate at least once"
    for row in committed:
        assert row["speedup_fraction"] >= row["required_speedup_fraction"]
        assert row["committed_eval_id"]

    store = LoopStore(run_dir)
    states = _inner_slot_states(store)
    for outer_idx in (1, 2):
        inner = [k for k in states if f":{outer_idx:03d}:inner-" in k]
        assert len(inner) == 3
        assert all(states[k] == "evaluated" for k in inner), {
            k: states[k] for k in inner
        }
    # Every evaluated trial is backed by a patchset and an evaluation record.
    assert store.read_index("current_speed_best_full")["cost_s"] < 100.0


def test_stale_round_proposal_is_rejected_not_reused(tmp_path: Path):
    """A proposal authored for round 1 must not silently answer round 2."""
    run_dir = tmp_path / "run"
    runner = install_stub_runner(run_dir)
    plan_path = _write(tmp_path, "plan.json", PLAN)

    with pytest.raises(NeedProposal) as excinfo:
        controller.run_hierarchical_loop(
            run_dir=run_dir,
            outer_rounds=2,
            inner_trials=1,
            # Bound to round 1, but the engine advances past round 1 and asks for 2.
            proposer=CallbackProposer(plan_path, kind=KIND_ALGORITHM_PLAN, round_idx=1),
            runner=runner,
        )
    assert excinfo.value.kind == KIND_ALGORITHM_PLAN
    assert excinfo.value.round_idx == 2


# A diff that fails the *parser*: correct headers, but the body is not a diff.
UNAPPLICABLE_DIFF = (
    "--- a/p7_evo_align.c\n"
    "+++ b/p7_evo_align.c\n"
    "@@ -14,3 +14,4 @@\n"
    " struct p7_evo_align_s {\n"
    "   int reserved;\n"
    "+  int scratch_probe;\n"
    " };\n"
)

# A diff that parses but fails *static verification*: the target path is not the
# component contract's file, so the engine rejects it and enters the repair path.
# This is the state that used to be unrecoverable (G4).
STATIC_REJECT_DIFF = (
    "--- a/p7_evo_align.c\n"
    "+++ b/not_the_contract_target.c\n"
    "@@ -14,3 +14,4 @@\n"
    " struct p7_evo_align_s {\n"
    "   int reserved;\n"
    "+  int scratch_probe;\n"
    " };\n"
)


def _drive_to_inner_repair(tmp_path: Path, *, diff: str = UNAPPLICABLE_DIFF):
    """Run the loop up to the point where the engine rejects a supplied patch."""
    run_dir = tmp_path / "run"
    runner = install_stub_runner(run_dir, patch_inner_slot=False)
    plan_path = _write(tmp_path, "plan.json", PLAN)

    # 1) Answer the outer plan -> engine asks for the inner candidate.
    with pytest.raises(NeedProposal):
        controller.run_hierarchical_loop(
            run_dir=run_dir,
            outer_rounds=1,
            inner_trials=1,
            proposer=CallbackProposer(plan_path, kind=KIND_ALGORITHM_PLAN, round_idx=1),
            runner=runner,
        )

    # 2) Supply a patch the engine cannot accept -> it must hand back the reason.
    inner_path = _write(tmp_path, "inner.json", {**PLAN, "unified_diff": diff})
    with pytest.raises(NeedProposal) as repair:
        controller.run_hierarchical_loop(
            run_dir=run_dir,
            outer_rounds=1,
            inner_trials=1,
            proposer=CallbackProposer(inner_path, kind=KIND_INNER, round_idx=1),
            runner=runner,
        )
    return run_dir, runner, inner_path, repair.value


def test_rejected_patch_is_handed_back_to_the_agent(tmp_path: Path):
    """A patch the engine refuses must become actionable feedback, not a dead slot."""
    run_dir, _runner, _inner_path, request = _drive_to_inner_repair(tmp_path)

    assert request.kind == KIND_INNER
    assert request.feedback, "engine must return the rejection reason for self-correction"

    store = LoopStore(run_dir)
    states = _inner_slot_states(store)
    slot = [k for k in states if ":001:inner-001" in k][0]
    # ``awaiting_proposal`` is retryable: not blocked, and not mistaken for a
    # mid-evaluation crash that carries a rejected patchset to replay.
    assert states[slot] == "awaiting_proposal"


def test_static_rejection_is_re_authorable_not_fatal(tmp_path: Path):
    """G4 regression: a static-verification rejection must remain recoverable.

    Before the fix, the deferred slot carried a ``patchset_id``, so the next
    invocation took the patchset-resume branch, reconstructed the rejected
    patchset and died with an uncaught

        RuntimeError("failed to reconstruct current-best config: <component>:
                      Patch has no file header; expected a unified diff.")

    which surfaced as a terminal ``engine_error`` -- the agent could never fix its
    patch. Now the slot lands in ``awaiting_proposal``, which no resume-terminal
    set contains, so resuming asks for a corrected proposal instead.
    """
    run_dir, runner, inner_path, request = _drive_to_inner_repair(
        tmp_path, diff=STATIC_REJECT_DIFF
    )
    assert request.feedback, "the git-apply failure must reach the agent"

    store = LoopStore(run_dir)
    slot = [k for k in _inner_slot_states(store) if ":001:inner-001" in k][0]
    assert _inner_slot_states(store)[slot] == "awaiting_proposal"

    # Answering the deferral must re-ask, never crash and never replay the patch.
    with pytest.raises(NeedProposal) as again:
        controller.run_hierarchical_loop(
            run_dir=run_dir,
            outer_rounds=1,
            inner_trials=1,
            proposer=CallbackProposer(inner_path, kind=KIND_INNER, round_idx=1),
            runner=runner,
        )
    assert again.value.feedback
    assert len(list(store.iter_invocations())) == 1, "re-attach must survive a deferral"


def test_repeated_bad_patches_keep_yielding_instead_of_blocking(tmp_path: Path):
    """Answering a rejection with another bad patch must stay retryable forever.

    If a deferral were a ``ProposalError`` subclass, the engine's retry loop would
    exhaust and mark the slot ``blocked`` -- the agent would be stranded with no
    way to recover. This test is the guard against that regression.
    """
    run_dir, runner, inner_path, _request = _drive_to_inner_repair(tmp_path)

    with pytest.raises(NeedProposal) as again:
        controller.run_hierarchical_loop(
            run_dir=run_dir,
            outer_rounds=1,
            inner_trials=1,
            proposer=CallbackProposer(inner_path, kind=KIND_INNER, round_idx=1),
            runner=runner,
        )
    assert again.value.feedback

    store = LoopStore(run_dir)
    states = _inner_slot_states(store)
    slot = [k for k in states if ":001:inner-001" in k][0]
    assert states[slot] != "blocked"
    assert len(list(store.iter_invocations())) == 1


def _drift_fingerprint(monkeypatch) -> None:
    """Make the environment fingerprint change between calls."""
    monkeypatch.setattr(
        PatchRunner,
        "reuse_fingerprint",
        lambda self, *, stage, runtime_identity=None: "drifted-fingerprint",
    )


def test_explicit_session_id_survives_environment_drift(tmp_path: Path, monkeypatch):
    """G8: an explicit session id must outrank the environment fingerprint.

    The fingerprint answers "is a cached evaluation still reusable?". Invocation
    identity answers "which interrupted run does this call continue?". Conflating
    them means any environment drift silently orphans the slot grid and the loop
    restarts — which for a callback bridge is state loss.
    """
    run_dir = tmp_path / "run"
    runner = install_stub_runner(run_dir, patch_inner_slot=False)

    with pytest.raises(NeedProposal):
        controller.run_hierarchical_loop(
            run_dir=run_dir,
            outer_rounds=1,
            inner_trials=1,
            proposer=CallbackProposer(kind=KIND_ALGORITHM_PLAN),
            runner=runner,
            session_id="sess-A",
        )

    _drift_fingerprint(monkeypatch)
    drifted_runner = install_stub_runner(run_dir, patch_inner_slot=False)
    plan_path = _write(tmp_path, "plan.json", PLAN)

    with pytest.raises(NeedProposal) as second:
        controller.run_hierarchical_loop(
            run_dir=run_dir,
            outer_rounds=1,
            inner_trials=1,
            proposer=CallbackProposer(plan_path, kind=KIND_ALGORITHM_PLAN, round_idx=1),
            runner=drifted_runner,
            session_id="sess-A",
        )

    # Resumed rather than restarted: the engine moved past the plan to the inner
    # trial, on the same invocation.
    assert second.value.kind == KIND_INNER
    assert len(list(LoopStore(run_dir).iter_invocations())) == 1


def test_without_session_id_fingerprint_still_splits_identity(tmp_path: Path, monkeypatch):
    """The legacy contract must be preserved for callers that pass no session id."""
    run_dir = tmp_path / "run"
    runner = install_stub_runner(run_dir, patch_inner_slot=False)

    with pytest.raises(NeedProposal):
        controller.run_hierarchical_loop(
            run_dir=run_dir,
            outer_rounds=1,
            inner_trials=1,
            proposer=CallbackProposer(kind=KIND_ALGORITHM_PLAN),
            runner=runner,
        )

    _drift_fingerprint(monkeypatch)
    drifted_runner = install_stub_runner(run_dir, patch_inner_slot=False)
    with pytest.raises(NeedProposal):
        controller.run_hierarchical_loop(
            run_dir=run_dir,
            outer_rounds=1,
            inner_trials=1,
            proposer=CallbackProposer(kind=KIND_ALGORITHM_PLAN),
            runner=drifted_runner,
        )

    assert len(list(LoopStore(run_dir).iter_invocations())) == 2
