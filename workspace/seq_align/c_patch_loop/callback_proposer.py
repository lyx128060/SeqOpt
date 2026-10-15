"""Callback proposer: let an external agent supply proposals across process boundaries.

This is the Phase-0 skeleton for moving the *whole* hierarchical optimization loop
into a DeepSeek-Harness / Cordis agent, without giving the agent control of the
control flow (the controller keeps the loop state; the agent is "called back"
whenever a proposal is needed).

Why this works with the existing engine (no controller rewrite required):

* ``run_hierarchical_loop`` wraps the invocation in ``except BaseException`` and
  finalizes the invocation as ``terminal_state="interrupted"`` before re-raising
  (controller.py, the ``_run_hierarchical_loop`` body).
* ``_open_invocation`` deliberately *re-attaches* to an existing invocation whose
  ``configuration_id`` matches and whose terminal state is ``partial`` /
  ``interrupted``, and idempotently materializes the outer x inner slot grid.
* Per-slot resume is already implemented: completed outer/inner slots are skipped,
  and the provisional parent chain is rebuilt from ``evaluated`` slots.

So a proposal request can be expressed as an exception that is NOT a
``ProposalError``: it propagates past every ``except ProposalError`` handler,
lands on the invocation-level ``except BaseException``, and the run is simply
resumed on the next CLI invocation.

CRITICAL INVARIANT
------------------
``NeedProposal`` MUST NOT subclass ``ProposalError`` or ``FatalProposalError``.
If it did, the callsites that convert proposal failures into *permanent slot
blocking* would swallow it:

* outer plan  -> controller.py ``except ProposalError`` -> ``_block_outer_and_inner``
* inner plan  -> controller.py ``_run_inner_full_slot`` ``except ProposalError`` -> slot "blocked"

A blocked slot is never retried, so the agent would lose the ability to correct
its proposal. As a plain ``Exception`` subclass, ``NeedProposal`` leaves the slot
``pending`` / ``interrupted`` and therefore retryable.

Proposal granularity
--------------------
One agent round-trip answers exactly one engine request, but a single JSON file
carries BOTH the inner patch plan and the inner diff, so an inner trial costs one
round-trip rather than two (mirrors the existing ``FileProposer``, which derives
the plan from the same document).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .file_proposer import FileProposer
from .models import (
    AlgorithmPlanProposal,
    ContextPackRecord,
    PatchPlanProposal,
    PatchProposal,
)
from .proposer import ProposalDeferred, ProposalError

# Request kinds the engine can ask an external agent for.
KIND_ALGORITHM_PLAN = "algorithm_plan"
KIND_INNER = "inner"  # answers both propose_plan() and propose() for one trial

VALID_KINDS = (KIND_ALGORITHM_PLAN, KIND_INNER)


class NeedProposal(ProposalDeferred):
    """The engine needs a proposal the callback did not have.

    Subclasses ``ProposalDeferred`` (which is a plain ``Exception``, NOT a
    ``ProposalError``) so the controller can recognise a deferral and mark the
    slot ``awaiting_proposal`` instead of blocking it permanently.
    """

    def __init__(
        self,
        *,
        kind: str,
        reason: str,
        feedback: str | None = None,
        round_idx: int | None = None,
        attempt_idx: int | None = None,
    ) -> None:
        self.kind = kind
        self.reason = reason
        self.feedback = feedback
        self.round_idx = round_idx
        self.attempt_idx = attempt_idx
        # Pass ``kind`` up as well: ProposalDeferred.__init__ assigns self.kind,
        # so calling it without the argument would clobber the value just set.
        super().__init__(self._message(), kind=kind)

    def _message(self) -> str:
        parts = [f"agent proposal required ({self.kind}): {self.reason}"]
        if self.round_idx is not None:
            parts.append(f"outer_round={self.round_idx}")
        if self.attempt_idx:
            parts.append(f"repair_attempt={self.attempt_idx}")
        return "; ".join(parts)

    def to_payload(self) -> dict[str, Any]:
        """Machine-readable request handed back to the harness agent."""
        payload: dict[str, Any] = {
            "status": "needs_proposal",
            "kind": self.kind,
            "reason": self.reason,
        }
        if self.round_idx is not None:
            payload["outer_round"] = self.round_idx
        if self.attempt_idx is not None:
            payload["repair_attempt"] = self.attempt_idx
        if self.feedback:
            # Feedback is the engine's own validation/compile output. Handing it
            # back verbatim is what makes the agent able to self-correct, which
            # is the whole point of moving the proposer out of this process.
            payload["feedback"] = self.feedback
        return payload


def _round_of(pack: Any) -> int | None:
    return getattr(pack, "round_idx", None)


class CallbackProposer(FileProposer):
    """Answers engine proposal requests from a file, or yields back to the agent.

    ``kind`` declares which engine request the supplied file answers. The engine's
    first real call after a resume reveals what it actually wants, so a mismatch
    is reported as a fresh ``NeedProposal`` (self-describing protocol).
    """

    provider_name = "callback"

    def __init__(
        self,
        proposal_path: str | Path | None = None,
        *,
        kind: str = KIND_ALGORITHM_PLAN,
        round_idx: int | None = None,
    ):
        if kind not in VALID_KINDS:
            raise ValueError(f"kind must be one of {VALID_KINDS}, got {kind!r}")
        self.kind = kind
        # When set, the proposal is bound to one specific outer round: the engine
        # asking for a different round yields a fresh request instead of silently
        # reusing a stale plan. Without this, an N-round run would look finished
        # while every round reused round 1's plan.
        self.round_idx = round_idx
        self.proposal_path: Path | None = None
        if proposal_path is not None:
            self.proposal_path = Path(proposal_path)
            if not self.proposal_path.is_file():
                raise ProposalError(f"proposal file not found: {self.proposal_path}")

    # -- internal ---------------------------------------------------------
    def _file_data(self) -> dict[str, Any]:
        if self.proposal_path is None:
            raise NeedProposal(
                kind=self.kind,
                reason="no proposal file supplied for this invocation",
            )
        with open(self.proposal_path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        if not isinstance(data, dict):
            raise ProposalError("proposal JSON must be an object")
        return data

    def _yield_back(
        self,
        *,
        kind: str,
        reason: str,
        feedback: str | None = None,
        pack: Any = None,
        attempt_idx: int | None = None,
    ) -> "NeedProposal":
        return NeedProposal(
            kind=kind,
            reason=reason,
            feedback=feedback,
            round_idx=_round_of(pack),
            attempt_idx=attempt_idx,
        )

    # -- BaseProposer -----------------------------------------------------
    def propose_algorithm_plan(
        self,
        pack: ContextPackRecord,
        *,
        feedback: str | None = None,
        attempt_idx: int | None = None,
    ) -> tuple[AlgorithmPlanProposal, dict[str, Any]]:
        if self.kind != KIND_ALGORITHM_PLAN:
            raise self._yield_back(
                kind=KIND_ALGORITHM_PLAN,
                reason="engine needs the outer algorithm plan before any inner trial",
                pack=pack,
            )
        if attempt_idx:
            # attempt_idx is only non-zero when a *previous* plan was rejected.
            # Yield so the agent, not this process, authors the correction.
            raise self._yield_back(
                kind=KIND_ALGORITHM_PLAN,
                reason="previous algorithm plan was rejected by the engine validator",
                feedback=feedback,
                pack=pack,
                attempt_idx=attempt_idx,
            )
        if self.round_idx is not None and _round_of(pack) != self.round_idx:
            raise self._yield_back(
                kind=KIND_ALGORITHM_PLAN,
                reason=(
                    f"engine is asking for outer round {_round_of(pack)}, but this proposal "
                    f"was authored for round {self.round_idx}"
                ),
                pack=pack,
            )
        data = self._file_data()
        if not data.get("algorithm_family"):
            raise self._yield_back(
                kind=KIND_ALGORITHM_PLAN,
                reason="proposal file has no 'algorithm_family' to bind an outer plan",
                pack=pack,
            )
        return super().propose_algorithm_plan(pack, feedback=feedback, attempt_idx=attempt_idx)

    def propose_plan(
        self,
        pack: ContextPackRecord,
        *,
        feedback: str | None = None,
        attempt_idx: int | None = None,
        algorithm_plan: AlgorithmPlanProposal | None = None,
    ) -> tuple[PatchPlanProposal, dict[str, Any]]:
        if self.kind != KIND_INNER:
            raise self._yield_back(
                kind=KIND_INNER,
                reason="outer algorithm plan is frozen; engine needs an inner candidate",
                pack=pack,
            )
        if attempt_idx:
            raise self._yield_back(
                kind=KIND_INNER,
                reason="previous inner patch plan was rejected by the engine validator",
                feedback=feedback,
                pack=pack,
                attempt_idx=attempt_idx,
            )
        return super().propose_plan(
            pack,
            feedback=feedback,
            attempt_idx=attempt_idx,
            algorithm_plan=algorithm_plan,
        )

    def propose(
        self,
        pack: ContextPackRecord,
        *,
        plan: PatchPlanProposal | None = None,
        algorithm_plan: AlgorithmPlanProposal | None = None,
    ) -> tuple[PatchProposal, dict[str, Any]]:
        if self.kind != KIND_INNER:
            raise self._yield_back(
                kind=KIND_INNER,
                reason="engine needs an inner candidate implementation",
                pack=pack,
            )
        try:
            return super().propose(pack, plan=plan, algorithm_plan=algorithm_plan)
        except ProposalError as exc:
            # An unparseable / contract-violating diff is correctable by the agent,
            # so surface it as a request instead of a terminal failure. A plain
            # ProposalError here would let the retry loop exhaust and block the slot.
            raise self._yield_back(
                kind=KIND_INNER,
                reason="supplied patch could not be parsed against the engine schema",
                feedback=str(exc),
                pack=pack,
            ) from exc

    def repair(
        self,
        pack: ContextPackRecord,
        *,
        previous: PatchProposal,
        feedback: str,
        attempt_idx: int,
        plan: PatchPlanProposal | None = None,
        algorithm_plan: AlgorithmPlanProposal | None = None,
    ) -> tuple[PatchProposal, dict[str, Any]]:
        # RESOLVED (G4): this used to be a hard limitation. The yield worked, but
        # the *next* invocation could not re-open proposal authoring:
        # ``_run_inner_full_slot`` saw a slot in state ``repairing`` carrying a
        # ``patchset_id``, took its patchset-resume branch, and called
        # ``_config_for_patchset_id`` on the already-rejected patchset, raising an
        # uncaught
        #   RuntimeError("failed to reconstruct current-best config: <component>:
        #                 Patch has no file header; expected a unified diff.")
        # which surfaced as a terminal ``engine_error``.
        #
        # The engine now marks a deferred slot ``awaiting_proposal`` instead:
        # ``ProposalDeferred`` (the parent of ``NeedProposal``) is recognised by the
        # controller and is absent from every resume-terminal set, so the slot stays
        # retryable and the agent can re-author. Pinned by
        # tests/test_seq_align_callback_harness.py::test_static_rejection_is_re_authorable_not_fatal.
        raise self._yield_back(
            kind=KIND_INNER,
            reason="engine rejected the applied patch and needs a corrected candidate",
            feedback=feedback,
            pack=pack,
            attempt_idx=attempt_idx,
        )
