"""File-backed C-patch proposer for the DeepSeek-Harness loop.

Lets an external agent (the Cordis Harness agent) drive one C-patch round by
writing a proposal JSON file and letting the existing
``c_patch_loop.controller.run_loop`` engine build / evaluate / accept it.

The engine (PatchRunner + build_context_pack + scorer + LoopStore) is reused
verbatim from the validated seq_align pipeline. This proposer ONLY supplies the
patch content that the engine would otherwise get from an OpenAI call, so the
entire optimization loop stays bit-identical to the shell path.

The proposal JSON is the exact ``PatchProposal`` shape:
    {
      "component_name": "p7_ForwardViterbiFilter",
      "rationale": "...",
      "expected_metric_effect": "...",
      "risk_notes": "...",
      "edits": [{"old": "old block", "new": "new block"}],
      // OR a legacy "unified_diff"
    }
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from .models import (
    AlgorithmPlanProposal,
    ContextPackRecord,
    PatchPlanProposal,
    PatchProposal,
)
from .proposer import BaseProposer, ProposalError, parse_patch_proposal, parse_patch_plan_proposal


class FileProposer(BaseProposer):
    """Read a C-patch proposal from a JSON file (written by the Harness agent)."""

    provider_name = "file"

    def __init__(self, proposal_path: str | Path):
        self.proposal_path = Path(proposal_path)
        if not self.proposal_path.is_file():
            raise ProposalError(f"proposal file not found: {self.proposal_path}")

    def _load_raw(self) -> dict[str, Any]:
        with open(self.proposal_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ProposalError("proposal JSON must be an object")
        return data

    def propose_algorithm_plan(
        self,
        pack: ContextPackRecord,
        *,
        feedback: str | None = None,
        attempt_idx: int | None = None,
    ) -> tuple[AlgorithmPlanProposal, dict[str, Any]]:
        # The Harness agent may drive the outer algorithm-plan layer by
        # supplying algorithm-level fields. When ``algorithm_family`` is present
        # the outer binding is strict (controller-owned island), so
        # target_contracts/target_functions are taken from the JSON and must
        # match the family's bound island; when absent, legacy behavior is kept
        # exactly (family defaults, empty contracts) so old proposal files work.
        data = self._load_raw()
        if data.get("algorithm_family"):
            family = data["algorithm_family"]
            target_contracts = [data["component_name"]] if data.get("component_name") else []
            target_functions = [data.get("target_function", "")] if data.get("target_function") else []
        else:
            family = "profile_target_comparison"
            target_contracts = []
            target_functions = []
        plan = AlgorithmPlanProposal(
            algorithm_family=family,
            baseline_algorithm_summary=data.get("baseline_algorithm_summary")
            or "Stock HMMER evaluated by the engine's baseline pass.",
            proposed_algorithm_change=data.get("algorithm_change")
            or (data.get("rationale") or "Agent-supplied algorithmic change."),
            speed_mechanism=data.get("speed_mechanism")
            or "Agent-supplied runtime change; validated by the engine.",
            quality_preservation_argument=data.get("quality_preservation_argument")
            or "Preserve the exact stock path unless the engine's TP/FP gate accepts.",
            target_contracts=target_contracts,
            target_functions=target_functions,
            exact_fallback_strategy=data.get("exact_fallback_strategy")
            or "Retain exact stock fallback for all borderline candidates.",
            profile_evidence_citation=data.get("profile_evidence_citation")
            or "File-driven agent proposal (no internal profile evidence).",
            profile_evidence_refs=list(data.get("profile_evidence_refs") or []),
            risk_notes=data.get("risk_notes") or "",
            related_experience_ids=list(data.get("related_experience_ids") or []),
        )
        return plan, {"provider": self.provider_name, "attempt_kind": "algorithm_plan_from_file"}

    def propose_plan(
        self,
        pack: ContextPackRecord,
        *,
        feedback: str | None = None,
        attempt_idx: int | None = None,
        algorithm_plan: AlgorithmPlanProposal | None = None,
    ) -> tuple[PatchPlanProposal, dict[str, Any]]:
        data = self._load_raw()
        # Derive a minimal plan from the proposal JSON; the engine validates it.
        # Optional plan-text fields are read from the JSON when present so the
        # Harness agent can satisfy the deterministic plan validator directly;
        # missing keys fall back to conservative defaults (backward compatible).
        plan = PatchPlanProposal(
            component_name=data["component_name"],
            target_functions=[data.get("target_function", "")] if data.get("target_function") else [],
            algorithm_hypothesis=data.get("rationale") or "Agent-supplied plan.",
            intended_code_change=data.get("intended_code_change")
            or "Apply the agent's exact edits in the selected component.",
            runtime_effective_argument=data.get("runtime_effective_argument")
            or "The diff alters executable C code in the selected runtime component.",
            profile_evidence_citation=data.get("profile_evidence_citation")
            or "File-driven agent proposal.",
            expected_metric_effect=data.get("expected_metric_effect") or "",
            risk_notes=data.get("risk_notes") or "",
            anti_degenerate_check=data.get("anti_degenerate_check")
            or "Preserve TP/FP by avoiding threshold changes, hit suppression, or recall-reducing shortcuts.",
        )
        return plan, {"provider": self.provider_name, "attempt_kind": "plan_from_file"}

    def propose(
        self,
        pack: ContextPackRecord,
        *,
        plan: PatchPlanProposal | None = None,
        algorithm_plan: AlgorithmPlanProposal | None = None,
    ) -> tuple[PatchProposal, dict[str, Any]]:
        raw = json.dumps(self._load_raw(), ensure_ascii=False)
        proposal = parse_patch_proposal(raw)
        return proposal, {"provider": self.provider_name, "model": "file-agent",
                          "attempt_kind": "patch_from_file"}

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
        raise ProposalError(
            "FileProposer does not self-repair; the Harness agent must rewrite the proposal file "
            "and re-invoke the tool."
        )
