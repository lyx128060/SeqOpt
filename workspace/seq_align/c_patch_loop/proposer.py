"""C-only proposal clients for the deterministic patch loop."""

from __future__ import annotations

import difflib
import json
import os
import re
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from pydantic import ValidationError

from ..components.c_kernels import ALL_C_KERNEL_CONTRACTS
from ..run_hmmer_cli import DEFAULT_HMMER_SRC_ROOT
from .context import ALGORITHM_FAMILIES, algorithm_plan_prompt, context_prompt, plan_prompt, repair_prompt
from .models import AlgorithmPlanProposal, ContextPackRecord, PatchPlanProposal, PatchProposal
from .store import LoopStore, sha256_text, utc_now


class ProposalError(ValueError):
    """Raised when a proposer cannot produce a valid structured patch."""

    def __init__(self, message: str, *, validation_errors: list[dict[str, Any]] | None = None):
        super().__init__(message)
        self.validation_errors = validation_errors


class FatalProposalError(ProposalError):
    """Raised when retrying later rounds cannot fix the proposer failure."""


class ProposalDeferred(Exception):
    """The proposal must be authored out-of-process and supplied later.

    Raised by proposers that do not own an LLM call — for example the callback
    bridge, where a harness agent writes the proposal in a separate process.

    Deliberately NOT a ``ProposalError`` subclass. The controller turns
    ``ProposalError`` into a *terminal* outcome (a blocked outer round or a
    blocked inner slot, both of which are skipped forever on resume). A deferral
    must instead leave the work retryable, so the controller gives it its own
    ``awaiting_proposal`` slot state.
    """

    def __init__(self, message: str, *, kind: str | None = None):
        super().__init__(message)
        self.kind = kind


class BaseProposer:
    """Interface for bounded C-patch proposers."""

    provider_name = "base"

    def propose_algorithm_plan(
        self,
        pack: ContextPackRecord,
        *,
        feedback: str | None = None,
        attempt_idx: int | None = None,
    ) -> tuple[AlgorithmPlanProposal, dict[str, Any]]:
        """Return one high-level outer plan with no source patch payload."""
        inner, metadata = self.propose_plan(pack, feedback=feedback, attempt_idx=attempt_idx)
        proposal = AlgorithmPlanProposal(
            algorithm_family="profile_target_comparison",
            baseline_algorithm_summary="Stock HMMER evaluates its current profile-search decision path.",
            proposed_algorithm_change=inner.algorithm_hypothesis,
            speed_mechanism=inner.runtime_effective_argument,
            quality_preservation_argument=inner.anti_degenerate_check or "Use the exact stock path whenever a decision may affect TP or FP.",
            target_contracts=[],
            target_functions=[],
            exact_fallback_strategy="Retain an exact stock fallback for all borderline candidates.",
            profile_evidence_citation=inner.profile_evidence_citation,
            profile_evidence_refs=list(inner.profile_evidence_refs),
            risk_notes=inner.risk_notes,
            related_experience_ids=list(inner.related_experience_ids),
        )
        return proposal, {**metadata, "attempt_kind": "algorithm_plan_from_inner"}

    def propose_plan(
        self,
        pack: ContextPackRecord,
        *,
        feedback: str | None = None,
        attempt_idx: int | None = None,
        algorithm_plan: AlgorithmPlanProposal | None = None,
    ) -> tuple[PatchPlanProposal, dict[str, Any]]:
        """Return a first-stage algorithm plan before diff generation.

        Legacy/test proposers that only implement ``propose`` are adapted by
        deriving a minimal plan from their patch proposal.
        """
        proposal, metadata = self._legacy_propose_without_plan(pack)
        contract = ALL_C_KERNEL_CONTRACTS[proposal.component_name]
        plan = PatchPlanProposal(
            component_name=proposal.component_name,
            target_functions=list(contract.c_target_functions or []),
            algorithm_hypothesis=proposal.rationale,
            intended_code_change="Generate the proposed bounded C diff for this component.",
            runtime_effective_argument="The diff is expected to alter executable C code in the selected runtime component.",
            profile_evidence_citation=_default_profile_citation(pack),
            expected_metric_effect=proposal.expected_metric_effect,
            risk_notes=proposal.risk_notes,
            anti_degenerate_check="Preserve TP and FP by avoiding threshold changes, hit suppression, or recall-reducing shortcuts.",
        )
        return plan, {**metadata, "attempt_kind": "plan_from_patch"}

    def propose(
        self,
        pack: ContextPackRecord,
        *,
        plan: PatchPlanProposal | None = None,
        algorithm_plan: AlgorithmPlanProposal | None = None,
    ) -> tuple[PatchProposal, dict[str, Any]]:
        raise NotImplementedError

    def _legacy_propose_without_plan(self, pack: ContextPackRecord) -> tuple[PatchProposal, dict[str, Any]]:
        try:
            return self.propose(pack)
        except TypeError as exc:
            if "plan" not in str(exc):
                raise
            raise

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
        """Return a revised proposal after validation feedback."""
        return self.propose(pack, plan=plan, algorithm_plan=algorithm_plan)


class FixtureProposer(BaseProposer):
    """Deterministic no-op-ish proposer used for tests and model-free smoke runs."""

    provider_name = "fixture"

    def __init__(self, hmmer_src_root: str | Path | None = None):
        self.hmmer_src_root = Path(hmmer_src_root or DEFAULT_HMMER_SRC_ROOT).resolve()

    def propose_algorithm_plan(
        self,
        pack: ContextPackRecord,
        *,
        feedback: str | None = None,
        attempt_idx: int | None = None,
    ) -> tuple[AlgorithmPlanProposal, dict[str, Any]]:
        component = "pipeline_flow" if "pipeline_flow" in ALL_C_KERNEL_CONTRACTS else sorted(ALL_C_KERNEL_CONTRACTS)[0]
        plan = AlgorithmPlanProposal(
            algorithm_family="profile_target_comparison",
            baseline_algorithm_summary="Stock HMMER executes the complete per-target pipeline.",
            proposed_algorithm_change="Add a guarded decision surface while retaining the exact stock path.",
            speed_mechanism="Exercise outer/inner lineage without changing normal fixture behavior.",
            quality_preservation_argument="The normal fixture path is the exact stock implementation.",
            target_contracts=[],
            target_functions=[],
            exact_fallback_strategy="Always use the exact stock path in normal execution.",
            profile_evidence_citation=_default_profile_citation(pack),
            implementation_constraints=["Preserve the public function interface."],
            risk_notes="Fixture-only algorithm plan.",
        )
        return plan, {
            "provider": self.provider_name,
            "model": "fixture",
            "raw_response_sha256": sha256_text(plan.model_dump_json()),
            "attempt_kind": "algorithm_plan",
        }

    def propose_plan(
        self,
        pack: ContextPackRecord,
        *,
        feedback: str | None = None,
        attempt_idx: int | None = None,
        algorithm_plan: AlgorithmPlanProposal | None = None,
    ) -> tuple[PatchPlanProposal, dict[str, Any]]:
        component = (
            algorithm_plan.target_contracts[0]
            if algorithm_plan and algorithm_plan.target_contracts
            else ("pipeline_flow" if "pipeline_flow" in ALL_C_KERNEL_CONTRACTS else sorted(ALL_C_KERNEL_CONTRACTS)[0])
        )
        contract = ALL_C_KERNEL_CONTRACTS[component]
        target_functions = list(algorithm_plan.target_functions) if algorithm_plan and algorithm_plan.target_functions else []
        if component == "pipeline_flow" and "p7_evo_align_Run" in (contract.c_target_functions or []):
            target_functions = ["p7_evo_align_Run"]
        elif not target_functions:
            target_functions = [contract.c_target_functions[0]]
        plan = PatchPlanProposal(
            component_name=component,
            target_functions=target_functions,
            algorithm_hypothesis="A guarded fixture branch can exercise the planning/provenance plumbing without changing TP/FP.",
            intended_code_change="Insert a volatile guarded error branch in the comparison island Run function.",
            runtime_effective_argument="The volatile local guard should force executable code generation while preserving normal behavior.",
            profile_evidence_citation=_default_profile_citation(pack),
            expected_metric_effect="No TP/FP change expected; this validates the two-stage workflow rather than optimizing speed.",
            risk_notes="Fixture-only plan, not intended as a real optimization.",
            anti_degenerate_check="TP/recall should remain unchanged; this does not suppress homolog hits.",
        )
        return plan, {
            "provider": self.provider_name,
            "model": "fixture",
            "raw_response_sha256": sha256_text(plan.model_dump_json()),
            "attempt_kind": "plan",
        }

    def propose(
        self,
        pack: ContextPackRecord,
        *,
        plan: PatchPlanProposal | None = None,
        algorithm_plan: AlgorithmPlanProposal | None = None,
    ) -> tuple[PatchProposal, dict[str, Any]]:
        if plan is None:
            plan, _ = self.propose_plan(pack)
        component = plan.component_name
        contract = ALL_C_KERNEL_CONTRACTS[component]
        function = (plan.target_functions or contract.c_target_functions)[0]
        diff = _comment_insertion_diff(
            self.hmmer_src_root,
            contract.c_target_file or "",
            function,
            "  volatile int evomaster_fixture_marker = 0; if (evomaster_fixture_marker) return eslFAIL;\n",
        )
        proposal = PatchProposal(
            component_name=component,
            rationale="Fixture proposal implements the approved plan with an executable guarded branch inside the comparison island.",
            unified_diff=diff,
            expected_metric_effect=plan.expected_metric_effect,
            risk_notes=plan.risk_notes,
        )
        return proposal, {
            "provider": self.provider_name,
            "model": "fixture",
            "raw_response_sha256": sha256_text(proposal.model_dump_json()),
            "attempt_kind": "patch",
        }


class OpenAICompatibleProposer(BaseProposer):
    """OpenAI-compatible JSON proposer with no tool access."""

    provider_name = "openai-compatible"

    def __init__(
        self,
        *,
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        temperature: float = 0.2,
        max_tokens: int = 8192,
        outer_max_tokens: int = 8192,
        timeout: int = 120,
        prefinish_repair_attempts: int | None = None,
        trace_store: LoopStore | None = None,
    ):
        self.model = model or os.getenv("OPENAI_MODEL") or "MiniMax-M3"
        self.base_url = base_url or os.getenv("GPT_BASE_URL")
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.outer_max_tokens = outer_max_tokens
        self.timeout = timeout
        self.reasoning_effort = os.getenv("SEQ_ALIGN_MODEL_REASONING_EFFORT", "medium")
        self.model_retries = max(0, _env_int("SEQ_ALIGN_MODEL_RETRIES", 3))
        self.model_retry_backoff_s = max(0.0, float(os.getenv("SEQ_ALIGN_MODEL_RETRY_BACKOFF_S", "5")))
        self.trace_store = trace_store
        self._trace_context: dict[str, Any] = {}
        self.prefinish_repair_attempts = (
            prefinish_repair_attempts
            if prefinish_repair_attempts is not None
            else _env_int("SEQ_ALIGN_PREFINISH_REPAIR_ATTEMPTS", 2)
        )
        if not self.api_key:
            raise ProposalError("OPENAI_API_KEY is required for the OpenAI-compatible proposer")

    def propose_algorithm_plan(
        self,
        pack: ContextPackRecord,
        *,
        feedback: str | None = None,
        attempt_idx: int | None = None,
    ) -> tuple[AlgorithmPlanProposal, dict[str, Any]]:
        prompt = algorithm_plan_prompt(pack)
        attempt_kind = "algorithm_plan"
        if feedback:
            prompt = _plan_validation_repair_prompt(pack_prompt=prompt, feedback=feedback, attempt_idx=attempt_idx or 1)
            attempt_kind = "algorithm_plan_validation_repair"
        return self._complete_algorithm_plan(prompt, attempt_kind=attempt_kind, attempt_idx=attempt_idx)

    def propose_plan(
        self,
        pack: ContextPackRecord,
        *,
        feedback: str | None = None,
        attempt_idx: int | None = None,
        algorithm_plan: AlgorithmPlanProposal | None = None,
    ) -> tuple[PatchPlanProposal, dict[str, Any]]:
        prompt = plan_prompt(pack, algorithm_plan=algorithm_plan)
        attempt_kind = "plan"
        if feedback:
            prompt = _plan_validation_repair_prompt(pack_prompt=prompt, feedback=feedback, attempt_idx=attempt_idx or 1)
            attempt_kind = "plan_validation_repair"
        return self._complete_plan(prompt, attempt_kind=attempt_kind, attempt_idx=attempt_idx)

    def propose(
        self,
        pack: ContextPackRecord,
        *,
        plan: PatchPlanProposal | None = None,
        algorithm_plan: AlgorithmPlanProposal | None = None,
    ) -> tuple[PatchProposal, dict[str, Any]]:
        return self._complete_patch(context_prompt(pack, plan=plan, algorithm_plan=algorithm_plan), attempt_kind="patch")

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
        prompt = repair_prompt(pack, previous=previous, feedback=feedback, attempt_idx=attempt_idx, plan=plan, algorithm_plan=algorithm_plan)
        return self._complete_patch(prompt, attempt_kind="repair", attempt_idx=attempt_idx)

    def _complete_algorithm_plan(self, prompt: str, *, attempt_kind: str, attempt_idx: int | None = None) -> tuple[AlgorithmPlanProposal, dict[str, Any]]:
        return self._complete_with_prefinish_repair(
            prompt,
            parser=parse_algorithm_plan_proposal,
            output_model=AlgorithmPlanProposal,
            attempt_kind=attempt_kind,
            attempt_idx=attempt_idx,
            system_content="You output only valid JSON for one high-level HMMER algorithm plan and never output code or a diff.",
            artifact_name="algorithm plan",
        )

    def _complete_plan(self, prompt: str, *, attempt_kind: str, attempt_idx: int | None = None) -> tuple[PatchPlanProposal, dict[str, Any]]:
        return self._complete_with_prefinish_repair(
            prompt,
            parser=parse_patch_plan_proposal,
            output_model=PatchPlanProposal,
            attempt_kind=attempt_kind,
            attempt_idx=attempt_idx,
            system_content="You output only valid JSON for one bounded C patch plan.",
            artifact_name="patch plan",
        )

    def _complete_patch(self, prompt: str, *, attempt_kind: str, attempt_idx: int | None = None) -> tuple[PatchProposal, dict[str, Any]]:
        return self._complete_with_prefinish_repair(
            prompt,
            parser=parse_patch_proposal,
            output_model=PatchProposal,
            attempt_kind=attempt_kind,
            attempt_idx=attempt_idx,
            system_content="You output only valid JSON for a single bounded C patch.",
            artifact_name="C patch",
        )

    def _complete_with_prefinish_repair(
        self,
        prompt: str,
        *,
        parser,
        output_model: type[AlgorithmPlanProposal | PatchPlanProposal | PatchProposal],
        attempt_kind: str,
        attempt_idx: int | None = None,
        system_content: str,
        artifact_name: str,
    ):
        current_prompt = prompt
        raw = ""
        metadata: dict[str, Any] = {}
        errors: list[str] = []
        for prefinish_idx in range(self.prefinish_repair_attempts + 1):
            raw, metadata = self._chat_json(
                current_prompt,
                attempt_kind=attempt_kind,
                attempt_idx=attempt_idx,
                system_content=system_content,
            )
            try:
                parsed = parser(raw)
            except (ProposalError, ValidationError) as exc:
                error = _schema_validation_feedback(exc, artifact_name=artifact_name)
                self._persist_model_trace(
                    prompt=current_prompt,
                    raw_response=raw,
                    metadata=metadata,
                    validation_status="rejected",
                    validation_error=error,
                    attempt_kind=attempt_kind,
                    attempt_idx=attempt_idx,
                    prefinish_idx=prefinish_idx,
                    system_content=system_content,
                    artifact_name=artifact_name,
                )
                errors.append(error)
                if prefinish_idx >= self.prefinish_repair_attempts:
                    raise ProposalError(
                        f"{artifact_name} prefinish validation failed after {prefinish_idx} repairs: {error}"
                    ) from exc
                current_prompt = _prefinish_repair_prompt(
                    original_prompt=prompt,
                    raw_response=raw,
                    error=error,
                    expected_schema=output_model.model_json_schema(),
                    artifact_name=artifact_name,
                    repair_idx=prefinish_idx + 1,
                )
                continue
            self._persist_model_trace(
                prompt=current_prompt,
                raw_response=raw,
                metadata=metadata,
                validation_status="ok",
                validation_error=None,
                attempt_kind=attempt_kind,
                attempt_idx=attempt_idx,
                prefinish_idx=prefinish_idx,
                system_content=system_content,
                artifact_name=artifact_name,
            )
            metadata = dict(metadata)
            metadata["prefinish_repair_attempts_used"] = prefinish_idx
            if errors:
                metadata["prefinish_errors"] = errors
            return parsed, metadata
        raise ProposalError(f"{artifact_name} prefinish validation failed")

    def _persist_model_trace(
        self,
        *,
        prompt: str,
        raw_response: str,
        metadata: dict[str, Any],
        validation_status: str,
        validation_error: str | None,
        attempt_kind: str,
        attempt_idx: int | None,
        prefinish_idx: int,
        system_content: str,
        artifact_name: str,
    ) -> None:
        if self.trace_store is None:
            return
        context = dict(self._trace_context)
        trace = {
            "schema_version": "1.0",
            "stage": context.pop("stage", None),
            "memory_references": context.pop("memory_references", []),
            "artifact_references": context.pop("artifact_references", []),
            "trace_context": context,
            "provider": self.provider_name,
            "model": self.model,
            "artifact_name": artifact_name,
            "attempt_kind": attempt_kind,
            "attempt_idx": attempt_idx,
            "prefinish_idx": prefinish_idx,
            "system_content": system_content,
            "prompt": prompt,
            "prompt_sha256": sha256_text(prompt),
            "raw_response": raw_response,
            "raw_response_sha256": sha256_text(raw_response),
            "metadata": metadata,
            "validation": {
                "status": validation_status,
                "error": validation_error,
            },
            "created_at": utc_now(),
        }
        self.trace_store.put_model_trace(trace)

    def _chat_json(self, prompt: str, *, attempt_kind: str, attempt_idx: int | None = None, system_content: str) -> tuple[str, dict[str, Any]]:
        try:
            from openai import APIConnectionError, APIStatusError, APITimeoutError, AuthenticationError, OpenAI
        except ImportError as exc:
            raise ProposalError("openai package is required for model proposals") from exc

        client_kwargs: dict[str, Any] = {"api_key": self.api_key}
        if self.base_url:
            client_kwargs["base_url"] = self.base_url
        client = OpenAI(**client_kwargs)
        last_error = None
        for retry_idx in range(self.model_retries + 1):
            try:
                # The local gateway is validated through /v1/responses. Using
                # this endpoint also avoids Chat Completions compatibility
                # stalls for large structured prompts.
                response = client.responses.create(
                    model=self.model,
                    instructions=system_content,
                    input=prompt,
                    max_output_tokens=self.max_tokens,
                    reasoning={"effort": self.reasoning_effort},
                    timeout=self.timeout,
                )
                break
            except AuthenticationError as exc:
                raise FatalProposalError(f"model authentication failed: {_openai_error_summary(exc)}") from exc
            except (APITimeoutError, APIConnectionError) as exc:
                last_error = exc
                if retry_idx >= self.model_retries:
                    label = "timed out" if isinstance(exc, APITimeoutError) else "connection failed"
                    raise ProposalError(f"model request {label} after {retry_idx + 1} attempts: {_openai_error_summary(exc)}") from exc
                delay = self.model_retry_backoff_s * (2 ** retry_idx)
                if delay:
                    time.sleep(delay)
            except APIStatusError as exc:
                if _is_fatal_model_status(exc):
                    raise FatalProposalError(f"model account or access failure: {_openai_error_summary(exc)}") from exc
                # Retry transient gateway/upstream failures (5xx). A preflight
                # can succeed while a large generation intermittently fails.
                status_code = getattr(exc, "status_code", None)
                if isinstance(status_code, int) and status_code >= 500:
                    last_error = exc
                    if retry_idx >= self.model_retries:
                        raise ProposalError(f"model request failed after {retry_idx + 1} attempts: {_openai_error_summary(exc)}") from exc
                    delay = self.model_retry_backoff_s * (2 ** retry_idx)
                    if delay:
                        time.sleep(delay)
                    continue
                raise ProposalError(f"model request failed: {_openai_error_summary(exc)}") from exc
        else:
            raise ProposalError(f"model request failed after retries: {last_error}")
        raw = getattr(response, "output_text", "") or ""
        if not raw:
            raise ProposalError("model response completed without output text")
        return raw, {
            "provider": self.provider_name,
            "model": self.model,
            "base_url_host": _safe_url_host(self.base_url),
            "response_id": getattr(response, "id", None),
            "finish_reason": getattr(response, "status", None),
            "usage": response.usage.model_dump() if getattr(response, "usage", None) else {},
            "raw_response_sha256": sha256_text(raw),
            "attempt_kind": attempt_kind,
            "repair_attempt_idx": attempt_idx,
        }


def _default_profile_citation(pack: ContextPackRecord) -> str:
    evidence = pack.profile_evidence or {}
    if not evidence.get("has_profile"):
        return "No measured profile evidence is available in this context pack."
    symbol_ref = _default_symbol_profile_ref(evidence)
    for key in ("current_best", "baseline"):
        record = evidence.get(key) or {}
        profile = record.get("profile") or {}
        dominant = profile.get("dominant_command")
        if dominant:
            fractions = profile.get("command_wall_fraction") or {}
            fraction = fractions.get(dominant)
            eval_id = record.get("eval_id")
            parts = [f"Measured profile evidence reports dominant_command={dominant}"]
            if fraction is not None:
                parts.append(f"wall_fraction={fraction}")
            if eval_id:
                parts.append(f"eval_id={eval_id}")
            top = profile.get("top_slowest_queries") or []
            if top:
                query = top[0].get("query_id") or top[0].get("query_idx")
                if query:
                    parts.append(f"top_slowest_query={query}")
            if symbol_ref:
                parts.append(symbol_ref)
            return "; ".join(parts) + "."
    for item in evidence.get("recent_profiles") or []:
        profile = item.get("profile") or {}
        dominant = profile.get("dominant_command")
        if dominant:
            suffix = f"; {symbol_ref}" if symbol_ref else ""
            return f"Measured recent profile evidence reports dominant_command={dominant}; eval_id={item.get('eval_id')}{suffix}."
    if symbol_ref:
        return f"Measured profile evidence is present; {symbol_ref}."
    return "Measured profile evidence is present, but no dominant command was summarized."



def _default_symbol_profile_ref(evidence: dict[str, Any]) -> str:
    for row in evidence.get("fine_grained_hotspots") or []:
        if not isinstance(row, dict):
            continue
        symbol = row.get("symbol")
        function = row.get("function")
        component = row.get("component_name")
        if symbol or function or component:
            parts = ["top_symbol_hotspot"]
            if symbol:
                parts.append(f"symbol={symbol}")
            if function:
                parts.append(f"function={function}")
            if component:
                parts.append(f"component={component}")
            return " ".join(parts)
    for row in evidence.get("contract_hotspots") or []:
        if isinstance(row, dict) and row.get("component_name"):
            return f"top_contract_hotspot component={row.get('component_name')} functions={row.get('functions') or []}"
    return ""



def parse_algorithm_plan_proposal(raw: str) -> AlgorithmPlanProposal:
    """Parse a strict high-level algorithm plan and reject patch payloads."""
    data = _parse_json_object(raw)
    forbidden = {
        "unified_diff", "edits", "python_payload", "patch", "source_code",
        "c_patch", "implementation", "intended_code_change",
    }
    present = sorted(forbidden.intersection(data))
    if present:
        raise ProposalError(f"algorithm plan must not contain implementation/diff fields: {present}")
    try:
        plan = AlgorithmPlanProposal(**data)
    except ValidationError as exc:
        raise ProposalError(
            "algorithm plan schema validation failed",
            validation_errors=exc.errors(include_url=False, include_context=True, include_input=True),
        ) from exc
    if plan.algorithm_family not in ALGORITHM_FAMILIES:
        raise ProposalError(f"unknown or implementation-only algorithm family: {plan.algorithm_family}")
    return plan


def parse_patch_plan_proposal(raw: str) -> PatchPlanProposal:
    """Parse a raw model response into a strict first-stage patch plan."""
    data = _parse_json_object(raw)
    if "unified_diff" in data or "python_payload" in data:
        raise ProposalError("patch plan must not contain unified_diff or python_payload")
    try:
        plan = PatchPlanProposal(**data)
    except ValidationError as exc:
        raise ProposalError(
            "patch plan schema validation failed",
            validation_errors=exc.errors(include_url=False, include_context=True, include_input=True),
        ) from exc
    if plan.component_name not in ALL_C_KERNEL_CONTRACTS:
        raise ProposalError(f"unknown C component in patch plan: {plan.component_name}")
    return plan


def parse_patch_proposal(raw: str) -> PatchProposal:
    """Parse a raw model response into a strict C-patch proposal."""
    data = _parse_json_object(raw)
    if "python_payload" in data:
        raise ProposalError("proposal contains forbidden python_payload")
    try:
        # The same proposal JSON may also carry patch-plan-only fields
        # (target_function, profile_evidence_citation, intended_code_change,
        # runtime_effective_argument, anti_degenerate_check) consumed by
        # FileProposer.propose_plan; ignore them here so the strict
        # PatchProposal schema still parses the diff payload.
        known = set(PatchProposal.model_fields)
        proposal = PatchProposal(**{key: value for key, value in data.items() if key in known})
    except ValidationError as exc:
        raise ProposalError(
            "proposal schema validation failed",
            validation_errors=exc.errors(include_url=False, include_context=True, include_input=True),
        ) from exc
    if proposal.component_name not in ALL_C_KERNEL_CONTRACTS:
        raise ProposalError(f"unknown C component: {proposal.component_name}")
    if not proposal.edits and not proposal.unified_diff.strip():
        raise ProposalError("proposal must include exact edits or a legacy unified_diff")
    for idx, edit in enumerate(proposal.edits, start=1):
        if not edit.old.strip():
            raise ProposalError(f"exact edit {idx} has an empty old block")
        if not edit.new.strip():
            raise ProposalError(f"exact edit {idx} has an empty new block")
        if edit.old == edit.new:
            raise ProposalError(f"exact edit {idx} old and new blocks are identical")
    if proposal.unified_diff.strip():
        diff_lines = proposal.unified_diff.splitlines()
        has_from = any(line.startswith("--- ") for line in diff_lines)
        has_to = any(line.startswith("+++ ") for line in diff_lines)
        if not (has_from and has_to):
            raise ProposalError("unified_diff must contain --- and +++ file headers")
    return proposal


def _parse_json_object(raw: str) -> dict[str, Any]:
    text = _strip_fences(raw.strip())
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise ProposalError("model response is not JSON") from exc
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError as inner_exc:
            raise ProposalError(f"model response is not valid JSON: {inner_exc.msg}") from inner_exc
    if not isinstance(data, dict):
        raise ProposalError("model response must be a JSON object")
    return data


def _prefinish_repair_prompt(
    *,
    original_prompt: str,
    raw_response: str,
    error: str,
    expected_schema: dict[str, Any],
    artifact_name: str,
    repair_idx: int,
) -> str:
    return (
        f"Your previous {artifact_name} response failed deterministic prefinish parsing/schema validation.\n"
        "This is an immediate repair attempt before the optimization round can proceed.\n"
        "Return only one corrected JSON object. Do not include markdown fences or explanatory text.\n"
        "Do not ask for tools. Do not omit required fields. Do not include forbidden fields.\n\n"
        f"PREFINISH_REPAIR_ATTEMPT: {repair_idx}\n"
        "DETERMINISTIC_VALIDATION_ERRORS:\n"
        f"{error}\n\n"
        "REQUIRED_OUTPUT_JSON_SCHEMA:\n"
        + json.dumps(expected_schema, indent=2, sort_keys=True)
        + "\n\n"
        "PREVIOUS_RAW_RESPONSE:\n"
        + raw_response[:16000]
        + "\n\nORIGINAL_REQUEST:\n"
        + original_prompt
    )


def _schema_validation_feedback(exc: ProposalError | ValidationError, *, artifact_name: str) -> str:
    """Return stable, field-addressable feedback for an immediate LLM repair."""
    if isinstance(exc, ValidationError):
        details: Any = exc.errors(include_url=False, include_context=True, include_input=True)
    elif exc.validation_errors is not None:
        details = exc.validation_errors
    else:
        details = [{"type": "proposal_error", "loc": [], "msg": str(exc)}]
    return json.dumps(
        {
            "artifact": artifact_name,
            "error_count": len(details),
            "errors": details,
        },
        indent=2,
        sort_keys=True,
        default=str,
    )



def _plan_validation_repair_prompt(*, pack_prompt: str, feedback: str, attempt_idx: int) -> str:
    return (
        "The previous patch plan passed JSON/schema parsing but failed deterministic plan validation.\n"
        "This is an immediate plan repair attempt before any C diff is requested.\n"
        "Return only one corrected patch-plan JSON object. Do not include a unified_diff, python_payload, "
        "markdown fences, or explanatory text.\n"
        "Keep the plan within the allowed C component/function contract and make the intended change executable.\n\n"
        f"PLAN_VALIDATION_REPAIR_ATTEMPT: {attempt_idx}\n"
        "VALIDATION_FEEDBACK:\n"
        f"{feedback}\n\n"
        "ORIGINAL_PLAN_REQUEST:\n"
        f"{pack_prompt}"
    )



def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default



def _openai_error_summary(exc: Exception) -> str:
    status = getattr(exc, "status_code", None)
    code = getattr(exc, "code", None)
    error_type = getattr(exc, "type", None)
    parts = []
    if status is not None:
        parts.append(f"status={status}")
    if code:
        parts.append(f"code={code}")
    if error_type:
        parts.append(f"type={error_type}")
    if not parts:
        return exc.__class__.__name__
    return ", ".join(parts)


def _is_fatal_model_status(exc: Exception) -> bool:
    """Classify account/access failures that cannot improve in a later round."""
    status = getattr(exc, "status_code", None)
    code = str(getattr(exc, "code", "") or "").strip().lower()
    fatal_codes = {
        "billing_hard_limit_reached",
        "insufficient_balance",
        "insufficient_quota",
        "quota_exceeded",
    }
    return status in {402, 403} or code in fatal_codes



def _strip_fences(text: str) -> str:
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        return "\n".join(lines).strip()
    return text


def _safe_url_host(url: str | None) -> str | None:
    if not url:
        return None
    parsed = urlparse(url)
    return parsed.netloc or parsed.path or None


def _comment_insertion_diff(source_root: Path, target_file: str, function: str, comment_line: str) -> str:
    source_file = _find_source_file(source_root, target_file)
    original = source_file.read_text(encoding="utf-8", errors="replace").splitlines(keepends=True)
    insert_at = _first_body_line(original, function)
    modified = list(original)
    if comment_line not in modified[max(0, insert_at - 3): insert_at + 4]:
        modified.insert(insert_at, comment_line)
    rel = Path("src") / target_file
    return "".join(difflib.unified_diff(
        original,
        modified,
        fromfile=f"a/{rel}",
        tofile=f"b/{rel}",
    ))


def _find_source_file(source_root: Path, target_file: str) -> Path:
    for candidate in (
        source_root / target_file,
        source_root / "src" / target_file,
        source_root / "src" / Path(target_file).name,
        source_root / Path(target_file).name,
    ):
        if candidate.exists():
            return candidate
    raise ProposalError(f"fixture source file not found: {target_file} under {source_root}")


def _first_body_line(lines: list[str], function: str) -> int:
    pattern = re.compile(rf"\b{re.escape(function)}\s*\(")
    for idx, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith(("/*", "*", "//")):
            continue
        if not pattern.search(line):
            continue
        for j in range(idx, min(idx + 10, len(lines))):
            if "{" in lines[j]:
                return j + 1
    raise ProposalError(f"fixture function not found: {function}")
