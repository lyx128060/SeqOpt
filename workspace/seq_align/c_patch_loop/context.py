"""Build bounded model context packs for C-patch proposals."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

from ..components.c_kernels import ACTIVE_C_KERNEL_CONTRACTS, ALL_C_KERNEL_CONTRACTS
from ..run_hmmer_cli import DEFAULT_HMMER_SRC_ROOT
from .models import (
    AlgorithmPlanProposal,
    ContextPackRecord,
    ContractSnapshot,
    DurableWisdomRecord,
    EvalSummary,
    InnerImplementationMemoryView,
    OuterAlgorithmMemoryView,
    PatchPlanProposal,
    PatchProposal,
    PatchRepairMemoryView,
    PhaseSummaryRecord,
    SourceExcerpt,
)
from .store import LoopStore, hash_file, record_hash, utc_now


PLAN_OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": [
        "component_name",
        "target_functions",
        "algorithm_hypothesis",
        "intended_code_change",
        "runtime_effective_argument",
        "profile_evidence_citation",
    ],
    "properties": {
        # New proposals may target only contracts compiled into the current
        # runtime. ALL_C_KERNEL_CONTRACTS remains the replay/parser registry for
        # immutable historical proposals.
        "component_name": {"type": "string", "enum": sorted(ACTIVE_C_KERNEL_CONTRACTS)},
        "target_functions": {"type": "array", "items": {"type": "string"}},
        "algorithm_hypothesis": {"type": "string"},
        "intended_code_change": {"type": "string"},
        "runtime_effective_argument": {"type": "string"},
        "profile_evidence_citation": {
            "type": "string",
            "description": "Measured profile evidence cited by this plan; cite the dominant command/query hotspot when profile_evidence.has_profile is true.",
        },
        "profile_evidence_refs": {"type": "array", "items": {"type": "string"}},
        "target_symbols": {"type": "array", "items": {"type": "string"}},
        "negative_experience_citation": {"type": "string"},
        "novelty_argument": {"type": "string"},
        "expected_metric_effect": {"type": "string"},
        "risk_notes": {"type": "string"},
        "anti_degenerate_check": {"type": "string"},
        "related_experience_ids": {"type": "array", "items": {"type": "string"}},
    },
    "additionalProperties": False,
}

# The first two names are the original controller-owned algorithm islands. The
# next four names expose real, independently bound runtime surfaces. Remaining
# names are compatibility aliases retained so immutable historical trajectories
# can still be parsed and replayed by the ALL_C_KERNEL_CONTRACTS registry.
ALGORITHM_FAMILIES = [
    "profile_target_comparison",
    "iterative_search",
    "dp_recurrence_strategy",
    "ssv_filter_strategy",
    "null2_scoring_strategy",
    "sequence_streaming_strategy",
    "pipeline_flow_strategy",
    "jackhmmer_search_loop_strategy",
    "hit_ranking_alignment_strategy",
    "profile_construction_strategy",
    "domain_reporting_strategy",
]

ALGORITHM_PLAN_OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": [
        "algorithm_family",
        "baseline_algorithm_summary",
        "proposed_algorithm_change",
        "speed_mechanism",
        "quality_preservation_argument",
        "profile_evidence_citation",
    ],
    "properties": {
        "algorithm_family": {"type": "string", "enum": ALGORITHM_FAMILIES},
        "baseline_algorithm_summary": {"type": "string"},
        "proposed_algorithm_change": {"type": "string"},
        "speed_mechanism": {"type": "string"},
        "quality_preservation_argument": {"type": "string"},
        "exact_fallback_strategy": {"type": "string"},
        "predicted_observables": {"type": "array", "items": {"type": "string"}},
        "profile_evidence_citation": {"type": "string"},
        "profile_evidence_refs": {"type": "array", "items": {"type": "string"}},
        "novelty_against_l2_l3": {"type": "string"},
        "implementation_constraints": {"type": "array", "items": {"type": "string"}},
        "risk_notes": {"type": "string"},
        "related_experience_ids": {"type": "array", "items": {"type": "string"}},
    },
    "additionalProperties": False,
}

# A bounded prompt keeps the proposal request practical for OpenAI-compatible
# gateways while retaining source excerpts, constraints, and recent evidence.
MAX_CONTEXT_TOKENS = int(os.getenv("SEQ_ALIGN_MODEL_CONTEXT_TOKENS", "24000"))
OUTER_RESERVED_OUTPUT_TOKENS = 8_192
INNER_RESERVED_OUTPUT_TOKENS = 32_768
_CHARS_PER_TOKEN = 4

APPLICATION_BASE_POLICY = (
    "Every inner trial records its approved algorithm_plan_id and explicit parent_version_id. "
    "The controller materializes that immutable parent before applying the incremental diff, "
    "then stores a reproducible stock-relative patchset. Source excerpts are from that explicit "
    "parent version, never from a mutable checkout."
)

OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["component_name", "rationale", "edits"],
    "properties": {
        "component_name": {"type": "string", "enum": sorted(ACTIVE_C_KERNEL_CONTRACTS)},
        "rationale": {"type": "string"},
        "edits": {
            "type": "array",
            "minItems": 1,
            "maxItems": 3,
            "items": {
                "type": "object",
                "required": ["old", "new"],
                "properties": {
                    "old": {"type": "string"},
                    "new": {"type": "string"},
                },
                "additionalProperties": False,
            },
        },
        "unified_diff": {"type": "string", "description": "Optional legacy fallback; prefer edits."},
        "expected_metric_effect": {"type": "string"},
        "risk_notes": {"type": "string"},
    },
    "additionalProperties": False,
}

HARD_CONSTRAINTS = [
    "Return exactly one JSON object matching required_output_schema.",
    "Propose exactly one C patch for exactly one allowed component.",
    "Do not propose python_payload or any Python file edits.",
    "Do not ask to run shell commands, inspect files, or execute benchmarks.",
    "The unified diff must touch one existing target file only.",
    "Every changed hunk must stay inside the selected contract's allowed C functions.",
    "Primary objective is lower runtime/cost_s, not higher precision ratio.",
    "TP must not get worse against the quality reference; FP must not get worse against the quality reference.",
    "Do not optimize speed by changing acceptance/inclusion thresholds, iteration counts, E-value cutoffs, or suppressing reported homolog hits; filter implementations may remove redundant work only when score/decision semantics or an exact fallback are preserved.",
    "For sequence_io parser patches, preserve residue classification, ignored/EOL/EOD handling, ascii->bpos state, and FASTA name/description semantics.",
    "Sequence parsing, streaming, and data movement are valid optimization targets only when every target and its residue/name semantics remain observable to the unchanged search pipeline.",
    "For ssv_filter_neon and legacy ssv_filter patches, preserve MAX_BANDS preprocessor guards around high-band calc_band_15..18 references.",
    "Prefer result-preserving hot-path C optimizations: remove redundant work, simplify branches/loops, cache reused computations, or avoid unnecessary arithmetic while keeping same scoring semantics.",
    "Make a concrete C algorithm update as an executable C algorithm change; comments, no-ops, unreachable code, and optimized-away markers are rejected.",
]

ALGORITHM_NOTES = {
    "pipeline_flow": (
        "The profile/target comparison island owns the per-target decision cascade behind a "
        "frozen p7_Pipeline driver and stock fallback. It may remove redundant exact work while "
        "preserving reporting, inclusion, score, and per-worker state semantics."
    ),
    "jackhmmer_loop": (
        "The query-local iteration island owns profile construction and hits-to-MSA/convergence "
        "decisions behind a frozen database/scheduling driver. Cross-round reuse must preserve "
        "the same included targets, final model, and convergence behavior."
    ),
    "dp_kernel_neon": (
        "Optimized AArch64/NEON Forward/Backward dynamic-programming routines compute profile-HMM "
        "sequence scores and posterior inputs in the current CLI runtime. Edits here affect core "
        "sensitivity and speed, so recurrence order, scaling, overflow, and downstream score "
        "semantics must remain equivalent."
    ),
    "ssv_filter_neon": (
        "Optimized AArch64/NEON SSV filter band-scoring routines are an early jackhmmer filter "
        "path. Full UniRef90 profiles measured calc_band_* as dominant hotspots; edits must "
        "preserve score semantics, overflow/J-state fallback behavior, MAX_BANDS guards, and "
        "TP/FP-sensitive decisions."
    ),
    "sequence_io": (
        "Easel FASTA parser routines stream the target database into jackhmmer. Full UniRef90 "
        "gprof runs measured addbuf, seebuf, and header_fasta as dominant hotspots; edits must "
        "preserve residue conversion, ignored/EOL/EOD consumption, line bookkeeping, database "
        "order, and FASTA name/description parsing. Parser/data-movement work is a valid outer "
        "surface when it does not skip, reorder, or alter targets."
    ),
    "null2_correction_neon": (
        "The current AArch64/NEON Null2 composition-bias correction adjusts scores for biased "
        "sequence composition. Optimizations must preserve the correction values consumed by "
        "domain scoring; threshold or correction-strength tuning is outside the speed search."
    ),
}

ALGORITHM_FAMILY_GUIDANCE = {
    "profile_target_comparison": (
        "Optimize one profile/target comparison cascade with unchanged reporting and inclusion "
        "semantics; ambiguous or unproven cases must use the frozen exact stock fallback."
    ),
    "iterative_search": (
        "Optimize cross-round model construction, reuse, or convergence work without reducing "
        "the configured rounds or changing the included-hit MSA/final model."
    ),
    "dp_recurrence_strategy": (
        "Optimize the active AArch64/NEON Forward/Backward recurrence while preserving numerical "
        "scaling, overflow behavior, and scores consumed downstream."
    ),
    "ssv_filter_strategy": (
        "Optimize the active AArch64/NEON SSV band/filter implementation without changing filter "
        "scores, cutoffs, pass/fail decisions, overflow fallback, or MAX_BANDS behavior."
    ),
    "null2_scoring_strategy": (
        "Optimize the active AArch64/NEON Null2 arithmetic without tuning correction strength or "
        "changing the values used for domain and hit scoring."
    ),
    "sequence_streaming_strategy": (
        "Optimize FASTA parsing, buffering, conversion, and data movement while presenting every "
        "target in database order with identical residues, names, descriptions, and parser errors."
    ),
}

# Compatibility aliases are intentionally described in terms of their current
# canonical semantic surface. They stay in the schema for old plan replay but
# are not additional edit boundaries.
ALGORITHM_FAMILY_GUIDANCE.update({
    "pipeline_flow_strategy": ALGORITHM_FAMILY_GUIDANCE["profile_target_comparison"],
    "domain_reporting_strategy": ALGORITHM_FAMILY_GUIDANCE["profile_target_comparison"],
    "jackhmmer_search_loop_strategy": ALGORITHM_FAMILY_GUIDANCE["iterative_search"],
    "hit_ranking_alignment_strategy": ALGORITHM_FAMILY_GUIDANCE["iterative_search"],
    "profile_construction_strategy": ALGORITHM_FAMILY_GUIDANCE["iterative_search"],
})


def contract_snapshots() -> list[ContractSnapshot]:
    """Snapshot only contracts available to new proposals in this runtime."""
    snapshots: list[ContractSnapshot] = []
    for name, contract in ACTIVE_C_KERNEL_CONTRACTS.items():
        snap = ContractSnapshot(
            name=name,
            kind=str(contract.kind),
            c_target_file=contract.c_target_file or "",
            c_target_functions=list(contract.c_target_functions or []),
            description=contract.description,
        )
        snap.contract_hash = record_hash(snap, id_field="contract_hash")
        snapshots.append(snap)
    return snapshots


def build_context_pack(
    store: LoopStore,
    *,
    round_idx: int,
    hmmer_src_root: str | Path | None = None,
    parent_patchset_id: str | None = None,
    current_best_patchset_id: str | None = None,
    baseline: EvalSummary | None = None,
    current_best: EvalSummary | None = None,
    max_recent: int = 3,
    accept_min_tp_fraction: float = 1.0,
    min_speedup_fraction: float = 0.03,
    eval_repeats: int = 1,
    wisdom_dir: str | Path | None = None,
    context_window_tokens: int = MAX_CONTEXT_TOKENS,
) -> ContextPackRecord:
    """Create and store the bounded context for one model proposal."""
    source_root = Path(hmmer_src_root or DEFAULT_HMMER_SRC_ROOT).resolve()
    recent_history = _recent_history(store, max_recent=max_recent)
    profile_evidence = _profile_evidence(store, baseline=baseline, current_best=current_best)
    recent_negative = _recent_negative_experience(store)
    durable_wisdom: list[DurableWisdomRecord] = []
    if wisdom_dir is not None:
        try:
            from .wisdom import retrieve_wisdom_for_context

            durable_wisdom = retrieve_wisdom_for_context(
                profile_evidence,
                recent_history,
                wisdom_dir=wisdom_dir,
            )
        except (FileNotFoundError, ValueError):
            durable_wisdom = []
    pack = ContextPackRecord(
        round_idx=round_idx,
        run_dir=str(store.run_dir),
        objective={
            "primary_metric": "cost_s",
            "quality_constraints": {"tp_min_fraction_of_reference": accept_min_tp_fraction, "fp_max_delta": 0},
            "direction": "minimize runtime while preserving TP and FP against the stock/reference evaluation",
            "min_speedup_fraction": min_speedup_fraction,
            "eval_repeats": eval_repeats,
        },
        parent_patchset_id=parent_patchset_id,
        current_best_patchset_id=current_best_patchset_id,
        baseline=baseline,
        current_best=current_best,
        allowed_c_contracts=contract_snapshots(),
        source_excerpts=_source_excerpts(source_root),
        original_algorithm_context=ALGORITHM_NOTES,
        profile_evidence=profile_evidence,
        recent_history=recent_history,
        recent_negative_experience=recent_negative,
        compressed_experience=store.latest_experience_text(),
        l1_raw_working_memory={
            "trajectory": recent_history,
            "recent_negative_experience": recent_negative,
        },
        l2_phase_summaries=_phase_summaries(store),
        l3_durable_wisdom=durable_wisdom,
        application_base_policy=APPLICATION_BASE_POLICY,
        context_window_tokens=context_window_tokens,
        required_output_schema=OUTPUT_SCHEMA,
        hard_constraints=HARD_CONSTRAINTS,
        created_at=utc_now(),
    )
    pack.context_pack_id = record_hash(pack, id_field="context_pack_id")
    store.put_context_pack(pack)
    return pack


def json_dumps(data: Any) -> str:
    return json.dumps(data, indent=2, ensure_ascii=False, sort_keys=True)


def _legacy_plan_prompt(pack: ContextPackRecord) -> str:
    """Render the first-stage algorithm planning prompt."""
    return (
        "You are planning one bounded C-language speed optimization patch for HMMER/Jackhmmer.\n"
        "The objective is to reduce runtime/cost_s while preserving TP and FP as hard constraints.\n"
        "You cannot inspect files or run tools. Use only the context pack below.\n"
        "Do not output a unified diff yet. Return exactly one JSON object and no markdown fences.\n\n"
        f"CONTEXT_PACK_ID: {pack.context_pack_id}\n"
        f"ROUND: {pack.round_idx}\n"
        f"RUN_DIR: {pack.run_dir}\n\n"
        "HARD CONSTRAINTS:\n"
        + "\n".join(f"- {c}" for c in pack.hard_constraints)
        + "\n- The plan must name one allowed component and allowed target function(s)."
        + "\n- The plan must describe a concrete executable-affecting C logic change, not a comment/no-op/optimized-away marker."
        + "\n- The plan must explain why rebuilt jackhmmer/hmmsearch executable hashes should change."
        + "\n- The plan must explain why TP/FP should remain unchanged; do not rely on hit suppression or threshold changes."
        + "\n- The expected metric effect should be a runtime speedup with unchanged TP and unchanged FP."
        + "\n\nPLAN_OUTPUT_SCHEMA:\n"
        + json_dumps(PLAN_OUTPUT_SCHEMA)
        + "\n\nMEASURED PROFILE EVIDENCE:\n"
        + json_dumps(pack.profile_evidence)
        + "\n\nMEASURED FINE-GRAINED PROFILE EVIDENCE:\n"
        + json_dumps(_fine_grained_prompt_evidence(pack.profile_evidence))
        + "\n\nRECENT NEGATIVE EXPERIENCE / DO NOT REPEAT:\n"
        + json_dumps(pack.recent_negative_experience)
        + "\nTreat measured profile evidence as authoritative over source-name guesses.\n"
        + "If profile_evidence.has_profile is true, profile_evidence_citation is required and must cite the dominant measured command/query hotspot it targets.\n"
        + "If profile_evidence.has_symbol_profile is true, cite a top symbol/function/contract hotspot; a generic jackhmmer citation is insufficient.\n"
        + "For a jackhmmer-dominant full profile, cite jackhmmer plus a wall fraction, eval id, or top slow query; do not propose hmmsearch-only work.\n"
        + "If your target overlaps RECENT NEGATIVE EXPERIENCE, explain why this is not the same failed patch pattern and what new evidence makes it worth trying.\n"
        + "If no profile evidence is present, profile_evidence_citation must say so and avoid inventing hotspots.\n"
        + "Profile-derived recommendations are priorities; allowed_c_contracts remains the authoritative edit boundary.\n"
        + "\nCONTEXT_PACK_JSON:\n"
        + pack.model_dump_json(indent=2)
    )


def _legacy_context_prompt(pack: ContextPackRecord, *, plan: PatchPlanProposal | dict[str, Any] | None = None) -> str:
    """Render a context pack and approved plan into a patch-diff prompt."""
    plan_json = "null"
    if plan is not None:
        plan_json = plan.model_dump_json(indent=2) if hasattr(plan, "model_dump_json") else json_dumps(plan)
    return (
        "You are implementing one bounded C-language speed optimization patch for HMMER/Jackhmmer.\n"
        "Primary objective: reduce runtime/cost_s. Hard constraints: TP must not decrease and FP must not increase.\n"
        "You cannot inspect files or run tools. Use only the context pack and approved patch plan below.\n"
        "Return exactly one JSON object and no markdown fences.\n\n"
        f"CONTEXT_PACK_ID: {pack.context_pack_id}\n"
        f"ROUND: {pack.round_idx}\n"
        f"RUN_DIR: {pack.run_dir}\n\n"
        "APPROVED_PATCH_PLAN_JSON:\n"
        + plan_json
        + "\n\nHARD CONSTRAINTS:\n"
        + "\n".join(f"- {c}" for c in pack.hard_constraints)
        + "\n\nACCEPTANCE REQUIREMENTS:\n"
        + "- Full evaluation accepts a candidate only when TP is not below the quality reference and FP is not above the quality reference.\n"
        + "- Among quality-preserving candidates, acceptance is based on meaningful lower cost_s versus the current speed best.\n"
        + "- Do not trade away recall, threshold hits, or suppress outputs to look faster or more precise.\n"
        + "\nALGORITHMIC PATCH REQUIREMENTS:\n"
        + "- Implement the approved patch plan faithfully; do not silently switch to an unrelated component or algorithm.\n"
        + "- Prefer result-preserving hot-path C changes: remove redundant computation, simplify branches/loops, cache reused values, or avoid unnecessary arithmetic.\n"
        + "- Do not submit parameter-only edits such as changing rt1/rt2/rt3/min_posterior, iteration counts, E-value thresholds, or standalone literal assignments.\n"
        + "- Submit a concrete C runtime update that should keep TP/FP unchanged while reducing work.\n"
        + "- The C diff must change executable logic in rebuilt jackhmmer/hmmsearch; comments, unreachable branches, and optimized-away no-ops are rejected.\n"
        + "- For null2_correction_neon (or a replayed legacy null2_correction plan), optimize observed-value arithmetic rather than changing constants or correction strength.\n"
        + "- For dp_kernel_neon (or replayed legacy dp_kernel), change the optimized recurrence only in a numerically safe, localized way.\n"
        + "- For ssv_filter_neon (or replayed legacy ssv_filter), preserve SSV score semantics, overflow/J-state fallback behavior, and pass/fail decisions.\n"
        + "- For ssv_filter_neon/get_xE, keep calc_band_15..18 references behind the same #if MAX_BANDS > 14 guard as the stock source.\n"
        + "- For sequence_io, preserve FASTA residue conversion, ignored/EOL/EOD handling, line bookkeeping, ascii->bpos state updates, and name/description parsing.\n"
        + "- For sequence_io/addbuf, do not remove the x <= 127 classification guard unless the replacement still consumes ignored/EOL bytes identically and proves TP/recall preservation.\n"
        + "- For sequence_io/seebuf/header_fasta, do not remove isascii/eslDSQ_* parser error checks or FASTA name/description semantics.\n"
        + "- For sequence_io, do not introduce SSE/SIMD intrinsics or new intrinsic headers into the Easel parser contract; use scalar parser-preserving C changes.\n"
        + "\nPATCH FORMAT REQUIREMENTS:\n"
        + "- Prefer the edits format: each edit.old must be copied exactly from CONTEXT_PACK_JSON source_excerpts, including spaces/tabs/newlines.\n"
        + "- Each edit.old block must occur exactly once in the selected target source; the controller will generate the unified diff programmatically.\n"
        + "- Do not include line numbers from the excerpt prefix in edit.old or edit.new; include only raw C source text.\n"
        + "- Use unified_diff only as a legacy fallback. Exact edit blocks are more reliable than hand-written diff hunks.\n"
        + "- Do not use absolute paths or run-local work paths.\n"
        + "\nREQUIRED_OUTPUT_SCHEMA:\n"
        + json_dumps(pack.required_output_schema)
        + "\n\nMEASURED PROFILE EVIDENCE:\n"
        + json_dumps(pack.profile_evidence)
        + "\n\nMEASURED FINE-GRAINED PROFILE EVIDENCE:\n"
        + json_dumps(_fine_grained_prompt_evidence(pack.profile_evidence))
        + "\n\nRECENT NEGATIVE EXPERIENCE / DO NOT REPEAT:\n"
        + json_dumps(pack.recent_negative_experience)
        + "\nTreat measured profile evidence as authoritative over source-name guesses.\n"
        + "Implement the approved plan's profile_evidence_citation target; do not drift to an unrelated hotspot or non-dominant command.\n"
        + "If symbol/function hotspots are present, the rationale should cite the approved symbol/function target, not just the command name.\n"
        + "If the plan overlaps recent negative experience, preserve its novelty argument in the implementation rationale.\n"
        + "If no profile evidence is present, say so and avoid inventing hotspots.\n"
        + "\nCONTEXT_PACK_JSON:\n"
        + pack.model_dump_json(indent=2)
    )


def _legacy_repair_prompt(
    pack: ContextPackRecord,
    *,
    previous: PatchProposal,
    feedback: str,
    attempt_idx: int,
    plan: PatchPlanProposal | dict[str, Any] | None = None,
) -> str:
    """Render a validation-feedback prompt for one bounded repair attempt."""
    return (
        "Your previous C patch was rejected by deterministic validation.\n"
        "Return a corrected single JSON object with the same schema. Do not explain outside JSON.\n"
        "Use the concrete validator feedback below and the same context pack; do not ask for tools.\n\n"
        f"REPAIR_ATTEMPT: {attempt_idx}\n"
        f"CONTEXT_PACK_ID: {pack.context_pack_id}\n\n"
        "VALIDATOR_FEEDBACK:\n"
        + feedback[:8000]
        + "\n\nPREVIOUS_PROPOSAL_JSON:\n"
        + previous.model_dump_json(indent=2)
        + "\n\nCORRECTION REQUIREMENTS:\n"
        + "- If git apply failed, switch to exact edits: copy edit.old exactly from the source excerpt and let the controller synthesize the diff.\n"
        + "- Keep edits inside one allowed function and one target file.\n"
        + "- If runtime provenance failed, make a real executable-affecting C logic change; comments, unreachable guards, and optimized-away no-ops will be rejected again.\n"
        + "- Do not repeat the same invalid diff unless the feedback says only metadata was wrong.\n\n"
        + _legacy_context_prompt(pack, plan=plan)
    )


def algorithm_plan_prompt(pack: ContextPackRecord) -> str:
    """Render the outer-loop prompt without implementation-only memory."""
    payload = {
        "context_pack_id": pack.context_pack_id,
        "round_idx": pack.round_idx,
        "memory_view": outer_algorithm_memory_view(pack).model_dump(mode="json"),
        "algorithm_family_guidance": ALGORITHM_FAMILY_GUIDANCE,
        "required_output_schema": ALGORITHM_PLAN_OUTPUT_SCHEMA,
    }
    return (
        "Propose one genuinely algorithm-level speed optimization for profile-HMM sequence search.\n"
        "For a new proposal, choose exactly one active semantic family: profile_target_comparison "
        "(one profile/target decision cascade), iterative_search (cross-round model/search/convergence), "
        "dp_recurrence_strategy (AArch64/NEON Forward/Backward DP), ssv_filter_strategy "
        "(AArch64/NEON SSV filter), null2_scoring_strategy (AArch64/NEON composition correction), "
        "or sequence_streaming_strategy (FASTA parsing/buffering/data movement).\n"
        "Legacy aliases remain schema-valid solely so immutable earlier trajectories can be replayed; "
        "they do not create additional source boundaries.\n"
        "Explore decision surfaces, adaptive exact work, reusable sufficient state, bounds, sparse work, "
        "parser/streaming algorithms, or other sequence-search innovations rather than literal tuning.\n"
        "The proposal must explain removed work and either an exact fallback or an equivalent argument "
        "that all externally consumed scores, targets, and decisions remain unchanged.\n"
        "Input parsing and data movement are valid only under sequence_streaming_strategy and must not "
        "skip, reorder, truncate, or alter any target, residue, name, description, or parser error.\n"
        "Do not choose a C file, contract, or function; the controller owns source binding.\n"
        "Do not provide source edits, a unified diff, code, implementation steps, or parameter tuning.\n"
        "TP must not decrease and FP must not increase against the comparison parent.\n"
        "Return exactly one JSON object matching ALGORITHM_PLAN_OUTPUT_SCHEMA.\n\n"
        "ALGORITHM_PLAN_OUTPUT_SCHEMA:\n"
        + json_dumps(ALGORITHM_PLAN_OUTPUT_SCHEMA)
        + "\n\nOUTER_ALGORITHM_PAYLOAD_JSON:\n"
        + _bounded_payload_json(payload, pack, reserved_tokens=OUTER_RESERVED_OUTPUT_TOKENS)
    )


def plan_prompt(
    pack: ContextPackRecord,
    *,
    algorithm_plan: AlgorithmPlanProposal | dict[str, Any] | None = None,
) -> str:
    """Render inner implementation planning for one frozen outer algorithm."""
    memory = inner_implementation_memory_view(pack, algorithm_plan)
    output_schema = _schema_for_algorithm(PLAN_OUTPUT_SCHEMA, algorithm_plan)
    payload = {
        "context_pack_id": pack.context_pack_id,
        "round_idx": pack.round_idx,
        "memory_view": memory.model_dump(mode="json"),
        "required_output_schema": output_schema,
    }
    approved_text = "null" if memory.approved_algorithm_plan is None else json_dumps(memory.approved_algorithm_plan)
    family_guidance = _family_guidance(algorithm_plan)
    return (
        "Plan one concrete C implementation of the approved outer algorithm.\n"
        "Do not output a diff yet. Bind the plan to one allowed contract and its source functions.\n"
        "Do not submit parameter-only edits, thresholds, iteration-count changes, comment-only edits, or no-ops.\n"
        "Explain runtime effectiveness, exact fallback, interface preservation, and TP/FP preservation.\n"
        "The active runtime backend is AArch64/NEON for DP, SSV, and Null2 families; do not redirect "
        "new work to legacy SSE-only contracts.\n"
        "For sequence streaming, parser/buffer/data-movement changes are allowed, but every database "
        "target and its exact parsed metadata/residues must reach the unchanged scoring path.\n"
        "SELECTED FAMILY GUIDANCE: " + family_guidance + "\n"
        "PATCH APPLICATION POLICY:\n- " + pack.application_base_policy + "\n"
        "- The candidate is based on the explicit parent_version_id, never an implicit mutable worktree.\n"
        "- The implementation must be a concrete C algorithm update.\n\n"
        "APPROVED_ALGORITHM_PLAN_JSON:\n" + approved_text + "\n\n"
        "PLAN_OUTPUT_SCHEMA:\n" + json_dumps(output_schema) + "\n\n"
        "INNER_IMPLEMENTATION_MEMORY_VIEW_JSON:\n"
        + _bounded_payload_json(payload, pack, reserved_tokens=INNER_RESERVED_OUTPUT_TOKENS)
    )


def context_prompt(
    pack: ContextPackRecord,
    *,
    plan: PatchPlanProposal | dict[str, Any] | None = None,
    algorithm_plan: AlgorithmPlanProposal | dict[str, Any] | None = None,
) -> str:
    """Render the final inner-loop full-patch prompt as one bounded payload."""
    memory = inner_implementation_memory_view(pack, algorithm_plan)
    output_schema = _schema_for_algorithm(pack.required_output_schema or OUTPUT_SCHEMA, algorithm_plan)
    payload = {
        "context_pack_id": pack.context_pack_id,
        "round_idx": pack.round_idx,
        "memory_view": memory.model_dump(mode="json"),
        "approved_patch_plan": _model_dict(plan),
        "recent_negative_experience": pack.recent_negative_experience,
        "required_output_schema": output_schema,
    }
    family_guidance = _family_guidance(algorithm_plan)
    return (
        "Implement the approved algorithm and inner plan as one complete C patch.\n"
        "Return exactly one JSON object. Do not ask for tools and Do not propose python_payload.\n"
        "Primary objective is lower cost_s; TP must not decrease and FP must not increase.\n"
        "Do not submit parameter-only edits; make a concrete C algorithm update.\n"
        "Preserve the frozen public function input/output interface and per-worker thread safety.\n"
        "Apply this selected-family invariant: " + family_guidance + "\n"
        "PATCH APPLICATION POLICY:\n- " + pack.application_base_policy + "\n"
        "- Exact edit.old text must come from the supplied explicit-parent HMMER source excerpts.\n"
        "- Preserve the x <= 127 parser classification guard for sequence_io and #if MAX_BANDS > 14 guards for SSV contracts.\n"
        "- A full unified_diff is allowed only within the approved algorithm island/contract.\n\n"
        "INNER_IMPLEMENTATION_PAYLOAD_JSON:\n"
        + _bounded_payload_json(payload, pack, reserved_tokens=INNER_RESERVED_OUTPUT_TOKENS)
    )


def repair_prompt(
    pack: ContextPackRecord,
    *,
    previous: PatchProposal,
    feedback: str,
    attempt_idx: int,
    plan: PatchPlanProposal | dict[str, Any] | None = None,
    algorithm_plan: AlgorithmPlanProposal | dict[str, Any] | None = None,
) -> str:
    """Render a focused repair prompt containing only the newest failure."""
    repair = patch_repair_memory_view(pack, feedback, algorithm_plan=algorithm_plan)
    memory = inner_implementation_memory_view(pack, algorithm_plan)
    output_schema = _schema_for_algorithm(pack.required_output_schema or OUTPUT_SCHEMA, algorithm_plan)
    payload = {
        "context_pack_id": pack.context_pack_id,
        "round_idx": pack.round_idx,
        "attempt_idx": attempt_idx,
        "memory_view": memory.model_dump(mode="json"),
        "approved_patch_plan": _model_dict(plan),
        "previous_proposal": previous.model_dump(mode="json"),
        "repair_memory": repair.model_dump(mode="json"),
        "required_output_schema": output_schema,
    }
    return (
        "Repair the current deterministic patch failure without changing the approved algorithm.\n"
        "Return exactly one corrected JSON object. Do not repeat the rejected diff unchanged.\n"
        "PATCH_REPAIR_MEMORY_VIEW_JSON:\n"
        + json_dumps(repair.model_dump(mode="json"))
        + "\n\nINNER_IMPLEMENTATION_PAYLOAD_JSON:\n"
        + _bounded_payload_json(payload, pack, reserved_tokens=INNER_RESERVED_OUTPUT_TOKENS)
    )


def outer_algorithm_memory_view(pack: ContextPackRecord) -> OuterAlgorithmMemoryView:
    """Select algorithm evidence and exclude implementation/workflow details."""
    experience: list[dict[str, Any]] = []
    for item in pack.recent_negative_experience:
        if item.get("scope") == "outer_algorithm":
            experience.append(item)
    for summary in pack.l2_phase_summaries:
        data = summary.model_dump(mode="json") if hasattr(summary, "model_dump") else dict(summary)
        for item in data.get("accepted_speedups", []) + data.get("rejected_mechanisms", []):
            if item.get("scope") == "outer_algorithm":
                experience.append(item)
    for item in pack.l3_durable_wisdom:
        data = item.model_dump(mode="json") if hasattr(item, "model_dump") else dict(item)
        if data.get("wisdom_kind") == "algorithm_wisdom" and data.get("confidence") != "deprecated":
            experience.append(data)
    terms = _profile_terms(pack.profile_evidence)
    return OuterAlgorithmMemoryView(
        objective=pack.objective,
        baseline=_model_dict(pack.baseline),
        current_best=_model_dict(pack.current_best),
        profile_evidence=_outer_profile_evidence(pack.profile_evidence),
        raw_current_trajectory=_bounded_items(_outer_trajectory(pack.recent_history), terms=terms, quota=10, budget=160_000),
        algorithm_experience=_bounded_items(experience, terms=terms, quota=24, budget=48_000),
        compressed_older_experience=_algorithm_summary(pack.compressed_experience),
    )


def inner_implementation_memory_view(
    pack: ContextPackRecord,
    algorithm_plan: AlgorithmPlanProposal | dict[str, Any] | None = None,
) -> InnerImplementationMemoryView:
    """Select source, raw current trajectory, and implementation experience."""
    experience: list[dict[str, Any]] = []
    for item in pack.recent_negative_experience:
        if _experience_scope(item) == "inner_implementation":
            experience.append(item)
    for summary in pack.l2_phase_summaries:
        data = summary.model_dump(mode="json") if hasattr(summary, "model_dump") else dict(summary)
        for item in data.get("accepted_speedups", []) + data.get("rejected_mechanisms", []):
            if item.get("scope") == "inner_implementation":
                experience.append(item)
    for item in pack.l3_durable_wisdom:
        data = item.model_dump(mode="json") if hasattr(item, "model_dump") else dict(item)
        if data.get("wisdom_kind") == "implementation_wisdom" and data.get("confidence") != "deprecated":
            experience.append(data)
    approved = _model_dict(algorithm_plan)
    terms = _profile_terms(pack.profile_evidence) | _text_terms(json_dumps(approved))
    target_contracts = {
        str(name) for name in (approved or {}).get("target_contracts", [])
        if str(name) in ACTIVE_C_KERNEL_CONTRACTS
    }
    allowed_contracts = [
        item for item in pack.allowed_c_contracts
        if item.name in ACTIVE_C_KERNEL_CONTRACTS
        and (not target_contracts or item.name in target_contracts)
    ]
    source_data = [
        item.model_dump(mode="json") for item in pack.source_excerpts
        if item.component_name in ACTIVE_C_KERNEL_CONTRACTS
        and (not target_contracts or item.component_name in target_contracts)
    ]
    return InnerImplementationMemoryView(
        objective=pack.objective,
        approved_algorithm_plan=approved,
        parent_version_id=pack.parent_patchset_id or pack.current_best_patchset_id,
        allowed_c_contracts=[item.model_dump(mode="json") for item in allowed_contracts],
        source_excerpts=_bounded_items(source_data, terms=terms, quota=len(source_data), budget=600_000),
        profile_evidence=pack.profile_evidence,
        raw_current_trajectory=_bounded_items(pack.recent_history, terms=terms, quota=10, budget=500_000),
        implementation_experience=_bounded_items(experience, terms=terms, quota=32, budget=64_000),
        compressed_older_experience=_implementation_summary(pack.compressed_experience),
        application_base_policy=pack.application_base_policy,
    )


def patch_repair_memory_view(
    pack: ContextPackRecord,
    feedback: str | dict[str, Any],
    *,
    algorithm_plan: AlgorithmPlanProposal | dict[str, Any] | None = None,
) -> PatchRepairMemoryView:
    """Keep only the latest validator failure and closely matching workflow lessons."""
    if isinstance(feedback, str):
        try:
            parsed = json.loads(feedback)
        except json.JSONDecodeError:
            parsed = {"failure_kind": "validation_error", "errors": [feedback[:8000]]}
    else:
        parsed = dict(feedback)
    records = parsed.get("validation_records") or []
    current = records[-1] if records else parsed
    if not isinstance(current, dict):
        current = {"failure_kind": "validation_error", "errors": [str(current)]}
    binding = current.get("plan_diff_binding") or current.get("binding_delta") or {}
    if isinstance(binding, dict) and isinstance(binding.get("containment"), dict):
        binding = {**binding, **binding.get("containment", {})}
    build = current.get("build_comparisons") or current.get("build_provenance_delta") or {}
    terms = _text_terms(json_dumps(current))
    live_experience = LoopStore(pack.run_dir).iter_experience()
    repair_experience = [
        item for item in [*pack.recent_negative_experience, *live_experience]
        if _experience_scope(item) == "workflow"
    ]
    return PatchRepairMemoryView(
        approved_algorithm_plan=_model_dict(algorithm_plan),
        current_validation_failure=current,
        binding_delta=binding if isinstance(binding, dict) else {},
        build_provenance_delta=build if isinstance(build, dict) else {},
        repair_experience=_bounded_items(repair_experience, terms=terms, quota=12, budget=32_000),
    )


def _bounded_items(
    items: list[dict[str, Any]],
    *,
    terms: set[str],
    quota: int,
    budget: int,
) -> list[dict[str, Any]]:
    """Select complete, semantically unique records under a JSON character budget."""
    unique: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in reversed(items):
        semantic_id = str(
            item.get("semantic_id") or item.get("experience_id") or item.get("wisdom_id")
            or item.get("eval_id") or item.get("patchset_id") or record_hash(item)
        )
        if semantic_id in seen:
            continue
        seen.add(semantic_id)
        unique.append(item)
    def score(item: dict[str, Any]) -> int:
        text = json.dumps(item, ensure_ascii=False, sort_keys=True).lower()
        return sum(1 for term in terms if term and term in text)
    indexed = list(enumerate(unique))
    indexed.sort(key=lambda pair: (-score(pair[1]), pair[0]))
    selected: list[dict[str, Any]] = []
    for _, item in indexed:
        if len(selected) >= quota:
            break
        candidate = selected + [item]
        if len(json.dumps(candidate, ensure_ascii=False, sort_keys=True)) <= budget:
            selected.append(item)
    return selected


def _bounded_payload_json(payload: dict[str, Any], pack: ContextPackRecord, *, reserved_tokens: int) -> str:
    max_chars = max(16_000, (pack.context_window_tokens - reserved_tokens) * _CHARS_PER_TOKEN - 16_000)
    text = json_dumps(payload)
    if len(text) <= max_chars:
        return text
    compact = json.loads(json.dumps(payload, ensure_ascii=False))
    memory = compact.get("memory_view") or {}
    for key in ("compressed_older_experience", "implementation_experience", "algorithm_experience", "raw_current_trajectory"):
        if key in memory:
            memory[key] = [] if isinstance(memory[key], list) else ""
        text = json_dumps(compact)
        if len(text) <= max_chars:
            return text
    if isinstance(memory.get("source_excerpts"), list):
        source = memory["source_excerpts"]
        memory["source_excerpts"] = []
        for excerpt in source:
            memory["source_excerpts"].append(excerpt)
            if len(json_dumps(compact)) > max_chars:
                memory["source_excerpts"].pop()
                break
    return json_dumps(compact)


def _model_dict(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    return dict(value)


def _family_guidance(
    algorithm_plan: AlgorithmPlanProposal | dict[str, Any] | None,
) -> str:
    plan = _model_dict(algorithm_plan) or {}
    family = str(plan.get("algorithm_family") or "")
    return ALGORITHM_FAMILY_GUIDANCE.get(
        family,
        "Preserve every externally consumed target, score, hit decision, and TP/FP outcome while removing measured runtime work.",
    )


def _schema_for_algorithm(
    schema: dict[str, Any],
    algorithm_plan: AlgorithmPlanProposal | dict[str, Any] | None,
) -> dict[str, Any]:
    """Bind an inner output schema to the controller-approved island."""
    bounded = json.loads(json.dumps(schema, ensure_ascii=False))
    plan = _model_dict(algorithm_plan) or {}
    names = sorted({
        str(name) for name in plan.get("target_contracts", [])
        if str(name) in ACTIVE_C_KERNEL_CONTRACTS
    })
    properties = bounded.get("properties") or {}
    component = properties.get("component_name")
    if names and isinstance(component, dict):
        component["enum"] = names
    return bounded


def _text_terms(text: str) -> set[str]:
    return {token.lower() for token in re.findall(r"[A-Za-z_][A-Za-z0-9_]{3,}", text)}


def _profile_terms(profile: dict[str, Any]) -> set[str]:
    return _text_terms(json.dumps(profile, ensure_ascii=False, sort_keys=True))


def _outer_profile_evidence(profile: dict[str, Any]) -> dict[str, Any]:
    """Expose only current-runtime contracts, including sequence streaming."""
    data = json.loads(json.dumps(profile, ensure_ascii=False))
    for key in ("recommended_contracts", "contract_hotspots", "function_hotspots", "fine_grained_hotspots"):
        if isinstance(data.get(key), list):
            data[key] = [
                item for item in data[key]
                if not item.get("component_name")
                or item.get("component_name") in ACTIVE_C_KERNEL_CONTRACTS
            ]
    return data


def _outer_trajectory(history: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Project current raw evidence without exposing implementation code."""
    out: list[dict[str, Any]] = []
    for item in history:
        out.append({
            "round_idx": item.get("round_idx"),
            "algorithm_plan": item.get("algorithm_plan"),
            "evaluation": item.get("evaluation"),
            "parent_version_id": item.get("parent_version_id"),
            "patchset_id": item.get("patchset_id"),
        })
    return out


def _algorithm_summary(text: str) -> str:
    # Parser, cache, and data-movement mechanisms are now genuine outer
    # algorithm evidence; retain them while bounding the older summary.
    return text[-16_000:]


def _implementation_summary(text: str) -> str:
    workflow_markers = (
        "workflow", "plan_validation_failure", "proposal_error", "patch_apply_failure",
        "compile_failure", "runtime_failure", "scope_violation", "repair lessons",
    )
    lines = [line for line in text.splitlines() if not any(term in line.lower() for term in workflow_markers)]
    return "\n".join(lines)[-24_000:]


def _experience_scope(item: dict[str, Any]) -> str:
    scope = item.get("scope")
    if scope:
        return str(scope)
    if item.get("lesson_kind") == "workflow" or item.get("outcome") in {
        "compile_failure", "patch_apply_failure", "runtime_failure",
        "runtime_provenance_failure", "scope_violation", "plan_validation_failure",
        "plan_proposal_error", "patch_proposal_error", "repair_proposal_error",
        "fatal_proposal_error",
    }:
        return "workflow"
    return "inner_implementation"


def _phase_summaries(store: LoopStore) -> list[PhaseSummaryRecord]:
    raw = store.read_index("phase_summaries")
    if not isinstance(raw, list):
        return []
    out: list[PhaseSummaryRecord] = []
    for item in raw[-12:]:
        try:
            out.append(PhaseSummaryRecord.model_validate(item))
        except (TypeError, ValueError):
            continue
    return out


def _source_excerpts(source_root: Path) -> list[SourceExcerpt]:
    excerpts: list[SourceExcerpt] = []
    for component_name, contract in ACTIVE_C_KERNEL_CONTRACTS.items():
        target_file = contract.c_target_file or ""
        source_file = _find_source_file(source_root, target_file)
        if not source_file.exists():
            excerpts.append(SourceExcerpt(
                component_name=component_name,
                file=target_file,
                function="<missing>",
                start_line=0,
                end_line=0,
                source_hash="missing",
                text=f"Source file not found under {source_root}: {target_file}",
            ))
            continue
        source_hash = hash_file(source_file)
        lines = source_file.read_text(encoding="utf-8", errors="replace").splitlines()
        for function in contract.c_target_functions:
            start, end = _function_range(lines, function)
            if start is None or end is None:
                text = f"Function {function} not found in {source_file}."
                excerpts.append(SourceExcerpt(
                    component_name=component_name,
                    file=str(source_file.relative_to(source_root)) if source_file.is_relative_to(source_root) else str(source_file),
                    function=function,
                    start_line=0,
                    end_line=0,
                    source_hash=source_hash,
                    text=text,
                ))
                continue
            body = "\n".join(f"{idx:5d}\t{lines[idx - 1]}" for idx in range(start, end + 1))
            excerpts.append(SourceExcerpt(
                component_name=component_name,
                file=str(source_file.relative_to(source_root)) if source_file.is_relative_to(source_root) else str(source_file),
                function=function,
                start_line=start,
                end_line=end,
                source_hash=source_hash,
                text=body,
            ))
    return excerpts


def _find_source_file(source_root: Path, target_file: str) -> Path:
    candidates = [
        source_root / target_file,
        source_root / "src" / target_file,
        source_root / "src" / Path(target_file).name,
        source_root / Path(target_file).name,
        # Split checkout: easel/ lives beside src/, so a target of
        # "easel/esl_sqio_ascii.c" resolves one level above the source root.
        source_root.parent / target_file,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _function_range(lines: list[str], function: str) -> tuple[int | None, int | None]:
    pattern = re.compile(rf"\b{re.escape(function)}\s*\(")
    for idx, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith(("/*", "*", "//")):
            continue
        if not pattern.search(line):
            continue
        brace_idx = _definition_brace_index(lines, idx)
        if brace_idx is None:
            continue
        end = _matching_function_end(lines, brace_idx)
        if end is not None:
            return idx + 1, end + 1
    return None, None


def _definition_brace_index(lines: list[str], start_idx: int) -> int | None:
    for idx in range(start_idx, min(start_idx + 8, len(lines))):
        line = lines[idx]
        semi = line.find(";")
        brace = line.find("{")
        if brace != -1 and (semi == -1 or brace < semi):
            return idx
        if semi != -1 and (brace == -1 or semi < brace):
            return None
    return None


def _matching_function_end(lines: list[str], brace_idx: int) -> int | None:
    depth = 0
    started = False
    in_block_comment = False
    for idx in range(brace_idx, len(lines)):
        code, in_block_comment = _strip_c_comments_and_strings(lines[idx], in_block_comment)
        for ch in code:
            if ch == "{":
                depth += 1
                started = True
            elif ch == "}":
                depth -= 1
                if started and depth == 0:
                    return idx
    return None


def _strip_c_comments_and_strings(line: str, in_block_comment: bool) -> tuple[str, bool]:
    out: list[str] = []
    i = 0
    in_string: str | None = None
    while i < len(line):
        ch = line[i]
        nxt = line[i + 1] if i + 1 < len(line) else ""
        if in_block_comment:
            if ch == "*" and nxt == "/":
                in_block_comment = False
                i += 2
            else:
                i += 1
            continue
        if in_string:
            if ch == "\\":
                i += 2
                continue
            if ch == in_string:
                in_string = None
            i += 1
            continue
        if ch == "/" and nxt == "*":
            in_block_comment = True
            i += 2
            continue
        if ch == "/" and nxt == "/":
            break
        if ch in {'"', "'"}:
            in_string = ch
            i += 1
            continue
        out.append(ch)
        i += 1
    return "".join(out), in_block_comment


def _profile_evidence(
    store: LoopStore,
    *,
    baseline: EvalSummary | None,
    current_best: EvalSummary | None,
    max_recent: int = 5,
) -> dict[str, Any]:
    evaluations = store.iter_evaluations()
    by_eval_id = {ev.get("eval_id"): ev for ev in evaluations if ev.get("eval_id")}
    patchsets = {p.get("patchset_id"): p for p in store.iter_patchsets()}

    baseline_ev = by_eval_id.get(baseline.eval_id) if baseline and baseline.eval_id else None
    current_ev = by_eval_id.get(current_best.eval_id) if current_best and current_best.eval_id else None
    recent: list[dict[str, Any]] = []
    for ev in sorted(evaluations, key=lambda item: str(item.get("created_at") or item.get("eval_id")), reverse=True):
        if ev.get("stage") not in {"smoke", "full"}:
            continue
        profile = _eval_profile(ev)
        if not profile:
            continue
        patchset = patchsets.get(ev.get("patchset_id"), {})
        patch = (patchset.get("patches") or [{}])[-1]
        recent.append({
            "eval_id": ev.get("eval_id"),
            "patchset_id": ev.get("patchset_id"),
            "round_idx": ev.get("round_idx"),
            "stage": ev.get("stage"),
            "decision": ev.get("decision"),
            "component_name": patch.get("component_name"),
            "metrics": _metric_view(ev),
            "profile": _compact_eval_profile(profile),
        })
        if len(recent) >= max_recent:
            break

    evidence = {
        "schema_version": "seq_align_profile_evidence.v2",
        "baseline": _profile_record(baseline_ev, label="baseline") if baseline_ev else None,
        "current_best": _profile_record(current_ev, label="current_best") if current_ev else None,
        "recent_profiles": recent,
        "observations": _profile_observations(baseline_ev, current_ev, recent),
    }
    evidence["has_profile"] = bool(evidence["baseline"] or evidence["current_best"] or recent)
    fine_hotspots = _collect_fine_grained_hotspots(evidence)
    evidence["fine_grained_hotspots"] = fine_hotspots
    evidence["function_hotspots"] = _function_hotspots_from_symbols(fine_hotspots)
    evidence["contract_hotspots"] = _contract_hotspots(evidence["function_hotspots"])
    evidence["has_symbol_profile"] = bool(fine_hotspots)
    evidence.update(_profile_guidance(evidence))
    return evidence


def _profile_guidance(evidence: dict[str, Any]) -> dict[str, Any]:
    symbol_contracts = [
        item for item in (evidence.get("contract_hotspots") or [])
        if item.get("component_name") in ACTIVE_C_KERNEL_CONTRACTS
    ]
    if symbol_contracts:
        recommended = []
        for item in symbol_contracts[:5]:
            component = str(item.get("component_name"))
            measured_functions = list(item.get("functions") or [])
            recommended.append({
                "component_name": component,
                "target_functions": _active_target_functions(component, measured_functions),
                "measured_functions": measured_functions,
                "priority": "measured-symbol-hotspot",
                "reason": item.get("evidence") or "external symbol profile mapped this contract to measured runtime hotspots.",
                "guidance": (
                    "Address the cited measured hotspot through the active contract while preserving "
                    "scores, target visibility, and downstream decisions."
                ),
            })
        return {
            "recommended_contracts": recommended,
            "function_hotspots": [
                item for item in (evidence.get("function_hotspots") or [])
                if item.get("component_name") in ACTIVE_C_KERNEL_CONTRACTS
            ],
            "avoid_targets": [
                "Do not ignore measured symbol/function hotspots when has_symbol_profile is true.",
                "A generic command-level citation is insufficient once fine-grained evidence is available.",
            ],
        }
    dominant, fraction = _profile_guidance_dominant(evidence)
    if not evidence.get("has_profile") or not dominant:
        return {
            "recommended_contracts": [],
            "function_hotspots": [],
            "avoid_targets": ["No measured profile evidence is available yet; avoid inventing command or function hotspots."],
        }
    if dominant in {"jackhmmer", "jackhmmer_iterations"}:
        frac_text = f"{fraction:.4f}" if fraction is not None else "unknown"
        evidence_key = "dominant_phase" if dominant == "jackhmmer_iterations" else "dominant_command"
        return {
            "recommended_contracts": [
                {
                    "component_name": "jackhmmer_loop" if dominant == "jackhmmer_iterations" else "pipeline_flow",
                    "target_functions": ["p7_evo_iter_BeginRound", "p7_evo_iter_EndRound"] if dominant == "jackhmmer_iterations" else ["p7_evo_align_Run"],
                    "priority": "high",
                    "reason": f"Measured profile {evidence_key}={dominant} with wall_fraction={frac_text}; this is the repeated search control path.",
                    "guidance": "Prefer a high-level search/pipeline decision change with an exact fallback.",
                },
                {
                    "component_name": "pipeline_flow" if dominant == "jackhmmer_iterations" else "jackhmmer_loop",
                    "target_functions": ["p7_evo_align_Run"] if dominant == "jackhmmer_iterations" else ["p7_evo_iter_BeginRound", "p7_evo_iter_EndRound"],
                    "priority": "high",
                    "reason": "The per-target filter cascade and cross-iteration loop jointly determine repeated jackhmmer work.",
                    "guidance": "Preserve reporting/inclusion semantics and use the stock exact path for borderline cases.",
                },
                {
                    "component_name": "dp_kernel_neon",
                    "target_functions": ["forward_engine", "backward_engine"],
                    "priority": "secondary",
                    "reason": "The active NEON DP recurrence is repeated downstream of the pipeline decision surface.",
                    "guidance": "Preserve recurrence scores and scaling; avoid threshold/default tuning.",
                },
                {
                    "component_name": "ssv_filter_neon",
                    "target_functions": ["get_xE", "p7_SSVFilter"],
                    "priority": "secondary",
                    "reason": "The active NEON SSV filter executes early for database targets in the jackhmmer cascade.",
                    "guidance": "Preserve filter scores, pass/fail decisions, overflow fallback, and MAX_BANDS guards.",
                },
                {
                    "component_name": "sequence_io",
                    "target_functions": ["seebuf", "addbuf", "header_fasta"],
                    "priority": "secondary",
                    "reason": "FASTA parsing and residue conversion are repeated for the searched target stream.",
                    "guidance": "Preserve every target, database order, residues, metadata, and parser errors.",
                },
            ],
            "function_hotspots": [
                {
                    "command": "jackhmmer",
                    "component_name": "pipeline_flow",
                    "functions": ["p7_evo_align_Run"],
                    "evidence": f"{evidence_key}={dominant} wall_fraction={frac_text}",
                },
                {
                    "command": "jackhmmer",
                    "component_name": "jackhmmer_loop",
                    "functions": ["p7_evo_iter_BeginRound", "p7_evo_iter_EndRound"],
                    "evidence": "cross-iteration jackhmmer search loop",
                },
            ],
            "avoid_targets": [
                "hmmsearch-only changes have negligible expected ROI when jackhmmer dominates measured wall time.",
                "Do not optimize speed by reducing iterations, suppressing reported hits, changing E-value cutoffs, or changing TP/FP-sensitive thresholds.",
                "Do not target a legacy SSE-only contract when the measured runtime uses the active AArch64/NEON backend.",
            ],
        }
    if dominant == "load_targets":
        return {
            "recommended_contracts": [
                {
                    "component_name": "sequence_io",
                    "target_functions": ["seebuf", "addbuf", "header_fasta"],
                    "priority": "high",
                    "reason": f"Measured dominant_phase=load_targets with wall_fraction={fraction}; this is the active FASTA streaming/materialization path.",
                    "guidance": "Reduce parser, buffering, conversion, or data-movement work while preserving every parsed target and parser semantic.",
                }
            ],
            "function_hotspots": [
                {
                    "command": "load_targets",
                    "component_name": "sequence_io",
                    "functions": ["seebuf", "addbuf", "header_fasta"],
                    "evidence": f"dominant_phase=load_targets wall_fraction={fraction}",
                }
            ],
            "avoid_targets": [
                "Do not skip, reorder, truncate, prefilter, or alter targets to make load_targets appear faster.",
                "Do not weaken residue classification, line/EOD handling, FASTA metadata, or parser-error behavior.",
            ],
        }
    return {
        "recommended_contracts": [],
        "function_hotspots": [
            {
                "command": dominant,
                "component_name": None,
                "functions": [],
                "evidence": f"dominant_command={dominant} wall_fraction={fraction}",
            }
        ],
        "avoid_targets": [f"Do not ignore measured dominant_command={dominant}; explain any chosen C contract against that profile evidence."],
    }



def _collect_fine_grained_hotspots(evidence: dict[str, Any], *, limit: int = 12) -> list[dict[str, Any]]:
    hotspots: list[dict[str, Any]] = []
    sources = []
    for key in ("current_best", "baseline"):
        record = evidence.get(key)
        if record:
            sources.append(record)
    sources.extend(evidence.get("recent_profiles") or [])
    seen: set[tuple[str, str, str | None]] = set()
    for record in sources:
        profile = (record.get("profile") or {}) if isinstance(record, dict) else {}
        for symbol in profile.get("symbol_hotspots") or []:
            if not isinstance(symbol, dict):
                continue
            row = dict(symbol)
            row.setdefault("eval_id", record.get("eval_id"))
            row.setdefault("patchset_id", record.get("patchset_id"))
            function = _symbol_to_function(str(row.get("symbol") or ""))
            mapping = _function_contract_mapping(function or str(row.get("symbol") or ""))
            if function:
                row["function"] = function
            if mapping:
                row.update(mapping)
            key = (str(row.get("command") or ""), str(row.get("symbol") or ""), row.get("eval_id"))
            if key in seen:
                continue
            seen.add(key)
            hotspots.append(row)
            if len(hotspots) >= limit:
                return hotspots
    return hotspots


def _function_hotspots_from_symbols(symbol_hotspots: list[dict[str, Any]], *, limit: int = 10) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], dict[str, Any]] = {}
    for symbol in symbol_hotspots:
        function = symbol.get("function") or _symbol_to_function(str(symbol.get("symbol") or ""))
        mapping = _function_contract_mapping(str(function or ""))
        if not function or not mapping:
            continue
        key = (str(mapping["component_name"]), str(function))
        item = grouped.setdefault(key, {
            "component_name": mapping["component_name"],
            "function": function,
            "functions": [function],
            "command": symbol.get("command"),
            "symbols": [],
            "approx_command_wall_s": 0.0,
            "evidence_refs": [],
        })
        item["approx_command_wall_s"] = round(float(item.get("approx_command_wall_s") or 0.0) + float(symbol.get("approx_command_wall_s") or 0.0), 4)
        if symbol.get("symbol") not in item["symbols"]:
            item["symbols"].append(symbol.get("symbol"))
        ref = symbol.get("eval_id") or symbol.get("query_id") or symbol.get("symbol")
        if ref and ref not in item["evidence_refs"]:
            item["evidence_refs"].append(ref)
    out = sorted(grouped.values(), key=lambda item: float(item.get("approx_command_wall_s") or 0.0), reverse=True)
    return out[:limit]


def _contract_hotspots(function_hotspots: list[dict[str, Any]], *, limit: int = 5) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    for hotspot in function_hotspots:
        component = hotspot.get("component_name")
        if not component:
            continue
        item = grouped.setdefault(str(component), {
            "component_name": component,
            "functions": [],
            "approx_command_wall_s": 0.0,
            "evidence_refs": [],
        })
        item["approx_command_wall_s"] = round(float(item.get("approx_command_wall_s") or 0.0) + float(hotspot.get("approx_command_wall_s") or 0.0), 4)
        function = hotspot.get("function")
        if function and function not in item["functions"]:
            item["functions"].append(function)
        for ref in hotspot.get("evidence_refs") or []:
            if ref not in item["evidence_refs"]:
                item["evidence_refs"].append(ref)
    out: list[dict[str, Any]] = []
    for item in sorted(grouped.values(), key=lambda row: float(row.get("approx_command_wall_s") or 0.0), reverse=True)[:limit]:
        item = dict(item)
        item["evidence"] = f"external symbol profile maps {item['functions']} to {item['component_name']} approx_command_wall_s={item.get('approx_command_wall_s')}"
        out.append(item)
    return out


def _symbol_to_function(symbol: str) -> str | None:
    clean = symbol.strip()
    if not clean:
        return None
    clean = clean.split("+", 1)[0]
    clean = re.sub(r"\.(?:isra|constprop)(?:\.\d+)?$", "", clean)
    if clean in _FUNCTION_TO_CONTRACT:
        return clean
    for function in sorted(_FUNCTION_TO_CONTRACT, key=len, reverse=True):
        if function in clean:
            return function
    return None


_FUNCTION_TO_CONTRACT: dict[str, dict[str, str]] = {
    "p7_Pipeline": {"component_name": "pipeline_flow", "c_target_file": "p7_evo_align.c"},
    "p7_evo_align_Run": {"component_name": "pipeline_flow", "c_target_file": "p7_evo_align.c"},
    "serial_loop": {"component_name": "jackhmmer_loop", "c_target_file": "p7_evo_iter.c"},
    "p7_Builder": {"component_name": "jackhmmer_loop", "c_target_file": "p7_evo_iter.c"},
    "p7_tophits_Threshold": {"component_name": "jackhmmer_loop", "c_target_file": "p7_evo_iter.c"},
    "p7_evo_iter_BeginRound": {"component_name": "jackhmmer_loop", "c_target_file": "p7_evo_iter.c"},
    "p7_evo_iter_EndRound": {"component_name": "jackhmmer_loop", "c_target_file": "p7_evo_iter.c"},
    "forward_engine": {"component_name": "dp_kernel_neon", "c_target_file": "impl_neon/fwdback.c"},
    "backward_engine": {"component_name": "dp_kernel_neon", "c_target_file": "impl_neon/fwdback.c"},
    "get_xE": {"component_name": "ssv_filter_neon", "c_target_file": "impl_neon/ssvfilter.c"},
    "p7_SSVFilter": {"component_name": "ssv_filter_neon", "c_target_file": "impl_neon/ssvfilter.c"},
    "seebuf": {"component_name": "sequence_io", "c_target_file": "easel/esl_sqio_ascii.c"},
    "addbuf": {"component_name": "sequence_io", "c_target_file": "easel/esl_sqio_ascii.c"},
    "header_fasta": {"component_name": "sequence_io", "c_target_file": "easel/esl_sqio_ascii.c"},
    "p7_Null2_ByExpectation": {"component_name": "null2_correction_neon", "c_target_file": "impl_neon/null2.c"},
    "p7_Null2_ByTrace": {"component_name": "null2_correction_neon", "c_target_file": "impl_neon/null2.c"},
    "p7_domaindef_ByPosteriorHeuristics": {"component_name": "pipeline_flow", "c_target_file": "p7_evo_align.c"},
}
_FUNCTION_TO_CONTRACT.update({
    f"calc_band_{idx}": {"component_name": "ssv_filter_neon", "c_target_file": "impl_neon/ssvfilter.c"}
    for idx in range(1, 19)
})


def _function_contract_mapping(function: str) -> dict[str, str] | None:
    if function in _FUNCTION_TO_CONTRACT:
        return dict(_FUNCTION_TO_CONTRACT[function])
    if function.startswith("p7_domaindef_"):
        return {"component_name": "pipeline_flow", "c_target_file": "p7_evo_align.c"}
    return None


def _active_target_functions(component_name: str, measured: list[str]) -> list[str]:
    """Translate measured (possibly frozen-driver) symbols to editable functions."""
    contract = ACTIVE_C_KERNEL_CONTRACTS.get(component_name)
    if contract is None:
        return []
    allowed = list(contract.c_target_functions or [])
    overlap = [function for function in measured if function in allowed]
    if overlap:
        return overlap
    preferred = {
        "pipeline_flow": ["p7_evo_align_Run"],
        "jackhmmer_loop": ["p7_evo_iter_BeginRound", "p7_evo_iter_EndRound"],
    }.get(component_name)
    return [function for function in (preferred or allowed) if function in allowed]


def _fine_grained_prompt_evidence(profile_evidence: dict[str, Any]) -> dict[str, Any]:
    return {
        "has_symbol_profile": profile_evidence.get("has_symbol_profile", False),
        "fine_grained_hotspots": (profile_evidence.get("fine_grained_hotspots") or [])[:8],
        "function_hotspots": (profile_evidence.get("function_hotspots") or [])[:8],
        "contract_hotspots": (profile_evidence.get("contract_hotspots") or [])[:5],
    }


def _profile_guidance_dominant(evidence: dict[str, Any]) -> tuple[str | None, float | None]:
    for key in ("current_best", "baseline"):
        record = evidence.get(key) or {}
        profile = record.get("profile") or {}
        dominant = profile.get("dominant_command")
        if dominant:
            fractions = profile.get("command_wall_fraction") or {}
            fraction = fractions.get(dominant)
            return str(dominant), float(fraction) if fraction is not None else None
        dominant = profile.get("dominant_algorithm_phase") or profile.get("dominant_phase")
        if dominant:
            fractions = profile.get("algorithm_phase_wall_fraction") or profile.get("phase_wall_fraction") or {}
            fraction = fractions.get(dominant)
            return str(dominant), float(fraction) if fraction is not None else None
    for item in evidence.get("recent_profiles") or []:
        profile = item.get("profile") or {}
        dominant = profile.get("dominant_command")
        if dominant:
            fractions = profile.get("command_wall_fraction") or {}
            fraction = fractions.get(dominant)
            return str(dominant), float(fraction) if fraction is not None else None
        dominant = profile.get("dominant_algorithm_phase") or profile.get("dominant_phase")
        if dominant:
            fractions = profile.get("algorithm_phase_wall_fraction") or profile.get("phase_wall_fraction") or {}
            fraction = fractions.get(dominant)
            return str(dominant), float(fraction) if fraction is not None else None
    return None, None



def _profile_record(ev: dict[str, Any] | None, *, label: str) -> dict[str, Any] | None:
    if not ev:
        return None
    profile = _eval_profile(ev)
    if not profile:
        return None
    return {
        "label": label,
        "eval_id": ev.get("eval_id"),
        "patchset_id": ev.get("patchset_id"),
        "round_idx": ev.get("round_idx"),
        "stage": ev.get("stage"),
        "decision": ev.get("decision"),
        "metrics": _metric_view(ev),
        "profile": _compact_eval_profile(profile),
    }


def _eval_profile(ev: dict[str, Any]) -> dict[str, Any]:
    result = ev.get("eval_result") or {}
    meta = result.get("meta") or {}
    profile = meta.get("profile") or {}
    return profile if isinstance(profile, dict) else {}


def _metric_view(ev: dict[str, Any]) -> dict[str, Any]:
    result = ev.get("eval_result") or {}
    return {
        "tp": result.get("tp", 0),
        "fp": result.get("fp", 0),
        "tp_fp_ratio": result.get("tp_fp_ratio", 0.0),
        "cost_s": result.get("cost_s", 0.0),
        "status": result.get("status", ev.get("status")),
    }


def _compact_eval_profile(profile: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": profile.get("schema_version"),
        "dominant_command": profile.get("dominant_command"),
        "command_wall_s": profile.get("command_wall_s") or {},
        "command_wall_fraction": profile.get("command_wall_fraction") or {},
        "backend": profile.get("backend"),
        "dominant_phase": profile.get("dominant_phase"),
        "phase_wall_s": profile.get("phase_wall_s") or {},
        "phase_wall_fraction": profile.get("phase_wall_fraction") or {},
        "dominant_algorithm_phase": profile.get("dominant_algorithm_phase"),
        "algorithm_phase_wall_s": profile.get("algorithm_phase_wall_s") or {},
        "algorithm_phase_wall_fraction": profile.get("algorithm_phase_wall_fraction") or {},
        "query_cost_stats": profile.get("query_cost_stats") or {},
        "iteration_stats": profile.get("iteration_stats") or {},
        "iteration_hit_stats": profile.get("iteration_hit_stats") or {},
        "top_slowest_queries": (profile.get("top_slowest_queries") or [])[:5],
        "external_profiler": _compact_external_profiler(profile.get("external_profiler") or {}),
        "symbol_hotspots": (profile.get("symbol_hotspots") or [])[:10],
        "command_symbol_hotspots": _compact_command_symbol_hotspots(profile.get("command_symbol_hotspots") or {}),
        "repeat_strategy": profile.get("repeat_strategy"),
        "selected_repeat_idx": profile.get("selected_repeat_idx"),
        "repeat_aggregate": profile.get("repeat_aggregate") or {},
    }


def _compact_external_profiler(data: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(data, dict) or not data:
        return {}
    return {
        "tool": data.get("tool"),
        "status": data.get("status"),
        "excluded_from_cost_s": data.get("excluded_from_cost_s"),
        "strategy": data.get("strategy"),
        "top_queries_requested": data.get("top_queries_requested"),
        "top_symbols_requested": data.get("top_symbols_requested"),
        "frequency": data.get("frequency"),
        "command_selector": data.get("command_selector"),
        "reason": data.get("reason"),
    }


def _compact_command_symbol_hotspots(data: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(data, dict):
        return {}
    return {str(command): list(rows or [])[:5] for command, rows in data.items() if isinstance(rows, list)}


def _profile_observations(
    baseline_ev: dict[str, Any] | None,
    current_ev: dict[str, Any] | None,
    recent: list[dict[str, Any]],
) -> list[str]:
    out: list[str] = []
    baseline_cost = float(((baseline_ev or {}).get("eval_result") or {}).get("cost_s") or 0.0)
    if baseline_ev and _eval_profile(baseline_ev):
        profile = _eval_profile(baseline_ev)
        dominant = profile.get("dominant_command")
        phase = profile.get("dominant_algorithm_phase") or profile.get("dominant_phase")
        if dominant:
            out.append(f"Baseline measured dominant command is {dominant}; target measured runtime hotspots rather than source-name guesses.")
        elif phase:
            fractions = profile.get("algorithm_phase_wall_fraction") or profile.get("phase_wall_fraction") or {}
            out.append(f"Baseline measured dominant_phase={phase}; phase_wall_fraction={fractions.get(phase)}.")
    if current_ev and baseline_ev and current_ev.get("eval_id") != baseline_ev.get("eval_id"):
        current_cost = float(((current_ev or {}).get("eval_result") or {}).get("cost_s") or 0.0)
        if baseline_cost and current_cost:
            ratio = round(current_cost / baseline_cost, 4)
            out.append(f"Current speed best costs {ratio}x baseline cost_s; preserve TP/FP while improving cost_s.")
    slower_quality = []
    for item in recent:
        metrics = item.get("metrics") or {}
        cost = float(metrics.get("cost_s") or 0.0)
        if item.get("decision") == "rejected_not_faster" and baseline_cost and cost:
            slower_quality.append(round(cost / baseline_cost, 4))
    if slower_quality:
        out.append(f"Recent quality-preserving candidates were slower than stock: cost ratios vs baseline={slower_quality[:5]}.")
    if not out:
        out.append("No measured profile evidence is available yet; avoid inventing hotspots.")
    return out


def _recent_negative_experience(store: LoopStore, *, max_items: int = 8) -> list[dict[str, Any]]:
    items = store.recent_negative_experience(max_items=max_items)
    if isinstance(items, list):
        return items[:max_items]
    return []


def _recent_history(store: LoopStore, *, max_recent: int) -> list[dict[str, Any]]:
    evaluations = store.iter_evaluations()
    patchsets = {p.get("patchset_id"): p for p in store.iter_patchsets()}
    patch_plans = {p.get("patch_plan_id"): p for p in store.iter_patch_plans()}
    algorithm_plans = {p.get("algorithm_plan_id"): p for p in store.iter_algorithm_plans()}
    trials = {p.get("implementation_trial_id"): p for p in store.iter_implementation_trials()}
    out: list[dict[str, Any]] = []
    for ev in sorted(evaluations, key=lambda item: str(item.get("created_at") or item.get("eval_id")), reverse=True):
        patchset = patchsets.get(ev.get("patchset_id"), {})
        patch_plan = patch_plans.get(patchset.get("patch_plan_id")) or {}
        algorithm_plan_id = patchset.get("algorithm_plan_id") or patch_plan.get("algorithm_plan_id")
        out.append({
            "round_idx": ev.get("round_idx"),
            "parent_version_id": patchset.get("parent_version_id") or patchset.get("parent_patchset_id"),
            "patchset_id": patchset.get("patchset_id"),
            "algorithm_plan": algorithm_plans.get(algorithm_plan_id),
            "implementation_trial": trials.get(patchset.get("implementation_trial_id")),
            "patch_plan": patch_plan or None,
            "patchset": patchset or None,
            "evaluation": ev,
        })
        if len(out) >= max_recent:
            break
    return out
