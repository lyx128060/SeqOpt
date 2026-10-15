"""Immutable record models for the deterministic seq_align C-patch loop."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ContractSnapshot(BaseModel):
    """Serializable copy of a C component contract used by a proposal."""

    name: str
    kind: str
    c_target_file: str
    c_target_functions: list[str] = Field(default_factory=list)
    description: str = ""
    contract_hash: str | None = None


class SourceExcerpt(BaseModel):
    """Function-local source included in a context pack."""

    component_name: str
    file: str
    function: str
    start_line: int
    end_line: int
    source_hash: str
    text: str


class EvalSummary(BaseModel):
    """Compact metrics embedded in context and decision records."""

    eval_id: str | None = None
    patchset_id: str | None = None
    stage: str | None = None
    status: str | None = None
    tp: int = 0
    fp: int = 0
    tp_fp_ratio: float = 0.0
    cost_s: float = 0.0
    n_seqs: int = 0
    iterations: int = 0
    errors: list[str] = Field(default_factory=list)
    # The conditions this ``cost_s`` was measured under (backend, partition,
    # actual per-query cores, job concurrency, stage, exact query set, max_iter,
    # database paths+hashes). A ``cost_s`` without one cannot be compared to
    # anything: an old record with no fingerprint predates the comparability gate
    # and must be treated as unverifiable rather than as a like-for-like number.
    # See ``comparability.py``.
    comparability: dict[str, Any] | None = None


class ContextPackRecord(BaseModel):
    """Model-facing package for one C-patch proposal round."""

    schema_version: str = "1.0"
    context_pack_id: str | None = None
    round_idx: int
    run_dir: str
    objective: dict[str, Any]
    parent_patchset_id: str | None = None
    current_best_patchset_id: str | None = None
    baseline: EvalSummary | None = None
    current_best: EvalSummary | None = None
    allowed_c_contracts: list[ContractSnapshot]
    source_excerpts: list[SourceExcerpt]
    original_algorithm_context: dict[str, str] = Field(default_factory=dict)
    profile_evidence: dict[str, Any] = Field(default_factory=dict)
    recent_history: list[dict[str, Any]] = Field(default_factory=list)
    recent_negative_experience: list[dict[str, Any]] = Field(default_factory=list)
    compressed_experience: str = ""
    # Hierarchical context-compression (HCC) layers.  The legacy fields above
    # remain available for old run artifacts; prompt builders consume these
    # explicit layers so each agent sees only the memory relevant to its role.
    l1_raw_working_memory: dict[str, Any] = Field(default_factory=dict)
    l2_phase_summaries: list["PhaseSummaryRecord"] = Field(default_factory=list)
    l3_durable_wisdom: list["DurableWisdomRecord"] = Field(default_factory=list)
    application_base_policy: str = ""
    context_window_tokens: int = 256_000
    required_output_schema: dict[str, Any] = Field(default_factory=dict)
    hard_constraints: list[str] = Field(default_factory=list)
    created_at: str | None = None


class AlgorithmPlanProposal(BaseModel):
    """High-level outer-loop algorithm proposal.

    It deliberately has no source-code or diff payload.  Source binding is
    recorded by the inner implementation plan so an outer proposal can remain
    an algorithmic hypothesis rather than a disguised patch.
    """

    model_config = ConfigDict(extra="forbid")

    algorithm_family: str
    baseline_algorithm_summary: str
    proposed_algorithm_change: str
    speed_mechanism: str
    quality_preservation_argument: str
    target_contracts: list[str] = Field(default_factory=list)
    target_functions: list[str] = Field(default_factory=list)
    exact_fallback_strategy: str = ""
    predicted_observables: list[str] = Field(default_factory=list)
    profile_evidence_citation: str = ""
    profile_evidence_refs: list[str] = Field(default_factory=list)
    novelty_against_l2_l3: str = ""
    implementation_constraints: list[str] = Field(default_factory=list)
    risk_notes: str = ""
    related_experience_ids: list[str] = Field(default_factory=list)


class AlgorithmPlanRecord(BaseModel):
    """Content-addressed, validated outer-loop algorithm plan."""

    schema_version: str = "2.0"
    algorithm_plan_id: str | None = None
    round_idx: int
    context_pack_id: str | None = None
    comparison_patchset_id: str | None = None
    parent_algorithm_plan_id: str | None = None
    parent_version_id: str | None = None
    algorithm_family: str
    baseline_algorithm_summary: str
    proposed_algorithm_change: str
    speed_mechanism: str
    quality_preservation_argument: str
    target_contracts: list[str] = Field(default_factory=list)
    target_functions: list[str] = Field(default_factory=list)
    exact_fallback_strategy: str = ""
    predicted_observables: list[str] = Field(default_factory=list)
    profile_evidence_citation: str = ""
    profile_evidence_refs: list[str] = Field(default_factory=list)
    novelty_against_l2_l3: str = ""
    implementation_constraints: list[str] = Field(default_factory=list)
    risk_notes: str = ""
    related_experience_ids: list[str] = Field(default_factory=list)
    validation_status: Literal["ok", "rejected"] = "ok"
    validation_errors: list[str] = Field(default_factory=list)
    proposal_metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: str | None = None


class PatchPlanProposal(BaseModel):
    """One structured algorithm plan returned before diff generation."""

    model_config = ConfigDict(extra="forbid")

    component_name: str
    target_functions: list[str] = Field(default_factory=list)
    algorithm_hypothesis: str
    intended_code_change: str
    runtime_effective_argument: str
    profile_evidence_citation: str = ""
    profile_evidence_refs: list[str] = Field(default_factory=list)
    target_symbols: list[str] = Field(default_factory=list)
    negative_experience_citation: str = ""
    novelty_argument: str = ""
    expected_metric_effect: str = ""
    risk_notes: str = ""
    anti_degenerate_check: str = ""
    related_experience_ids: list[str] = Field(default_factory=list)


class PatchPlanRecord(BaseModel):
    """Immutable plan artifact linking context to a later C diff."""

    schema_version: str = "1.0"
    patch_plan_id: str | None = None
    parent_patch_plan_id: str | None = None
    algorithm_plan_id: str | None = None
    implementation_trial_id: str | None = None
    parent_version_id: str | None = None
    inner_idx: int | None = None
    round_idx: int
    context_pack_id: str | None = None
    parent_patchset_id: str | None = None
    component_name: str
    c_target_file: str
    c_target_functions: list[str] = Field(default_factory=list)
    algorithm_hypothesis: str
    intended_code_change: str
    runtime_effective_argument: str
    profile_evidence_citation: str = ""
    profile_evidence_refs: list[str] = Field(default_factory=list)
    target_symbols: list[str] = Field(default_factory=list)
    negative_experience_citation: str = ""
    novelty_argument: str = ""
    expected_metric_effect: str = ""
    risk_notes: str = ""
    anti_degenerate_check: str = ""
    related_experience_ids: list[str] = Field(default_factory=list)
    validation_status: Literal["ok", "rejected"] = "ok"
    validation_errors: list[str] = Field(default_factory=list)
    proposal_metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: str | None = None


class ImplementationTrialRecord(BaseModel):
    """One inner-loop implementation attempt bound to an approved algorithm."""

    schema_version: str = "2.0"
    implementation_trial_id: str | None = None
    round_idx: int
    inner_idx: int
    algorithm_plan_id: str
    parent_version_id: str
    context_pack_id: str | None = None
    patch_plan_id: str | None = None
    implementation_approach: str
    source_binding: dict[str, Any] = Field(default_factory=dict)
    interface_preservation_argument: str = ""
    fallback_argument: str = ""
    thread_safety_argument: str = ""
    expected_profile_effect: str = ""
    trajectory_analysis: str = ""
    proposal_metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: str | None = None


class ExactEditBlock(BaseModel):
    """Exact old/new source replacement used to synthesize a unified diff."""

    old: str
    new: str


class PatchProposal(BaseModel):
    """One structured C patch returned by a proposer."""

    model_config = ConfigDict(extra="forbid")

    component_name: str
    rationale: str
    unified_diff: str = ""
    edits: list[ExactEditBlock] = Field(default_factory=list)
    expected_metric_effect: str = ""
    risk_notes: str = ""


class PatchRecord(BaseModel):
    """One diff within a patchset."""

    patch_id: str | None = None
    component_name: str
    c_target_file: str
    c_target_functions: list[str] = Field(default_factory=list)
    unified_diff: str
    edit_synthesis_errors: list[str] = Field(default_factory=list)
    diff_sha256: str | None = None
    rationale: str = ""
    expected_metric_effect: str = ""
    risk_notes: str = ""


class PatchsetRecord(BaseModel):
    """Immutable set of ordered patches evaluated as one candidate."""

    schema_version: str = "1.0"
    patchset_id: str | None = None
    round_idx: int
    parent_patchset_id: str | None = None
    parent_version_id: str | None = None
    algorithm_plan_id: str | None = None
    implementation_trial_id: str | None = None
    patch_plan_id: str | None = None
    base_hmmer_source: str
    source_hashes: dict[str, str] = Field(default_factory=dict)
    contract_snapshot: ContractSnapshot | None = None
    patches: list[PatchRecord] = Field(default_factory=list)
    plan_diff_binding: PlanDiffBindingRecord | None = None
    context_pack_id: str | None = None
    proposal_metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: str | None = None


class EvaluationRecord(BaseModel):
    """Immutable verification/build/evaluation evidence for a patchset."""

    schema_version: str = "1.0"
    eval_id: str | None = None
    patchset_id: str
    round_idx: int
    stage: Literal["static_verify", "smoke", "full"]
    status: str
    decision: str = "pending"
    violations: list[str] = Field(default_factory=list)
    eval_result: dict[str, Any] | None = None
    eval_config: dict[str, Any] = Field(default_factory=dict)
    build: dict[str, Any] = Field(default_factory=dict)
    dataset_manifest: dict[str, Any] = Field(default_factory=dict)
    source_hashes: dict[str, str] = Field(default_factory=dict)
    plan_diff_binding: PlanDiffBindingRecord | None = None
    metric_delta: dict[str, Any] = Field(default_factory=dict)
    errors: list[str] = Field(default_factory=list)
    attempt_ids: list[str] = Field(default_factory=list)
    primary_attempt_id: str | None = None
    evaluation_fingerprint: str | None = None
    query_completion_ids: list[str] = Field(default_factory=list)
    requested_queries: int | None = None
    completed_queries: int | None = None
    resumed_queries: int | None = None
    interruption: dict[str, Any] | None = None
    created_at: str | None = None


class ExperienceRecord(BaseModel):
    """Evidence-linked lesson used by later context packs."""

    schema_version: str = "1.0"
    experience_id: str | None = None
    source_eval_ids: list[str] = Field(default_factory=list)
    source_patchset_ids: list[str] = Field(default_factory=list)
    source_algorithm_plan_ids: list[str] = Field(default_factory=list)
    source_patch_plan_ids: list[str] = Field(default_factory=list)
    component_name: str | None = None
    scope: Literal["outer_algorithm", "inner_implementation", "workflow"] = "inner_implementation"
    polarity: Literal["positive", "negative", "neutral"] = "neutral"
    failure_stage: str | None = None
    failure_class: str | None = None
    outcome: str
    lesson_kind: str = "algorithm"
    severity: str = "info"
    metric_summary: dict[str, Any] = Field(default_factory=dict)
    patch_signature: dict[str, Any] = Field(default_factory=dict)
    lesson: str
    positive_patterns: list[str] = Field(default_factory=list)
    negative_patterns: list[str] = Field(default_factory=list)
    applicability: list[str] = Field(default_factory=list)
    retrieval_tags: list[str] = Field(default_factory=list)
    mechanism_projection: dict[str, Any] = Field(default_factory=dict)
    mechanism_fingerprint: str | None = None
    compression_epoch: int = 0
    supersedes: list[str] = Field(default_factory=list)
    created_at: str | None = None

    @model_validator(mode="after")
    def _derive_legacy_scope(self) -> "ExperienceRecord":
        # Older producers only populated lesson_kind/outcome.  Derive the new
        # dual-track fields without making those historical artifacts invalid.
        workflow_outcomes = {
            "compile_failure", "patch_apply_failure", "runtime_failure",
            "runtime_provenance_failure", "scope_violation",
            "plan_validation_failure", "plan_proposal_error",
            "patch_proposal_error", "repair_proposal_error",
            "fatal_proposal_error",
        }
        if self.lesson_kind == "workflow" or self.outcome in workflow_outcomes:
            self.scope = "workflow"
            if self.polarity == "neutral":
                self.polarity = "negative"
        if not self.mechanism_projection:
            self.mechanism_projection = {
                "track": "workflow" if self.scope == "workflow" else self.scope,
                "component_name": self.component_name,
                "outcome": self.outcome,
            }
        return self


class PhaseSummaryRecord(BaseModel):
    """Compressed, evidence-linked summary of an older optimization phase."""

    schema_version: str = "1.0"
    phase_summary_id: str | None = None
    run_dir: str
    phase_kind: str
    source_eval_ids: list[str] = Field(default_factory=list)
    source_patchset_ids: list[str] = Field(default_factory=list)
    accepted_speedups: list[dict[str, Any]] = Field(default_factory=list)
    rejected_mechanisms: list[dict[str, Any]] = Field(default_factory=list)
    query_observations: list[dict[str, Any]] = Field(default_factory=list)
    summary: str = ""
    created_at: str | None = None


class DurableWisdomRecord(BaseModel):
    """Long-lived compact experience safe to retrieve across runs."""

    schema_version: str = "1.0"
    wisdom_id: str | None = None
    wisdom_kind: Literal["algorithm_wisdom", "implementation_wisdom", "workflow_wisdom"] = "implementation_wisdom"
    source: str = "run"
    confidence: Literal["candidate", "confirmed", "deprecated"] = "candidate"
    algorithm_family: str | None = None
    mechanism: str = ""
    component_name: str | None = None
    target_functions: list[str] = Field(default_factory=list)
    target_symbols: list[str] = Field(default_factory=list)
    lesson: str
    source_eval_ids: list[str] = Field(default_factory=list)
    source_patchset_ids: list[str] = Field(default_factory=list)
    source_patch_plan_ids: list[str] = Field(default_factory=list)
    source_algorithm_plan_ids: list[str] = Field(default_factory=list)
    metric_evidence: dict[str, Any] = Field(default_factory=dict)
    diff_sha256: str | None = None
    retrieval_tags: list[str] = Field(default_factory=list)
    mechanism_projection: dict[str, Any] = Field(default_factory=dict)
    mechanism_fingerprint: str | None = None
    supersedes: list[str] = Field(default_factory=list)
    created_at: str | None = None


class OuterAlgorithmMemoryView(BaseModel):
    """Memory exposed to the high-level outer algorithm proposer."""

    objective: dict[str, Any] = Field(default_factory=dict)
    baseline: dict[str, Any] | None = None
    current_best: dict[str, Any] | None = None
    profile_evidence: dict[str, Any] = Field(default_factory=dict)
    raw_current_trajectory: list[dict[str, Any]] = Field(default_factory=list)
    algorithm_experience: list[dict[str, Any]] = Field(default_factory=list)
    compressed_older_experience: str = ""


class InnerImplementationMemoryView(BaseModel):
    """Memory exposed to implementation planning and patch generation."""

    objective: dict[str, Any] = Field(default_factory=dict)
    approved_algorithm_plan: dict[str, Any] | None = None
    parent_version_id: str | None = None
    allowed_c_contracts: list[dict[str, Any]] = Field(default_factory=list)
    source_excerpts: list[dict[str, Any]] = Field(default_factory=list)
    profile_evidence: dict[str, Any] = Field(default_factory=dict)
    raw_current_trajectory: list[dict[str, Any]] = Field(default_factory=list)
    implementation_experience: list[dict[str, Any]] = Field(default_factory=list)
    compressed_older_experience: str = ""
    application_base_policy: str = ""


class PatchRepairMemoryView(BaseModel):
    """Minimal memory for repairing the current deterministic failure."""

    approved_algorithm_plan: dict[str, Any] | None = None
    current_validation_failure: dict[str, Any] = Field(default_factory=dict)
    binding_delta: dict[str, Any] = Field(default_factory=dict)
    build_provenance_delta: dict[str, Any] = Field(default_factory=dict)
    repair_experience: list[dict[str, Any]] = Field(default_factory=list)


class DiffHunkFootprint(BaseModel):
    schema_version: str = "1.0"
    old_start: int
    old_count: int
    new_start: int
    new_count: int
    changed_old_lines: list[int] = Field(default_factory=list)
    changed_new_lines: list[int] = Field(default_factory=list)
    pre_functions: list[str] = Field(default_factory=list)
    post_functions: list[str] = Field(default_factory=list)


class ActualDiffFootprint(BaseModel):
    schema_version: str = "1.0"
    extraction_status: str
    target_file: str | None = None
    hunks: list[DiffHunkFootprint] = Field(default_factory=list)
    pre_functions: list[str] = Field(default_factory=list)
    post_functions: list[str] = Field(default_factory=list)
    actual_functions: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)


class PlanDiffBindingRecord(BaseModel):
    schema_version: str = "1.0"
    binding_status: str
    final_component_name: str | None = None
    final_allowed_file: str | None = None
    final_allowed_functions: list[str] = Field(default_factory=list)
    actual_footprint: ActualDiffFootprint | None = None
    containment: dict[str, bool] = Field(default_factory=dict)
    errors: list[str] = Field(default_factory=list)


class EvaluationAttemptRecord(BaseModel):
    schema_version: str = "1.0"
    attempt_id: str | None = None
    invocation_id: str
    patchset_id: str
    round_idx: int
    stage: str
    attempt_kind: str = "primary"
    evaluation_fingerprint: str
    status: str
    parent_attempt_id: str | None = None
    requested_queries: int = 0
    completed_queries: int = 0
    resumed_queries: int = 0
    query_completion_ids: list[str] = Field(default_factory=list)
    binary_sha256: dict[str, str] = Field(default_factory=dict)
    timing_protocol_fingerprint: str | None = None
    slurm_job_ids: list[str] = Field(default_factory=list)
    aggregate_result: dict[str, Any] | None = None
    interruption: dict[str, Any] | None = None
    errors: list[str] = Field(default_factory=list)
    created_at: str | None = None


class QueryCompletionRecord(BaseModel):
    schema_version: str = "1.0"
    query_completion_id: str | None = None
    attempt_id: str
    invocation_id: str
    evaluation_fingerprint: str
    query_fingerprint: str
    patchset_id: str
    stage: str
    attempt_kind: str
    query_idx: int
    query_id: str
    query_path: str
    query_sha256: str
    binary_sha256: dict[str, str] = Field(default_factory=dict)
    input_sha256: dict[str, str] = Field(default_factory=dict)
    status: str
    tp: int = 0
    fp: int = 0
    cost_s: float = 0.0
    iterations: int = 0
    artifacts: dict[str, Any] = Field(default_factory=dict)
    errors: list[str] = Field(default_factory=list)
    created_at: str | None = None


class InvocationRecord(BaseModel):
    schema_version: str = "1.0"
    invocation_id: str
    configuration_id: str
    configuration: dict[str, Any]
    requested_outer: int
    inner_budget: int
    created_at: str | None = None


class InvocationSummaryRecord(BaseModel):
    schema_version: str = "1.0"
    invocation_summary_id: str | None = None
    invocation_id: str
    parent_invocation_summary_id: str | None = None
    episode_idx: int = 1
    configuration_id: str
    requested_outer: int
    inner_budget: int
    completed_slots: int = 0
    blocked_slots: int = 0
    terminal_state: str
    high_water_mark: str
    proposal_attempts: int = 0
    valid_plans: int = 0
    evaluated_candidates: int = 0
    created_at: str | None = None


class SlotEventRecord(BaseModel):
    schema_version: str = "1.0"
    slot_event_id: str | None = None
    slot_id: str
    invocation_id: str
    configuration_id: str
    outer_idx: int
    inner_idx: int | None = None
    parent_slot_event_id: str | None = None
    state: str
    algorithm_plan_id: str | None = None
    patch_plan_id: str | None = None
    patchset_id: str | None = None
    eval_id: str | None = None
    details: dict[str, Any] = Field(default_factory=dict)
    result: dict[str, Any] = Field(default_factory=dict)
    created_at: str | None = None


class RunSummaryRecord(BaseModel):
    """Derived run projection; canonical records remain the source of truth."""

    schema_version: str = "1.0"
    run_summary_id: str | None = None
    high_water_mark: str
    requested: int = 0
    evaluated: int = 0
    proposal_attempts: int = 0
    valid_plans: int = 0
    evaluated_candidates: int = 0
    blocked_slots: int = 0
    terminal_state: str
    invocation_ids: list[str] = Field(default_factory=list)
    results: list[dict[str, Any]] = Field(default_factory=list)
    created_at: str | None = None


class RunMigrationRecord(BaseModel):
    schema_version: str = "1.0"
    migration_id: str | None = None
    source_run_id: str
    source_run_path: str
    target_run_id: str
    target_run_path: str
    migration_reason: str
    ruleset_version: str
    status: str
    patchset_ids: list[str] = Field(default_factory=list)
    evaluation_ids: list[str] = Field(default_factory=list)
    source_artifact_hashes: dict[str, str] = Field(default_factory=dict)
    source_round_lineage: dict[str, int] = Field(default_factory=dict)
    created_at: str | None = None
