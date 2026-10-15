"""Deterministic verification/evaluation runner for one C-patch proposal."""

from __future__ import annotations

import difflib
import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from statistics import median
from typing import Any

from ..assembler import AssembledConfig, assemble, extract_actual_diff_footprint
from ..components.base import ComponentModification
from ..components.c_kernels import ALL_C_KERNEL_CONTRACTS
from ..eval_schema import EvalResult
from ..pending_mods import round_dir, write_accepted, write_pending
from ..run_hmmer_cli import DEFAULT_HMMER_SRC_ROOT, _normalize_patch_paths
from .. import run_hmmer_cli, run_jack
from .experience import (
    algorithm_mechanism_projection,
    append_and_compress,
    experience_from_evaluation,
    implementation_mechanism_projection,
    mechanism_fingerprint,
    workflow_experience,
)
from .models import (
    AlgorithmPlanProposal,
    AlgorithmPlanRecord,
    ContextPackRecord,
    ContractSnapshot,
    EvalSummary,
    EvaluationAttemptRecord,
    EvaluationRecord,
    ExactEditBlock,
    ImplementationTrialRecord,
    PatchPlanProposal,
    PatchPlanRecord,
    PatchProposal,
    PatchRecord,
    PatchsetRecord,
    PlanDiffBindingRecord,
    QueryCompletionRecord,
)
from .store import (
    LoopStore,
    canonical_json,
    hash_file,
    load_json,
    record_hash,
    sha256_text,
    utc_now,
    write_json,
)


logger = logging.getLogger(__name__)


class RuntimeIdentityUnavailable(RuntimeError):
    """The configured evaluation runtime could not be identified safely."""


class PatchRunner:
    """Run static verification, smoke/full eval, and provenance recording."""

    def __init__(
        self,
        store: LoopStore,
        *,
        hmmer_src_root: str | Path | None = None,
        data_dir: str | Path | None = None,
        astral_db: str | Path | None = None,
        uniref_db: str | Path | None = None,
        scop_txt: str | Path | None = None,
        cpus: int = 1,
        max_iter: int = 1,
        full_max_iter: int | None = None,
        accept_min_ratio_delta: float = 0.0,
        accept_min_tp_fraction: float = 1.0,
        min_speedup_fraction: float = 0.03,
        eval_repeats: int = 1,
        paired_current_best_repeats: int = 0,
        near_speedup_band_fraction: float = 0.02,
        require_algorithmic_patch: bool = True,
        external_profiler: str = "off",
        perf_top_queries: int = 0,
        perf_top_symbols: int = 20,
        perf_frequency: int = 99,
        perf_command: str = "dominant",
        query_timeout: float | None = None,
        eval_backend: str | None = None,
        posthoc_profile: str = "off",
        screen_inner_trials: bool = False,
        screen_min_speedup_fraction: float = 0.5,
        screen_reject_speedup_fraction: float = 0.0,
        screen_queries: int = 5,
        screen_eval_repeats: int = 1,
    ):
        self.store = store
        configured_root = Path(hmmer_src_root or DEFAULT_HMMER_SRC_ROOT).resolve()
        # The CLI default historically points at hmmer-3.4/src, but the
        # checkout root is required for isolated copies because Easel lives
        # beside src/ (hmmer-3.4/easel) and configure is at the top level.
        if configured_root.name == "src" and (configured_root.parent / "configure").is_file():
            configured_root = configured_root.parent
        self.hmmer_src_root = configured_root
        self.data_dir = Path(data_dir) if data_dir is not None else run_jack.DIR_DATA_TEST
        self.astral_db = Path(astral_db or run_jack.ASTRAL_DB)
        self.uniref_db = Path(uniref_db or run_jack.UNIREF_DB)
        self.scop_txt = Path(scop_txt or run_jack.SCOP_TXT)
        self.cpus = cpus
        self.max_iter = max_iter
        self.full_max_iter = full_max_iter or max_iter
        self.accept_min_ratio_delta = accept_min_ratio_delta
        self.accept_min_tp_fraction = accept_min_tp_fraction
        self.min_speedup_fraction = min_speedup_fraction
        self.eval_repeats = max(1, eval_repeats)
        self.paired_current_best_repeats = max(0, paired_current_best_repeats)
        self.near_speedup_band_fraction = max(0.0, near_speedup_band_fraction)
        self.require_algorithmic_patch = require_algorithmic_patch
        self.external_profiler = external_profiler
        self.perf_top_queries = max(0, perf_top_queries)
        self.perf_top_symbols = max(1, perf_top_symbols)
        self.perf_frequency = max(1, perf_frequency)
        self.perf_command = perf_command
        self.query_timeout = float(query_timeout) if query_timeout is not None else None
        if self.query_timeout is not None and self.query_timeout <= 0:
            raise ValueError("query_timeout must be positive")
        self.eval_backend = (eval_backend or os.getenv("SEQ_ALIGN_EVAL_BACKEND", "local")).lower()
        if self.eval_backend not in {"local", "slurm"}:
            raise ValueError("eval_backend must be 'local' or 'slurm'")
        # Post-hoc profiling is a *separate* step, not a per-eval configuration.
        # ``external_profiler``/``perf_*`` above are kept only because the profiling
        # step may also ask for a symbol-level (gprof/perf) pass; they no longer
        # reach run_all_queries for the ordinary evaluation, so no trial pays for
        # them. See run_posthoc_profile().
        self.posthoc_profile = (posthoc_profile or "off").lower()
        if self.posthoc_profile not in {"off", "auto", "local", "slurm"}:
            raise ValueError("posthoc_profile must be 'off', 'auto', 'local' or 'slurm'")
        # Inner-trial screening. The threshold is zero on this path (the trial only
        # ranks candidates for the outer commit gate), so paying the full
        # repeats+paired measurement per trial buys precision nothing consumes.
        # Instead every trial is screened on a small query subset with a single
        # measurement; only candidates that could plausibly clear the commit gate
        # earn the full 24h-class measurement. See screen_candidate().
        self.screen_inner_trials = bool(screen_inner_trials)
        self.screen_min_speedup_fraction = max(0.0, float(screen_min_speedup_fraction))
        # 0 = the inner screen only ORDERS candidates and never discards one. A
        # positive value is a WIDE-BAND hard reject, to be set from the measured
        # zero-effect distribution: the screen runs on a tiny separate database and
        # a handful of queries, so its noise is far larger than the commit gate and
        # a near-threshold reject there discards real candidates on noise alone.
        self.screen_reject_speedup_fraction = max(0.0, float(screen_reject_speedup_fraction))
        self.screen_queries = max(1, int(screen_queries))
        self.screen_eval_repeats = max(1, int(screen_eval_repeats))
        self._query_manifest_cache: dict[str, list[dict[str, Any]]] = {}

    def build_algorithm_plan(
        self,
        proposal: AlgorithmPlanProposal,
        context_pack: ContextPackRecord,
        proposal_metadata: dict[str, Any],
        *,
        parent_version_id: str | None = None,
        parent_algorithm_plan_id: str | None = None,
    ) -> AlgorithmPlanRecord:
        """Bind one high-level outer proposal to exactly one algorithm island."""
        target_contracts, target_functions, errors = _algorithm_plan_binding(proposal)
        combined = "\n".join(
            (
                proposal.baseline_algorithm_summary,
                proposal.proposed_algorithm_change,
                proposal.speed_mechanism,
                proposal.quality_preservation_argument,
                proposal.exact_fallback_strategy,
            )
        ).lower()
        if not proposal.proposed_algorithm_change.strip() or not proposal.speed_mechanism.strip():
            errors.append("algorithm plan must describe a non-empty mechanism and speed argument")
        if not proposal.quality_preservation_argument.strip():
            errors.append("algorithm plan must explain TP/FP preservation")
        if _is_parameter_only_plan_text(combined):
            errors.append("outer algorithm plan is parameter tuning rather than an algorithmic mechanism")
        if any(token in combined for token in ("unified diff", "@@ -", "python_payload")):
            errors.append("outer algorithm plan must not contain implementation or diff payloads")
        if _is_implementation_only_algorithm_plan(proposal, combined):
            errors.append("outer algorithm plan is implementation-only rather than a high-level HMMER algorithm change")
        if not _has_high_level_decision_surface(combined):
            errors.append("outer algorithm plan must name a high-level HMMER decision surface")
        errors.extend(_memory_novelty_errors(proposal, context_pack, track="outer_algorithm"))
        errors.extend(
            _negative_algorithm_experience_errors(
                proposal,
                context_pack.recent_negative_experience,
            )
        )
        errors.extend(
            _related_experience_errors(
                proposal,
                self.store,
                expected_scope="outer_algorithm",
            )
        )
        record = AlgorithmPlanRecord(
            round_idx=context_pack.round_idx,
            context_pack_id=context_pack.context_pack_id,
            comparison_patchset_id=parent_version_id,
            parent_version_id=parent_version_id,
            parent_algorithm_plan_id=parent_algorithm_plan_id,
            algorithm_family=proposal.algorithm_family,
            baseline_algorithm_summary=proposal.baseline_algorithm_summary,
            proposed_algorithm_change=proposal.proposed_algorithm_change,
            speed_mechanism=proposal.speed_mechanism,
            quality_preservation_argument=proposal.quality_preservation_argument,
            target_contracts=target_contracts,
            target_functions=target_functions,
            exact_fallback_strategy=proposal.exact_fallback_strategy,
            predicted_observables=list(proposal.predicted_observables),
            profile_evidence_citation=proposal.profile_evidence_citation,
            profile_evidence_refs=list(proposal.profile_evidence_refs),
            novelty_against_l2_l3=proposal.novelty_against_l2_l3,
            implementation_constraints=list(proposal.implementation_constraints),
            risk_notes=proposal.risk_notes,
            related_experience_ids=list(proposal.related_experience_ids),
            validation_status="ok" if not errors else "rejected",
            validation_errors=errors,
            proposal_metadata=proposal_metadata,
            created_at=utc_now(),
        )
        record.algorithm_plan_id = record_hash(record, id_field="algorithm_plan_id")
        self.store.put_algorithm_plan(record)
        return record

    def build_implementation_trial(
        self,
        *,
        round_idx: int,
        inner_idx: int,
        algorithm_plan: AlgorithmPlanRecord,
        parent_version_id: str,
        context_pack_id: str | None,
        patch_plan_id: str | None,
        proposal_metadata: dict[str, Any],
        implementation_approach: str,
        trajectory_analysis: str = "",
        interface_preservation_argument: str = "",
        fallback_argument: str = "",
        thread_safety_argument: str = "",
        expected_profile_effect: str = "",
    ) -> ImplementationTrialRecord:
        contract_name = algorithm_plan.target_contracts[0]
        contract = ALL_C_KERNEL_CONTRACTS[contract_name]
        record = ImplementationTrialRecord(
            round_idx=round_idx,
            inner_idx=inner_idx,
            algorithm_plan_id=algorithm_plan.algorithm_plan_id or "",
            parent_version_id=parent_version_id,
            context_pack_id=context_pack_id,
            patch_plan_id=patch_plan_id,
            implementation_approach=implementation_approach,
            source_binding={
                "component_name": contract_name,
                "target_file": contract.c_target_file,
                "frozen_exports": list(contract.c_target_functions or []),
            },
            interface_preservation_argument=interface_preservation_argument,
            fallback_argument=fallback_argument,
            thread_safety_argument=thread_safety_argument,
            expected_profile_effect=expected_profile_effect,
            trajectory_analysis=trajectory_analysis,
            proposal_metadata=proposal_metadata,
            created_at=utc_now(),
        )
        record.implementation_trial_id = record_hash(
            record, id_field="implementation_trial_id"
        )
        self.store.put_implementation_trial(record)
        return record

    def build_patch_plan(
        self,
        proposal: PatchPlanProposal,
        context_pack: ContextPackRecord,
        proposal_metadata: dict[str, Any],
        *,
        parent_patch_plan_id: str | None = None,
        parent_patchset_id: str | None = None,
        algorithm_plan: AlgorithmPlanRecord | None = None,
        implementation_trial_id: str | None = None,
        inner_idx: int | None = None,
    ) -> PatchPlanRecord:
        """Validate and store the model's first-stage algorithm plan."""
        errors = validate_patch_plan(
            proposal,
            profile_evidence=context_pack.profile_evidence,
            recent_negative_experience=context_pack.recent_negative_experience,
        )
        errors.extend(_memory_novelty_errors(proposal, context_pack, track="inner_implementation"))
        errors.extend(
            _related_experience_errors(
                proposal,
                self.store,
                expected_scope="inner_implementation",
            )
        )
        if algorithm_plan is not None:
            if algorithm_plan.validation_status != "ok":
                errors.append("inner plan references a rejected outer algorithm plan")
            if proposal.component_name not in algorithm_plan.target_contracts:
                errors.append(
                    f"component {proposal.component_name!r} is not in approved algorithm plan target_contracts"
                )
            if not set(proposal.target_functions or []) <= set(algorithm_plan.target_functions):
                errors.append("inner plan targets functions outside approved algorithm plan target_functions")
        contract = ALL_C_KERNEL_CONTRACTS.get(proposal.component_name)
        target_file = contract.c_target_file if contract else ""
        target_functions = list(proposal.target_functions or (contract.c_target_functions if contract else []))
        record = PatchPlanRecord(
            parent_patch_plan_id=parent_patch_plan_id,
            algorithm_plan_id=algorithm_plan.algorithm_plan_id if algorithm_plan else None,
            implementation_trial_id=implementation_trial_id,
            parent_version_id=parent_patchset_id,
            inner_idx=inner_idx,
            round_idx=context_pack.round_idx,
            context_pack_id=context_pack.context_pack_id,
            parent_patchset_id=parent_patchset_id,
            component_name=proposal.component_name,
            c_target_file=target_file or "",
            c_target_functions=target_functions,
            algorithm_hypothesis=proposal.algorithm_hypothesis,
            intended_code_change=proposal.intended_code_change,
            runtime_effective_argument=proposal.runtime_effective_argument,
            profile_evidence_citation=proposal.profile_evidence_citation,
            profile_evidence_refs=list(proposal.profile_evidence_refs),
            target_symbols=list(proposal.target_symbols),
            negative_experience_citation=proposal.negative_experience_citation,
            novelty_argument=proposal.novelty_argument,
            expected_metric_effect=proposal.expected_metric_effect,
            risk_notes=proposal.risk_notes,
            anti_degenerate_check=proposal.anti_degenerate_check,
            related_experience_ids=list(proposal.related_experience_ids),
            validation_status="ok" if not errors else "rejected",
            validation_errors=errors,
            proposal_metadata=proposal_metadata,
            created_at=utc_now(),
        )
        record.patch_plan_id = record_hash(record, id_field="patch_plan_id")
        self.store.put_patch_plan(record)
        if record.validation_status == "rejected":
            append_and_compress(self.store, workflow_experience(
                outcome="plan_validation_failure",
                lesson=(
                    f"Patch plan failed validation before C diff generation: component={record.component_name}, "
                    f"target_functions={record.c_target_functions}, errors={record.validation_errors}."
                ),
                severity="medium",
                component_name=record.component_name,
                metric_summary={
                    "round_idx": record.round_idx,
                    "context_pack_id": record.context_pack_id,
                    "patch_plan_id": record.patch_plan_id,
                    "validation_errors": list(record.validation_errors),
                },
                patch_signature={
                    "component_name": record.component_name,
                    "target_file": record.c_target_file,
                    "target_functions": list(record.c_target_functions),
                    "patch_plan_id": record.patch_plan_id,
                    "round_idx": record.round_idx,
                },
                negative_patterns=["plan validation failed", *record.validation_errors],
                retrieval_tags=["plan_validation", record.component_name],
            ))
        return record

    def build_patchset(
        self,
        proposal: PatchProposal,
        context_pack: ContextPackRecord,
        proposal_metadata: dict[str, Any],
        *,
        parent_patchset_id: str | None = None,
        patch_plan: PatchPlanRecord | None = None,
        algorithm_plan_id: str | None = None,
        algorithm_plan: AlgorithmPlanRecord | None = None,
        implementation_trial_id: str | None = None,
        parent_version_id: str | None = None,
    ) -> PatchsetRecord:
        """Convert an incremental proposal into a reproducible squashed patchset.

        Model proposals are authored against the current incumbent source.  Runtime
        evaluation, however, always starts from the immutable stock checkout.  Keep
        both properties by applying the incremental diff to the materialized parent
        and storing one stock-relative diff per component in the child patchset.
        """
        contract = ALL_C_KERNEL_CONTRACTS[proposal.component_name]
        approved_algorithm = algorithm_plan
        resolved_algorithm_plan_id = algorithm_plan_id
        if patch_plan is not None and patch_plan.algorithm_plan_id:
            if resolved_algorithm_plan_id and resolved_algorithm_plan_id != patch_plan.algorithm_plan_id:
                resolved_algorithm_plan_id = algorithm_plan_id
            else:
                resolved_algorithm_plan_id = patch_plan.algorithm_plan_id
        if approved_algorithm is None and resolved_algorithm_plan_id:
            try:
                approved_algorithm = self.store.get_algorithm_plan(resolved_algorithm_plan_id)
            except (FileNotFoundError, ValueError):
                approved_algorithm = None
        if approved_algorithm is not None and not resolved_algorithm_plan_id:
            resolved_algorithm_plan_id = approved_algorithm.algorithm_plan_id
        snapshot = ContractSnapshot(
            name=contract.name,
            kind=str(contract.kind),
            c_target_file=contract.c_target_file or "",
            c_target_functions=list(contract.c_target_functions or []),
            description=contract.description,
        )
        snapshot.contract_hash = record_hash(snapshot, id_field="contract_hash")
        parent_root = self.materialize_patchset(parent_patchset_id)
        incremental_diff, edit_errors = self._proposal_diff(
            proposal,
            contract.c_target_file or "",
            source_root=parent_root,
        )
        plan_diff_binding = (
            _plan_diff_binding(
                proposal=proposal,
                contract=contract,
                incremental_diff=incremental_diff,
                source_root=parent_root,
                patch_plan=patch_plan,
                algorithm_plan=approved_algorithm,
                requested_algorithm_plan_id=resolved_algorithm_plan_id,
            )
            if not edit_errors and incremental_diff.strip()
            else None
        )
        unified_diff = incremental_diff
        final_root: Path | None = None
        if not edit_errors:
            final_root, apply_errors = self._apply_incremental_source(
                parent_root=parent_root,
                parent_patchset_id=parent_patchset_id,
                component_name=proposal.component_name,
                target_file=contract.c_target_file or "",
                incremental_diff=incremental_diff,
            )
            edit_errors.extend(apply_errors)
            if final_root is not None and not edit_errors:
                unified_diff = _stock_relative_diff(
                    self.hmmer_src_root,
                    final_root,
                    contract.c_target_file or "",
                )
                if not unified_diff.strip():
                    edit_errors.append("incremental patch produced no stock-relative source change")
        patch = PatchRecord(
            component_name=proposal.component_name,
            c_target_file=contract.c_target_file or "",
            c_target_functions=list(contract.c_target_functions or []),
            unified_diff=unified_diff,
            edit_synthesis_errors=edit_errors,
            rationale=proposal.rationale,
            expected_metric_effect=proposal.expected_metric_effect,
            risk_notes=proposal.risk_notes,
        )
        patch.diff_sha256 = sha256_text(unified_diff)
        patch.patch_id = record_hash(patch, id_field="patch_id")
        inherited: list[PatchRecord] = []
        if parent_patchset_id:
            parent = self.store.get_patchset(parent_patchset_id)
            inherited = [
                item.model_copy(deep=True)
                for item in parent.patches
                if item.component_name != proposal.component_name
            ]
        patches = [*inherited, patch]
        patches.sort(key=lambda item: (item.component_name, item.c_target_file, item.patch_id or ""))
        source_hashes = self._source_hashes_for_root(final_root or parent_root, patches)
        patchset = PatchsetRecord(
            round_idx=context_pack.round_idx,
            parent_patchset_id=parent_patchset_id,
            parent_version_id=parent_version_id or parent_patchset_id,
            algorithm_plan_id=resolved_algorithm_plan_id,
            implementation_trial_id=implementation_trial_id,
            patch_plan_id=patch_plan.patch_plan_id if patch_plan else None,
            base_hmmer_source=str(self.hmmer_src_root),
            source_hashes=source_hashes,
            contract_snapshot=snapshot,
            patches=patches,
            plan_diff_binding=plan_diff_binding,
            context_pack_id=context_pack.context_pack_id,
            proposal_metadata=proposal_metadata,
            created_at=utc_now(),
        )
        patchset.patchset_id = record_hash(patchset, id_field="patchset_id")
        self.store.put_patchset(patchset)
        self._write_legacy_pending(patchset)
        return patchset

    def materialize_patchset(self, patchset_id: str | None) -> Path:
        """Return an immutable source tree containing a stored patchset.

        The empty/stock parent is represented by the repository source root.  A
        content-addressed child is copied and patched once, then guarded by a marker
        containing the patchset ID so controller restarts reuse the exact same tree.
        """
        if not patchset_id:
            return self.hmmer_src_root
        patchset = self.store.get_patchset(patchset_id)
        destination = self.store.work_dir / "materialized_sources" / patchset_id / "hmmer-src"
        marker = destination.parent / "complete.json"
        if marker.is_file() and destination.is_dir():
            try:
                data = json.loads(marker.read_text(encoding="utf-8"))
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                data = {}
            if data.get("patchset_id") == patchset_id:
                return destination

        if destination.parent.exists():
            shutil.rmtree(destination.parent)
        destination.parent.mkdir(parents=True, exist_ok=True)
        run_hmmer_cli._copy_source(self.hmmer_src_root, destination)
        if patchset.patches:
            errors = _apply_patch_records(destination, patchset.patches)
            if errors:
                shutil.rmtree(destination.parent, ignore_errors=True)
                raise ValueError(
                    f"cannot materialize parent patchset {patchset_id}: " + "; ".join(errors)
                )
        marker.write_text(
            json.dumps({"patchset_id": patchset_id}, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return destination

    def _apply_incremental_source(
        self,
        *,
        parent_root: Path,
        parent_patchset_id: str | None,
        component_name: str,
        target_file: str,
        incremental_diff: str,
    ) -> tuple[Path | None, list[str]]:
        key = sha256_text(
            json.dumps(
                {
                    "parent": parent_patchset_id or "stock",
                    "component": component_name,
                    "target_file": target_file,
                    "diff": incremental_diff,
                },
                sort_keys=True,
            )
        )
        root = self.store.work_dir / "candidate_sources" / key / "hmmer-src"
        if root.parent.exists():
            shutil.rmtree(root.parent)
        root.parent.mkdir(parents=True, exist_ok=True)
        run_hmmer_cli._copy_source(parent_root, root)
        patch = PatchRecord(
            component_name=component_name,
            c_target_file=target_file,
            unified_diff=incremental_diff,
        )
        errors = _apply_patch_records(root, [patch])
        if errors:
            return None, errors
        return root, []

    def static_verify(self, patchset: PatchsetRecord) -> tuple[EvaluationRecord, Any | None]:
        edit_errors = [error for patch in patchset.patches for error in patch.edit_synthesis_errors]
        binding_errors = (
            list(patchset.plan_diff_binding.errors)
            if patchset.plan_diff_binding is not None
            and patchset.plan_diff_binding.binding_status != "ok"
            else []
        )
        if binding_errors or edit_errors:
            cfg = AssembledConfig()
            algorithmic_errors: list[str] = []
            git_apply_errors: list[str] = []
        else:
            mods = _mods_from_patchset(patchset)
            cfg = assemble(mods, hmmer_src_root=self.hmmer_src_root)
            algorithmic_errors = [] if cfg.errors else self._algorithmic_patch_check(patchset)
            git_apply_errors = [] if cfg.errors or algorithmic_errors else self._git_apply_check(patchset)
        errors = binding_errors + edit_errors + list(cfg.errors) + algorithmic_errors + git_apply_errors
        status = "ok" if not errors else "compile_error"
        if status == "ok":
            decision = "needs_smoke"
        elif binding_errors:
            decision = "rejected_plan_diff_binding"
        elif algorithmic_errors:
            decision = "rejected_parameter_tuning"
        elif edit_errors or git_apply_errors:
            decision = "rejected_patch_apply"
        else:
            decision = "rejected_static"
        record = EvaluationRecord(
            patchset_id=patchset.patchset_id or "",
            round_idx=patchset.round_idx,
            stage="static_verify",
            status=status,
            decision=decision,
            violations=errors,
            eval_config=self._eval_config(stage="static_verify"),
            source_hashes=patchset.source_hashes,
            plan_diff_binding=patchset.plan_diff_binding,
            errors=errors,
            created_at=utc_now(),
        )
        record.eval_id = record_hash(record, id_field="eval_id")
        self.store.put_evaluation(record)
        if status != "ok":
            append_and_compress(self.store, experience_from_evaluation(patchset, record))
        return record, cfg if status == "ok" else None

    def screen_candidate(
        self,
        patchset: PatchsetRecord,
        cfg: Any,
        *,
        current_best: EvalSummary | None,
        quality_reference: EvalSummary | None,
        keep_work: bool = False,
        query_timeout_reference: str | Path | None = None,
    ) -> EvaluationRecord:
        """Run the cheap pre-screen for one inner trial.

        One single-measurement evaluation on the small query subset (``smoke``
        stage, ``screen_eval_repeats`` repeats) instead of the full
        repeats+paired measurement. The inner threshold is zero -- the trial only
        ranks candidates for the outer commit gate -- so the screen exists to
        *reject obviously uncompetitive candidates cheaply*, not to accept anyone:
        a candidate that clears it still runs the full measurement before any
        commit decision. Quality is enforced against the smoke reference, so the
        screen can never wave through a recall regression.
        """
        previous_repeats = self.eval_repeats
        self.eval_repeats = self.screen_eval_repeats
        try:
            record = self.evaluate(
                patchset,
                cfg,
                stage="smoke",
                current_best=current_best,
                quality_reference=quality_reference,
                keep_work=keep_work,
                commit_indexes=False,
                write_legacy_result=False,
                query_timeout_reference=query_timeout_reference,
                attempt_kind="screen",
            )
        finally:
            self.eval_repeats = previous_repeats
        meta = dict(record.eval_result or {})
        meta.setdefault("meta", {})
        meta["meta"] = {
            **(meta.get("meta") or {}),
            "screening": {
                "stage": "smoke",
                "queries": self.screen_queries,
                "eval_repeats": self.screen_eval_repeats,
                "min_speedup_fraction": self.screen_min_speedup_fraction,
                "screen_reference_eval_id": getattr(quality_reference, "eval_id", None),
                "outer_parent_eval_id": getattr(current_best, "eval_id", None),
            },
        }
        record.eval_result = meta
        return record

    def evaluate_stock_baseline(
        self,
        *,
        stage: str,
        keep_work: bool = False,
        commit_indexes: bool = True,
        write_legacy_result: bool = True,
    ) -> EvaluationRecord:
        """Run and store a stock-HMMER quality/speed reference for this run.

        The round-000 baseline is one of exactly two post-hoc profiling trigger
        points: its stage breakdown is the reference every later candidate is read
        against. The profiling pass is a separate evaluation and cannot affect the
        baseline's own cost_s.

        ``commit_indexes`` must be passed ``False`` by any INNER caller. Only the
        outer-round commit path may write the authoritative indexes
        (``baseline``/``current_best``/``accepted_lineage``/``leaderboard``); an
        inner screen that writes them can publish a state the round never chose.
        """
        patchset = self.build_stock_patchset(stage=stage)
        record = self.evaluate(
            patchset,
            AssembledConfig(),
            stage=stage,
            current_best=None,
            quality_reference=None,
            keep_work=keep_work,
            baseline=True,
            commit_indexes=commit_indexes,
            write_legacy_result=write_legacy_result,
        )
        if record.status == "ok":
            self.run_posthoc_profile(patchset)
        return record

    def evaluate_pyhmmer_profile_prepass(
        self,
        *,
        stage: str = "full",
        max_target_bytes: int | None = None,
    ) -> EvaluationRecord:
        """Store a diagnostic pyhmmer phase profile before C-patch search.

        This profile is evidence for target selection only.  It is deliberately
        not an accepted baseline and therefore cannot become the speed or
        quality incumbent used by the compiled-HMMER acceptance gate.
        """
        if stage not in {"smoke", "full"}:
            raise ValueError(f"unsupported pyhmmer profile stage: {stage}")
        max_iter = self.full_max_iter if stage == "full" else self.max_iter
        data_dir = self._data_dir_for_stage(stage)
        uniref_db = self._uniref_for_stage(stage, data_dir)
        result = run_jack.run_all_queries(
            AssembledConfig(),
            data_dir=data_dir,
            cpus=self.cpus,
            stage=stage,
            max_iter=max_iter,
            astral_db=self.astral_db,
            uniref_db=uniref_db,
            scop_txt=self.scop_txt,
            max_target_bytes=max_target_bytes,
        )
        eval_config = self._eval_config(
            stage=stage,
            max_iter=max_iter,
            data_dir=data_dir,
            uniref_db=uniref_db,
        )
        eval_config.update({
            "backend": "pyhmmer",
            "profile_prepass": True,
            "max_target_bytes": max_target_bytes,
        })
        record = EvaluationRecord(
            patchset_id="stock_pyhmmer_profile_prepass",
            round_idx=0,
            stage=stage,  # type: ignore[arg-type]
            status=result.status,
            decision="profile_prepass",
            eval_result=result.model_dump(mode="json"),
            eval_config=eval_config,
            dataset_manifest=self._dataset_manifest(data_dir=data_dir, uniref_db=uniref_db),
            source_hashes=_stock_source_hashes(self.hmmer_src_root),
            errors=list(result.errors),
            created_at=utc_now(),
        )
        record.eval_id = record_hash(record, id_field="eval_id")
        self.store.put_evaluation(record)
        self.store.write_index(
            f"pyhmmer_profile_prepass_{stage}",
            record.model_dump(mode="json"),
        )
        return record

    def build_stock_patchset(self, *, stage: str = "full") -> PatchsetRecord:
        """Create the immutable empty patchset used for stock baseline timing."""
        patchset = PatchsetRecord(
            round_idx=0,
            parent_patchset_id=None,
            patch_plan_id=None,
            base_hmmer_source=str(self.hmmer_src_root),
            source_hashes=_stock_source_hashes(self.hmmer_src_root),
            contract_snapshot=None,
            patches=[],
            context_pack_id=None,
            proposal_metadata={"provider": "stock", "stage": stage},
            created_at=utc_now(),
        )
        patchset.patchset_id = record_hash(patchset, id_field="patchset_id")
        self.store.put_patchset(patchset)
        return patchset

    def reevaluate_patchset_full(self, patchset_id: str) -> EvaluationRecord:
        """Run a stored patchset through Full only, in its own attempt namespace."""
        path = self.store.patchsets_dir / f"{patchset_id}.json"
        if not path.is_file():
            raise FileNotFoundError(f"stored patchset not found: {patchset_id}")
        patchset = PatchsetRecord(**json.loads(path.read_text(encoding="utf-8")))
        cfg: Any
        if patchset.patches:
            cfg = assemble(_mods_from_patchset(patchset), hmmer_src_root=self.hmmer_src_root)
            if cfg.errors:
                raise RuntimeError("failed to reconstruct stored patchset: " + "; ".join(cfg.errors))
        else:
            cfg = AssembledConfig()
        return self.evaluate(
            patchset,
            cfg,
            stage="full",
            attempt_kind="existing_patchset_full_reevaluation",
        )

    def evaluate(
        self,
        patchset: PatchsetRecord,
        cfg: Any,
        *,
        stage: str,
        current_best: EvalSummary | None = None,
        quality_reference: EvalSummary | None = None,
        keep_work: bool = False,
        baseline: bool = False,
        commit_indexes: bool = True,
        write_legacy_result: bool = True,
        query_timeout_reference: str | Path | None = None,
        attempt_kind: str = "primary",
        profiling: bool = False,
        record_experience: bool = True,
        paired_reference: EvalSummary | None = None,
    ) -> EvaluationRecord:
        """Run smoke/full compiled-HMMER eval for an already verified patchset.

        ``profiling`` selects the instrumented profiling build. It is only ever set
        by :meth:`run_posthoc_profile`, never by the inner trial loop -- profiling
        overhead inside the loop would corrupt the very speedup signal it measures.

        ``paired_reference`` is the summary the *verdict* is measured against, and it
        is deliberately separate from ``current_best``. ``current_best`` ranks the
        candidate inside the round (on later trials it is a faster sibling from the
        SAME round), but the round's verdict must be taken against the patchset the
        round actually started from. Defaults to ``current_best`` when not given.
        """
        if stage not in {"smoke", "full"}:
            raise ValueError(f"unsupported eval stage: {stage}")
        max_iter = self.full_max_iter if stage == "full" else self.max_iter
        # Only a profiling run changes the seed. Appending a suffix unconditionally
        # would move every existing work directory and defeat resume matching.
        seed_source = f"{patchset.patchset_id}:{stage}:{max_iter}:{self.cpus}:{self.eval_repeats}"
        if profiling:
            seed_source += ":profiling"
        eval_seed = sha256_text(seed_source)[:16]
        work_root = self.store.work_dir / eval_seed / f"hmmer_cli_{stage}"
        data_dir = self._data_dir_for_stage(stage)
        uniref_db = self._uniref_for_stage(stage, data_dir)
        eval_kwargs = {
            "data_dir": data_dir,
            "cpus": self.cpus,
            "stage": stage,
            "max_iter": max_iter,
            "astral_db": self.astral_db,
            "uniref_db": uniref_db,
            "scop_txt": self.scop_txt,
            "hmmer_src_root": self.hmmer_src_root,
            "work_root": work_root,
            "keep_work": keep_work,
            # Deliberately absent: external_profiler / perf_* . These used to be
            # threaded into every evaluation, which put profiler overhead inside the
            # inner trial loop. Profiling is now a separate post-commit pass; see
            # run_posthoc_profile(). run_all_queries defaults external_profiler to
            # "off", so omitting the keys changes nothing but the overhead.
            "query_timeout": self.query_timeout,
            "_profiling": profiling,
            "_slurm_mods": [mod.model_dump(mode="json") for mod in _mods_from_patchset(patchset)],
            "_slurm_round_idx": patchset.round_idx,
            "_patchset_id": patchset.patchset_id or "",
            "_attempt_kind": attempt_kind,
            "_slurm_query_timeout_reference": (
                str(Path(query_timeout_reference).resolve())
                if query_timeout_reference is not None
                else None
            ),
        }
        result = self._run_eval_repeats(cfg, **eval_kwargs)
        build = dict(result.meta.get("build") or {})
        evidence = _attempt_evidence(result)
        evidence_errors = _attempt_evidence_errors(evidence)
        # Freeze the conditions this number was measured under, from what the
        # evaluation actually recorded (worker meta + hashed inputs), and attach the
        # same fingerprint to the summary every later decision reads. Without this a
        # bare cost_s can be subtracted from an incomparable one.
        dataset_manifest = self._dataset_manifest(data_dir=data_dir, uniref_db=uniref_db)
        eval_config = {
            **self._eval_config(
                stage=stage,
                max_iter=max_iter,
                work_root=work_root,
                data_dir=data_dir,
                uniref_db=uniref_db,
            ),
            "reuse_fingerprint": self.reuse_fingerprint(stage=stage),
        }
        comparability = self._measurement_comparability(
            eval_config=eval_config,
            result=result,
            dataset_manifest=dataset_manifest,
            data_dir=data_dir,
        )
        result.meta = dict(result.meta or {})
        # ``setdefault``: a fingerprint the evaluation already recorded is evidence
        # from the run itself. Recomputing the same fields here is only a fallback,
        # and clobbering a recorded one with a controller-side reconstruction would
        # replace a fact with a derivation.
        result.meta.setdefault("comparability", comparability)
        eval_config["comparability"] = result.meta["comparability"]
        if baseline:
            if result.status != "ok":
                decision = f"rejected_baseline_{result.status}"
            elif evidence.get("checkpoint_enabled") and evidence_errors:
                decision = "rejected_baseline_incomplete_evidence"
            else:
                decision = "accepted_baseline"
        else:
            decision = self._decision(
                result,
                stage=stage,
                current_best=current_best,
                quality_reference=quality_reference or current_best,
                build=build,
            )
        if decision == "rejected_not_runtime_effective":
            result.errors.append(
                "runtime provenance failed: patched target did not change rebuilt "
                "jackhmmer/hmmsearch executable hashes"
            )
        if not baseline:
            decision, paired = self._apply_paired_current_best_confirmation(
                decision,
                result,
                cfg=cfg,
                stage=stage,
                current_best=current_best,
                work_root=work_root,
                eval_kwargs=eval_kwargs,
                paired_reference=paired_reference,
            )
            if paired:
                result.meta.setdefault("paired_current_best", paired)
        metric_delta = self._metric_delta(result, current_best, quality_reference=quality_reference)
        if not baseline and (result.meta or {}).get("paired_current_best"):
            metric_delta["paired_confirmation"] = result.meta["paired_current_best"]
        attempt_ids = list(evidence.get("attempt_ids") or [])
        primary_attempt_id = evidence.get("primary_attempt_id")
        query_completion_ids = list(evidence.get("query_completion_ids") or [])
        record = EvaluationRecord(
            patchset_id=patchset.patchset_id or "",
            round_idx=patchset.round_idx,
            stage=stage,  # type: ignore[arg-type]
            status=result.status,
            decision=decision,
            eval_result=result.model_dump(mode="json"),
            eval_config=eval_config,
            build=build,
            dataset_manifest=dataset_manifest,
            source_hashes=patchset.source_hashes,
            metric_delta=metric_delta,
            errors=list(result.errors),
            attempt_ids=attempt_ids,
            primary_attempt_id=primary_attempt_id,
            evaluation_fingerprint=evidence.get("evaluation_fingerprint"),
            query_completion_ids=query_completion_ids,
            requested_queries=evidence.get("requested_queries"),
            completed_queries=evidence.get("completed_queries"),
            resumed_queries=evidence.get("resumed_queries"),
            created_at=utc_now(),
        )
        record.eval_id = record_hash(record, id_field="eval_id")
        self.store.put_evaluation(record)
        if write_legacy_result:
            self._write_legacy_full_eval(patchset.round_idx, result, record)
        if record_experience:
            append_and_compress(self.store, experience_from_evaluation(patchset, record))
        if commit_indexes and decision.startswith("accepted"):
            self._update_indexes(patchset, record, baseline=baseline)
        return record

    def _run_eval_repeats(self, cfg: Any, *, eval_repeats: int | None = None, **kwargs: Any) -> EvalResult:
        repeat_results: list[EvalResult] = []
        repeats = max(1, eval_repeats if eval_repeats is not None else self.eval_repeats)
        base_work_root = Path(kwargs["work_root"])
        profiling = bool(kwargs.pop("_profiling", False))
        slurm_mods_value = kwargs.pop("_slurm_mods", None)
        if slurm_mods_value is None:
            slurm_mods_value = _mods_from_cfg(cfg)
        slurm_mods = list(slurm_mods_value)
        slurm_round_idx = int(kwargs.pop("_slurm_round_idx", 0))
        slurm_query_timeout_reference = kwargs.pop("_slurm_query_timeout_reference", None)
        patchset_id = str(kwargs.pop("_patchset_id", ""))
        requested_kind = str(kwargs.pop("_attempt_kind", "primary"))
        for repeat_idx in range(repeats):
            repeat_kwargs = dict(kwargs)
            repeat_kwargs["work_root"] = base_work_root / f"repeat_{repeat_idx + 1:02d}"
            attempt_kind = "repeat" if repeats > 1 and requested_kind == "primary" else requested_kind
            attempt_namespace = f"{base_work_root.name}/repeat_{repeat_idx + 1:02d}"
            if self.eval_backend == "slurm":
                # No external-profiler guard is needed any more: profiling is no
                # longer part of the per-eval configuration, so it can never reach
                # this path. (The guard used to reject it because a Slurm array
                # cannot host an in-process profiler.)
                from ..slurm_eval import submit_and_wait

                repeat_results.append(self._run_eval_attempt(
                    lambda: submit_and_wait(
                        stage=str(repeat_kwargs["stage"]),
                        mods=slurm_mods,
                        run_dir=self.store.run_dir,
                        round_idx=slurm_round_idx,
                        data_dir=repeat_kwargs["data_dir"],
                        max_iter=int(repeat_kwargs["max_iter"]),
                        astral_db=repeat_kwargs["astral_db"],
                        uniref_db=repeat_kwargs["uniref_db"],
                        scop_txt=repeat_kwargs["scop_txt"],
                        query_timeout_reference=slurm_query_timeout_reference,
                        profiling=profiling,
                    ),
                    patchset_id=patchset_id,
                    round_idx=slurm_round_idx,
                    attempt_kind=attempt_kind,
                    attempt_namespace=attempt_namespace,
                    kwargs=repeat_kwargs,
                    supports_query_checkpoints=False,
                ))
            else:
                repeat_results.append(self._run_eval_attempt(
                    lambda **hooks: run_hmmer_cli.run_all_queries(
                        cfg, **repeat_kwargs, profiling=profiling, **hooks
                    ),
                    patchset_id=patchset_id,
                    round_idx=slurm_round_idx,
                    attempt_kind=attempt_kind,
                    attempt_namespace=attempt_namespace,
                    kwargs=repeat_kwargs,
                    # A profiling pass must actually EXECUTE every query. Resuming a
                    # query from a checkpoint reconstructs it without ``stage_timing``
                    # and never runs the instrumented binary, so the pass would finish
                    # ``ok`` with no stage report and no error. The fingerprint does
                    # not include the profiling mode, so a repeated profiling attempt
                    # would otherwise resume everything and measure nothing.
                    supports_query_checkpoints=not profiling,
                ))
        if len(repeat_results) == 1:
            result = repeat_results[0]
            result.meta.setdefault("timing", {})["eval_repeats"] = 1
            if isinstance(result.meta.get("profile"), dict):
                result.meta["profile"] = _repeat_profile(result.meta["profile"], repeat_idx=1, eval_repeats=1)
            return result
        result = _median_eval_result(repeat_results)
        evidences = [_attempt_evidence(item) for item in repeat_results]
        selected_idx = int((result.meta.get("timing") or {}).get("selected_repeat_idx") or 1) - 1
        selected = evidences[max(0, min(selected_idx, len(evidences) - 1))]
        result.meta["evaluation_attempt"] = _merge_attempt_evidence(evidences, selected=selected)
        return result

    def _run_eval_attempt(
        self,
        backend,
        *,
        patchset_id: str,
        round_idx: int,
        attempt_kind: str,
        attempt_namespace: str,
        kwargs: dict[str, Any],
        supports_query_checkpoints: bool,
    ) -> EvalResult:
        """Persist an immutable started/terminal ledger around one backend run."""
        stage = str(kwargs["stage"])
        manifest = self._query_manifest(Path(kwargs["data_dir"]))
        input_sha256 = {
            "astral_db": _safe_hash(Path(kwargs["astral_db"])),
            "uniref_db": _safe_hash(Path(kwargs["uniref_db"])),
            "scop_txt": _safe_hash(Path(kwargs["scop_txt"])),
        }
        timing_payload = {
            "schema_version": "seq_align_timing_protocol.v1",
            "stage": stage,
            "cpus": int(kwargs.get("cpus", self.cpus)),
            "max_iter": int(kwargs.get("max_iter", 0)),
            "query_timeout": kwargs.get("query_timeout"),
            "queries": [{"idx": item["query_idx"], "sha256": item["query_sha256"]} for item in manifest],
            "inputs": input_sha256,
        }
        timing_fingerprint = sha256_text(canonical_json(timing_payload))
        evaluation_fingerprint = sha256_text(canonical_json({
            "schema_version": "seq_align_evaluation.v1",
            "patchset_id": patchset_id,
            "attempt_kind": attempt_kind,
            "attempt_namespace": attempt_namespace,
            "timing_protocol_fingerprint": timing_fingerprint,
        }))
        invocation_id = str(getattr(self, "invocation_id", None) or "standalone")
        started = EvaluationAttemptRecord(
            invocation_id=invocation_id,
            patchset_id=patchset_id,
            round_idx=round_idx,
            stage=stage,
            attempt_kind=attempt_kind,
            evaluation_fingerprint=evaluation_fingerprint,
            timing_protocol_fingerprint=timing_fingerprint,
            status="started",
            requested_queries=len(manifest),
            created_at=utc_now(),
        )
        started.attempt_id = record_hash(started, id_field="attempt_id")
        self.store.put_evaluation_attempt(started)

        state: dict[str, Any] = {
            "checkpoint_enabled": False,
            "binary_sha256": {},
            "completion_ids": [],
            "resumed_ids": [],
        }

        def resume_provider(binary_sha256: dict[str, str]) -> dict[int, EvalResult]:
            state["checkpoint_enabled"] = True
            state["binary_sha256"] = _normalize_binary_hashes(binary_sha256)
            resumed: dict[int, EvalResult] = {}
            by_idx = {item["query_idx"]: item for item in manifest}
            for raw in self.store.iter_query_completions():
                idx = int(raw.get("query_idx", 0) or 0)
                expected = by_idx.get(idx)
                if expected is None:
                    continue
                if raw.get("evaluation_fingerprint") != evaluation_fingerprint:
                    continue
                if _normalize_binary_hashes(raw.get("binary_sha256") or {}) != state["binary_sha256"]:
                    continue
                if raw.get("query_sha256") != expected["query_sha256"] or raw.get("status") != "ok":
                    continue
                if _completion_artifact_errors(raw):
                    continue
                completion = QueryCompletionRecord(**raw)
                resumed[idx] = _eval_result_from_completion(completion, stage=stage)
                state["completion_ids"].append(completion.query_completion_id)
                state["resumed_ids"].append(completion.query_completion_id)
            return resumed

        def on_query_complete(query_idx: int, query_path: str, query_work: Path, query_result: EvalResult) -> None:
            idx = int(query_idx)
            manifest_item = next((item for item in manifest if item["query_idx"] == idx), None)
            resolved = Path(query_path).resolve()
            query_sha = manifest_item["query_sha256"] if manifest_item else _safe_hash(resolved)
            profile = query_result.meta.get("profile") if isinstance(query_result.meta, dict) else {}
            artifacts = dict((profile or {}).get("artifacts") or {}) if isinstance(profile, dict) else {}
            query_id = str((query_result.meta or {}).get("query_id") or _fasta_query_id(resolved))
            query_fingerprint = sha256_text(canonical_json({
                "evaluation_fingerprint": evaluation_fingerprint,
                "query_idx": idx,
                "query_sha256": query_sha,
                "binary_sha256": state["binary_sha256"],
            }))
            completion = QueryCompletionRecord(
                attempt_id=started.attempt_id or "",
                invocation_id=invocation_id,
                evaluation_fingerprint=evaluation_fingerprint,
                query_fingerprint=query_fingerprint,
                patchset_id=patchset_id,
                stage=stage,
                attempt_kind=attempt_kind,
                query_idx=idx,
                query_id=query_id,
                query_path=str(resolved),
                query_sha256=query_sha,
                binary_sha256=dict(state["binary_sha256"]),
                input_sha256=input_sha256,
                status=query_result.status,
                tp=query_result.tp,
                fp=query_result.fp,
                cost_s=query_result.cost_s,
                iterations=query_result.iterations,
                artifacts=artifacts,
                errors=list(query_result.errors),
                created_at=utc_now(),
            )
            completion.query_completion_id = record_hash(completion, id_field="query_completion_id")
            self.store.put_query_completion(completion)
            state["completion_ids"].append(completion.query_completion_id)

        try:
            result = backend(
                resume_provider=resume_provider,
                on_query_complete=on_query_complete,
            ) if supports_query_checkpoints else backend()
        except run_hmmer_cli.EvaluationInterrupted as exc:
            interruption = {
                "type": type(exc).__name__,
                "message": str(exc),
                "requested_queries": len(manifest),
            }
            if isinstance(exc, run_hmmer_cli.QueryTimeoutError):
                interruption["timed_out_query"] = {
                    "query_idx": exc.query_idx,
                    "query_path": exc.query_path,
                    "label": exc.label,
                    "timeout_s": exc.timeout_s,
                    "wall_s": exc.wall_s,
                }
            self._put_terminal_attempt(
                started,
                status="interrupted",
                state=state,
                requested_queries=len(manifest),
                aggregate_result=None,
                interruption=interruption,
                errors=[str(exc)],
            )
            raise

        terminal = self._put_terminal_attempt(
            started,
            status="complete",
            state=state,
            requested_queries=len(manifest),
            aggregate_result=result.model_dump(mode="json"),
            interruption=None,
            errors=list(result.errors),
            reported_completed=result.n_seqs,
        )
        result.meta = dict(result.meta)
        completion_ids = _unique_strings(state["completion_ids"])
        evidence_errors: list[str] = []
        if state["checkpoint_enabled"] and len(completion_ids) != len(manifest):
            evidence_errors.append(
                f"committed queries are {len(completion_ids)}, expected {len(manifest)}"
            )
        completion_id_set = set(completion_ids)
        completion_by_id = {
            item.get("query_completion_id"): item
            for item in self.store.iter_query_completions()
            if item.get("query_completion_id") in completion_id_set
        }
        for completion_id in completion_ids:
            evidence_errors.extend(_completion_artifact_errors(completion_by_id.get(completion_id) or {}))
        result.meta["evaluation_attempt"] = {
            "checkpoint_enabled": bool(state["checkpoint_enabled"]),
            "attempt_ids": [terminal.attempt_id],
            "primary_attempt_id": terminal.attempt_id,
            "evaluation_fingerprint": evaluation_fingerprint,
            "query_completion_ids": completion_ids,
            "requested_queries": len(manifest),
            "completed_queries": terminal.completed_queries,
            "resumed_queries": len(_unique_strings(state["resumed_ids"])),
            "timing_protocol_fingerprint": timing_fingerprint,
            "binary_sha256": dict(state["binary_sha256"]),
            "evidence_errors": evidence_errors,
        }
        # Fingerprint EVERY attempt, not just the ones that reach ``evaluate``.
        # The paired confirmation measures its candidate and its reference through
        # this path, and those two numbers are divided to produce the verdict's
        # per-pair statistic -- so a fingerprint that only ``evaluate`` set would
        # leave the one comparison that actually decides commits unguarded.
        result.meta.setdefault("comparability", self._measurement_comparability(
            eval_config=self._eval_config(
                stage=stage,
                max_iter=int(kwargs.get("max_iter", 0) or 0),
                work_root=Path(kwargs["work_root"]) if kwargs.get("work_root") else None,
                data_dir=Path(kwargs["data_dir"]),
                uniref_db=Path(kwargs["uniref_db"]),
            ),
            result=result,
            dataset_manifest=self._dataset_manifest(
                data_dir=Path(kwargs["data_dir"]), uniref_db=Path(kwargs["uniref_db"])
            ),
            data_dir=Path(kwargs["data_dir"]),
        ))
        return result

    def _put_terminal_attempt(
        self,
        started: EvaluationAttemptRecord,
        *,
        status: str,
        state: dict[str, Any],
        requested_queries: int,
        aggregate_result: dict[str, Any] | None,
        interruption: dict[str, Any] | None,
        errors: list[str],
        reported_completed: int = 0,
    ) -> EvaluationAttemptRecord:
        ids = _unique_strings(state["completion_ids"])
        terminal = EvaluationAttemptRecord(
            invocation_id=started.invocation_id,
            patchset_id=started.patchset_id,
            round_idx=started.round_idx,
            stage=started.stage,
            attempt_kind=started.attempt_kind,
            evaluation_fingerprint=started.evaluation_fingerprint,
            timing_protocol_fingerprint=started.timing_protocol_fingerprint,
            status=status,
            parent_attempt_id=started.attempt_id,
            requested_queries=requested_queries,
            completed_queries=max(len(ids), int(reported_completed or 0)),
            resumed_queries=len(_unique_strings(state["resumed_ids"])),
            query_completion_ids=ids,
            binary_sha256=dict(state["binary_sha256"]),
            aggregate_result=aggregate_result,
            interruption=interruption,
            errors=errors,
            created_at=utc_now(),
        )
        terminal.attempt_id = record_hash(terminal, id_field="attempt_id")
        self.store.put_evaluation_attempt(terminal)
        return terminal

    def _query_manifest(self, data_dir: Path) -> list[dict[str, Any]]:
        key = str(data_dir.resolve())
        if key in self._query_manifest_cache:
            cached = self._query_manifest_cache[key]
            if all(_manifest_file_identity_matches(item) for item in cached):
                return [dict(item) for item in cached]
            self._query_manifest_cache.pop(key, None)
        paths = [Path(path).resolve() for path in run_jack.query_fastas(data_dir)]
        run_jack.validate_query_fastas([str(path) for path in paths])
        manifest: list[dict[str, Any]] = []
        for idx, path in enumerate(paths, start=1):
            stat = path.stat()
            manifest.append({
                "query_idx": idx,
                "query_id": _fasta_query_id(path),
                "query_path": str(path),
                "query_sha256": hash_file(path),
                "size": stat.st_size,
                "device": stat.st_dev,
                "inode": stat.st_ino,
                "mtime_ns": stat.st_mtime_ns,
            })
        self._query_manifest_cache[key] = manifest
        return [dict(item) for item in manifest]

    def _apply_paired_current_best_confirmation(
        self,
        decision: str,
        result: EvalResult,
        *,
        cfg: Any,
        stage: str,
        current_best: EvalSummary | None,
        work_root: Path,
        eval_kwargs: dict[str, Any],
        paired_reference: EvalSummary | None = None,
    ) -> tuple[str, dict[str, Any] | None]:
        # The trigger and the statistic both measure against the VERDICT reference
        # (the round parent), never against the in-round ranking incumbent.
        reference_summary = paired_reference or current_best
        if not self._should_run_paired_current_best(
            result, decision=decision, stage=stage, current_best=reference_summary
        ):
            return decision, None
        trigger_reasons = (
            ["first_candidate_confirmation"]
            if reference_summary is None
            else ["accepted_best_confirmation"]
            if decision == "accepted_speedup"
            else ["near_threshold_confirmation"]
        )
        paired: dict[str, Any] = {
            "schema_version": "seq_align_paired_current_best.v2",
            "status": "pending",
            "original_decision": decision,
            # Named for the role it plays: the reference the verdict is taken
            # against (the round parent on the hierarchical path).
            "reference_role": "round_parent" if paired_reference is not None else "current_best",
            "current_best_eval_id": reference_summary.eval_id if reference_summary else None,
            "current_best_patchset_id": reference_summary.patchset_id if reference_summary else None,
            "paired_current_best_repeats": self.paired_current_best_repeats,
            "near_speedup_band_fraction": self.near_speedup_band_fraction,
            "trigger_reasons": trigger_reasons,
            "mode": "paired_current_best" if reference_summary is not None else "repeated_candidate",
            # The single unpaired measurement this candidate entered the paired
            # confirmation with. Kept so the record can still be audited after
            # result.cost_s is replaced by the paired median below.
            "primary_candidate_cost_s": result.cost_s,
            "schedule": [],
            "candidate_cost_s_values": [],
            "reference_cost_s_values": [],
            "pair_speedup_fractions": [],
            "timing_protocol_fingerprints": [],
            "run_errors": [],
            "evidence_errors": [],
            "comparability_errors": [],
            "quality_errors": [],
            "validation_errors": [],
            "confirmed": False,
        }
        try:
            ref_cfg = self._config_for_patchset_id(
                reference_summary.patchset_id if reference_summary else None
            )
            if reference_summary is not None and ref_cfg is None:
                paired["run_errors"].append("current-best patchset could not be reconstructed")
                paired["status"] = "run_error"
                paired["validation_errors"] = list(paired["run_errors"])
                return "rejected_paired_confirmation_error", paired

            candidate_results: list[EvalResult] = []
            reference_results: list[EvalResult] = []
            order = 0
            for pair_idx in range(1, self.paired_current_best_repeats + 1):
                roles = ("candidate",) if ref_cfg is None else (
                    ("candidate", "reference") if pair_idx % 2 else ("reference", "candidate")
                )
                for role in roles:
                    order += 1
                    role_cfg = cfg if role == "candidate" else ref_cfg
                    role_patchset_id = (
                        str(eval_kwargs.get("_patchset_id") or "")
                        if role == "candidate"
                        else str(reference_summary.patchset_id or "")
                    )
                    role_result = self._run_named_eval_attempt(
                        role_cfg,
                        eval_kwargs=eval_kwargs,
                        work_root=work_root / "paired_current_best" / f"pair_{pair_idx:02d}_{role}",
                        attempt_kind=f"paired_{role}",
                        attempt_namespace=f"paired_current_best/pair_{pair_idx:02d}_{role}",
                        patchset_id=role_patchset_id,
                    )
                    evidence = _attempt_evidence(role_result)
                    schedule_item = {
                        "pair_idx": pair_idx,
                        "order": order,
                        "role": role,
                        "status": role_result.status,
                        "tp": role_result.tp,
                        "fp": role_result.fp,
                        "cost_s": role_result.cost_s,
                        "attempt_id": evidence.get("primary_attempt_id"),
                        "binary_sha256": dict(evidence.get("binary_sha256") or {}),
                        "timing_protocol_fingerprint": evidence.get("timing_protocol_fingerprint"),
                    }
                    paired["schedule"].append(schedule_item)
                    if evidence.get("timing_protocol_fingerprint"):
                        paired["timing_protocol_fingerprints"].append(evidence["timing_protocol_fingerprint"])
                    if role_result.status != "ok":
                        paired["run_errors"].append(
                            f"{role} pair {pair_idx} returned {role_result.status}"
                        )
                    paired["evidence_errors"].extend(
                        f"{role} pair {pair_idx}: {error}"
                        for error in _attempt_evidence_errors(evidence)
                    )
                    (candidate_results if role == "candidate" else reference_results).append(role_result)
        except Exception as exc:
            paired["run_errors"].append(f"{type(exc).__name__}: {exc}")
            paired["status"] = "run_error"
            paired["validation_errors"] = list(paired["run_errors"])
            return "rejected_paired_confirmation_error", paired

        paired["timing_protocol_fingerprints"] = _unique_strings(paired["timing_protocol_fingerprints"])
        candidate_costs = [item.cost_s for item in candidate_results]
        reference_costs = [item.cost_s for item in reference_results]
        paired["candidate_cost_s_values"] = candidate_costs
        paired["reference_cost_s_values"] = reference_costs
        paired["candidate_median_cost_s"] = float(median(candidate_costs)) if candidate_costs else None
        paired["reference_median_cost_s"] = float(median(reference_costs)) if reference_costs else None
        paired["candidate_cost_s"] = paired["candidate_median_cost_s"]
        paired["paired_cost_s"] = paired["reference_median_cost_s"]

        if reference_results:
            # Refuse to divide a candidate measurement by a reference measurement
            # taken under different conditions. A paired schedule interleaves the
            # two roles, but interleaving only removes drift -- it does not make two
            # different configurations the same measurement.
            from .comparability import IncomparableMeasurement, compare_summaries

            for pair_idx, (candidate_item, reference_item) in enumerate(
                zip(candidate_results, reference_results), start=1
            ):
                try:
                    compare_summaries(reference_item, candidate_item)
                except IncomparableMeasurement as exc:
                    paired["comparability_errors"].append(
                        f"pair {pair_idx}: {exc}"
                    )
            for candidate_item, reference_item in zip(candidate_results, reference_results):
                if candidate_item.tp < reference_item.tp * self.accept_min_tp_fraction:
                    paired["quality_errors"].append(
                        f"candidate TP {candidate_item.tp} < reference TP {reference_item.tp}"
                    )
                if candidate_item.fp > reference_item.fp:
                    paired["quality_errors"].append(
                        f"candidate FP {candidate_item.fp} > reference FP {reference_item.fp}"
                    )
            pair_speedups = [
                ((reference_item.cost_s - candidate_item.cost_s) / reference_item.cost_s)
                if reference_item.cost_s else 0.0
                for candidate_item, reference_item in zip(candidate_results, reference_results)
            ]
            paired["pair_speedup_fractions"] = pair_speedups
            paired["median_pair_speedup_fraction"] = float(median(pair_speedups)) if pair_speedups else 0.0
            paired["speedup_fraction"] = paired["median_pair_speedup_fraction"]
            paired["statistic"] = "median_pair_speedup_fraction"
            paired["required_cost_s"] = (
                paired["reference_median_cost_s"] * (1.0 - self.min_speedup_fraction)
                if paired["reference_median_cost_s"] is not None else None
            )
        else:
            paired["statistic"] = "median_candidate_cost_s"
            paired["median_pair_speedup_fraction"] = None
            paired["speedup_fraction"] = None
            paired["required_cost_s"] = None

        candidate_quality = sorted({(item.tp, item.fp) for item in candidate_results})
        if len(candidate_quality) > 1:
            values = ", ".join(f"tp={tp} fp={fp}" for tp, fp in candidate_quality)
            paired["quality_errors"].append(
                f"candidate TP/FP is unstable across repeats: {values}"
            )

        if paired["run_errors"]:
            paired["status"] = next(
                (item.status for item in candidate_results + reference_results if item.status != "ok"),
                "run_error",
            )
            paired["validation_errors"] = list(paired["run_errors"])
            return "rejected_paired_confirmation_error", paired
        if paired["comparability_errors"]:
            # A pair whose two halves were not measured alike cannot produce a
            # per-pair ratio, so the paired statistic is undefined: refuse rather
            # than average incomparable ratios into a verdict.
            paired["status"] = "incomparable"
            paired["validation_errors"] = list(paired["comparability_errors"])
            return "rejected_incomparable_measurement", paired
        if paired["evidence_errors"]:
            paired["status"] = "incomplete_evidence"
            paired["validation_errors"] = list(paired["evidence_errors"])
            return "rejected_paired_incomplete_evidence", paired
        if paired["quality_errors"]:
            paired["status"] = "quality_regression"
            paired["validation_errors"] = list(paired["quality_errors"])
            result.cost_s = round(float(paired["candidate_median_cost_s"]), 4)
            return "rejected_paired_quality_regression", paired

        paired["status"] = "ok"
        if reference_results:
            paired["confirmed"] = paired["median_pair_speedup_fraction"] > self.min_speedup_fraction - 1e-12
            # The stored cost_s must not depend on the verdict: whichever way the
            # paired estimate lands, the candidate's cost is the paired median.
            # The pre-pairing measurement stays available as
            # paired["primary_candidate_cost_s"].
            result.cost_s = round(float(paired["candidate_median_cost_s"]), 4)
            if not paired["confirmed"]:
                return "rejected_not_faster_paired", paired
            return "accepted_speedup_paired", paired
        paired["confirmed"] = True
        result.cost_s = round(float(paired["candidate_median_cost_s"]), 4)
        return "accepted_full_paired", paired

    def _should_run_paired_current_best(
        self,
        result: EvalResult,
        *,
        decision: str,
        stage: str,
        current_best: EvalSummary | None,
    ) -> bool:
        if self.paired_current_best_repeats <= 0 or stage != "full" or result.status != "ok":
            return False
        if current_best is None:
            return decision == "accepted_full"
        if not current_best.patchset_id or not current_best.cost_s:
            return False
        if decision == "accepted_speedup":
            return True
        if decision != "rejected_not_faster":
            return False
        speedup_fraction = (current_best.cost_s - result.cost_s) / current_best.cost_s
        return abs(speedup_fraction - self.min_speedup_fraction) <= self.near_speedup_band_fraction + 1e-12

    def _run_named_eval_attempt(
        self,
        cfg: Any,
        *,
        eval_kwargs: dict[str, Any],
        work_root: Path,
        attempt_kind: str,
        attempt_namespace: str,
        patchset_id: str,
    ) -> EvalResult:
        kwargs = {key: value for key, value in eval_kwargs.items() if not key.startswith("_")}
        kwargs["work_root"] = work_root
        # ``_profiling`` is a behavioural flag, not an underscore-private detail to
        # be dropped: carrying it into the rebuild is what makes a re-measured
        # evaluation instrumented when the caller asked for instrumentation.
        profiling = bool(eval_kwargs.get("_profiling", False))
        round_idx = int(eval_kwargs.get("_slurm_round_idx") or 0)
        if self.eval_backend == "slurm":
            from ..slurm_eval import submit_and_wait

            return self._run_eval_attempt(
                lambda: submit_and_wait(
                    stage=str(kwargs["stage"]),
                    mods=_mods_from_cfg(cfg),
                    run_dir=self.store.run_dir,
                    round_idx=round_idx,
                    data_dir=kwargs["data_dir"],
                    max_iter=int(kwargs["max_iter"]),
                    astral_db=kwargs["astral_db"],
                    uniref_db=kwargs["uniref_db"],
                    scop_txt=kwargs["scop_txt"],
                    query_timeout_reference=eval_kwargs.get("_slurm_query_timeout_reference"),
                    profiling=profiling,
                    # Pin the recorded per-query core count to the one the worker
                    # will actually dispatch. A mismatch is a hard error rather than
                    # a record that claims a measurement shape it never ran.
                    expected_query_cpus=int(self.cpus),
                ),
                patchset_id=patchset_id,
                round_idx=round_idx,
                attempt_kind=attempt_kind,
                attempt_namespace=attempt_namespace,
                kwargs=kwargs,
                supports_query_checkpoints=False,
            )
        return self._run_eval_attempt(
            lambda **hooks: run_hmmer_cli.run_all_queries(cfg, **kwargs, profiling=profiling, **hooks),
            patchset_id=patchset_id,
            round_idx=round_idx,
            attempt_kind=attempt_kind,
            attempt_namespace=attempt_namespace,
            kwargs=kwargs,
            # Same reason as the repeat path: instrumentation cannot be served from a
            # checkpoint.
            supports_query_checkpoints=not profiling,
        )

    def _config_for_patchset_id(self, patchset_id: str | None) -> AssembledConfig | None:
        if not patchset_id:
            return None
        path = self.store.patchsets_dir / f"{patchset_id}.json"
        if not path.exists():
            return None
        patchset = PatchsetRecord(**json.loads(path.read_text(encoding="utf-8")))
        if not patchset.patches:
            return AssembledConfig()
        cfg = assemble(_mods_from_patchset(patchset), hmmer_src_root=self.hmmer_src_root)
        if cfg.errors:
            raise RuntimeError("failed to reconstruct current-best config: " + "; ".join(cfg.errors))
        return cfg

    def _decision(
        self,
        result: EvalResult,
        *,
        stage: str,
        current_best: EvalSummary | None,
        quality_reference: EvalSummary | None = None,
        build: dict[str, Any] | None = None,
    ) -> str:
        if result.status != "ok":
            return f"rejected_{result.status}"
        if build and build.get("runtime_effective") is False:
            return "rejected_not_runtime_effective"

        reference = quality_reference or current_best
        # Refuse BEFORE any tp/fp or cost arithmetic. A tp/fp comparison across a
        # different query set is as meaningless as a cost subtraction across a
        # different core count, and every verdict below is a subtraction or a ratio.
        refusal = self._refuse_incomparable(reference, result, stage=stage)
        if refusal is not None:
            return refusal

        if reference is not None:
            min_tp = reference.tp * self.accept_min_tp_fraction
            if result.tp + 1e-12 < min_tp:
                return "rejected_quality_tp_regression"
            if result.fp > reference.fp:
                return "rejected_quality_fp_regression"

        if current_best is None:
            return "accepted_smoke" if stage == "smoke" else "accepted_full"

        target_cost = current_best.cost_s * (1.0 - self.min_speedup_fraction)
        if result.cost_s + 1e-12 < target_cost:
            return "accepted_smoke_speedup" if stage == "smoke" else "accepted_speedup"
        return "rejected_not_faster"

    def _metric_delta(
        self,
        result: EvalResult,
        current_best: EvalSummary | None,
        *,
        quality_reference: EvalSummary | None = None,
    ) -> dict[str, Any]:
        out: dict[str, Any] = {
            "objective": "speed_under_tp_fp_non_regression",
            "min_speedup_fraction": self.min_speedup_fraction,
        }
        if current_best is None:
            out["against_speed_best"] = None
        else:
            from .comparability import IncomparableMeasurement, compare_summaries

            incomparable = (result.meta or {}).get("incomparable_measurement")
            if not incomparable:
                try:
                    compare_summaries(current_best, result)
                except IncomparableMeasurement as exc:
                    incomparable = {"mismatches": dict(exc.mismatches), "message": str(exc)}
                    result.meta = dict(result.meta or {})
                    result.meta["incomparable_measurement"] = incomparable
            if incomparable:
                # Do NOT emit cost_s_delta / speedup_fraction. An absent ratio is
                # the honest output for two numbers that were not measured alike;
                # a number here would be read as a result.
                out.update({
                    "against_speed_best": current_best.eval_id,
                    "speed_best_cost_s": current_best.cost_s,
                    "required_cost_s": None,
                    "cost_s_delta": None,
                    "speedup_fraction": None,
                    "incomparable_measurement": incomparable,
                })
            else:
                out.update({
                    "against_speed_best": current_best.eval_id,
                    "speed_best_cost_s": current_best.cost_s,
                    "required_cost_s": current_best.cost_s * (1.0 - self.min_speedup_fraction),
                    "cost_s_delta": result.cost_s - current_best.cost_s,
                    "speedup_fraction": ((current_best.cost_s - result.cost_s) / current_best.cost_s) if current_best.cost_s else 0.0,
                })
        reference = quality_reference or current_best
        if reference is None:
            out["against_quality_reference"] = None
        else:
            out.update({
                "against_quality_reference": reference.eval_id,
                "tp_delta": result.tp - reference.tp,
                "fp_delta": result.fp - reference.fp,
                "tp_fp_ratio_delta": result.tp_fp_ratio - reference.tp_fp_ratio,
                "min_allowed_tp": reference.tp * self.accept_min_tp_fraction,
                "max_allowed_fp": reference.fp,
            })
        return out

    def _eval_config(
        self,
        *,
        stage: str,
        max_iter: int | None = None,
        work_root: Path | None = None,
        data_dir: Path | None = None,
        uniref_db: Path | None = None,
    ) -> dict[str, Any]:
        return {
            "backend": f"hmmer_cli_{self.eval_backend}",
            "stage": stage,
            "cpus": self.cpus,
            "max_iter": max_iter,
            "eval_repeats": self.eval_repeats,
            "paired_current_best_repeats": self.paired_current_best_repeats,
            "near_speedup_band_fraction": self.near_speedup_band_fraction,
            "min_speedup_fraction": self.min_speedup_fraction,
            "accept_min_tp_fraction": self.accept_min_tp_fraction,
            "objective": "speed_under_tp_fp_non_regression",
            # No external_profiler / perf_* here: the evaluation configuration and
            # its reuse fingerprint must describe how the trial was *measured*, not
            # which optional diagnostic might later be run against it. Keeping them
            # here made every fingerprint depend on a diagnostic-only setting.
            "query_timeout": self.query_timeout,
            "data_dir": str(data_dir or self.data_dir),
            "astral_db": str(self.astral_db),
            "uniref_db": str(uniref_db or self.uniref_db),
            "scop_txt": str(self.scop_txt),
            "hmmer_src_root": str(self.hmmer_src_root),
            "work_root": str(work_root) if work_root else None,
        }

    def _measurement_comparability(
        self,
        *,
        eval_config: dict[str, Any],
        result: EvalResult,
        dataset_manifest: dict[str, Any],
        data_dir: Path,
    ) -> dict[str, Any]:
        """Collect the conditions a cost number is only valid under.

        Every value is taken from the evaluation's own record: the worker's meta
        (what it actually dispatched) and the hashed input manifest (which exact
        bytes it searched). Configuration files are never consulted, because a
        config field is a claim about a run and the whole point of the gate is to
        distrust that claim.
        """
        from .comparability import comparison_fingerprint

        meta = dict(result.meta or {})

        def _sha(info: Any) -> str | None:
            if isinstance(info, dict):
                value = info.get("sha256")
                return str(value) if value else None
            return None

        if not meta.get("uniref_db_sha256"):
            meta["uniref_db_sha256"] = _sha(dataset_manifest.get("uniref_db"))
        if not meta.get("astral_db_sha256"):
            meta["astral_db_sha256"] = _sha(dataset_manifest.get("astral_db"))
        if not meta.get("scop_txt_sha256"):
            meta["scop_txt_sha256"] = _sha(dataset_manifest.get("scop_txt"))
        manifest = self._query_manifest(Path(data_dir))
        fingerprint = comparison_fingerprint(
            eval_config=eval_config,
            meta=meta,
            query_manifest=manifest,
        )
        fingerprint["schema_version"] = "seqalign_comparability.v1"
        return fingerprint

    def _refuse_incomparable(self, reference: EvalSummary | None, result: EvalResult, *, stage: str) -> str | None:
        """Return a rejection decision when ``result`` may not be compared, else None.

        Called on every path that turns two cost numbers into a verdict. A refusal
        is a *decision*, not an exception: the loop keeps running, the candidate is
        never accepted, and the mismatch is recorded field by field so it can be
        read out of the ledger instead of silently producing a speedup.
        """
        if reference is None:
            return None
        from .comparability import IncomparableMeasurement, compare_summaries

        reference_summary = EvalSummary(
            eval_id=reference.eval_id,
            patchset_id=reference.patchset_id,
            stage=reference.stage,
            status=reference.status,
            tp=reference.tp,
            fp=reference.fp,
            cost_s=reference.cost_s,
            comparability=reference.comparability,
        )
        candidate_summary = EvalSummary(
            eval_id=(result.meta or {}).get("eval_id"),
            patchset_id=(result.meta or {}).get("patchset_id"),
            stage=stage,
            status=result.status,
            tp=result.tp,
            fp=result.fp,
            cost_s=result.cost_s,
            comparability=(result.meta or {}).get("comparability"),
        )
        try:
            compare_summaries(reference_summary, candidate_summary)
        except IncomparableMeasurement as exc:
            result.errors.append(
                "refused to compare: " + "; ".join(
                    f"{key}={value!r}" for key, value in sorted(exc.mismatches.items())
                )
            )
            result.meta = dict(result.meta or {})
            result.meta["incomparable_measurement"] = {
                "schema_version": "seqalign_incomparable.v1",
                "mismatches": dict(exc.mismatches),
                "reference_eval_id": reference.eval_id,
                "message": str(exc),
            }
            return "rejected_incomparable_measurement"
        return None

    def runtime_identity(self) -> dict[str, Any]:
        """Return the binaries used by the configured evaluation runtime.

        Local and host-based Slurm evaluations retain the historical host
        binary fingerprint. Docker-backed Slurm evaluations instead probe the
        immutable configured image, because its prebuilt HMMER binaries are
        not present on the controller host.
        """
        if self.eval_backend == "slurm":
            runtime = os.getenv("SEQ_ALIGN_SLURM_RUNTIME", "host").strip().lower()
            if runtime == "docker":
                return _slurm_docker_runtime_identity(self.store)
            if runtime != "host":
                raise RuntimeIdentityUnavailable(
                    f"unsupported SEQ_ALIGN_SLURM_RUNTIME={runtime!r}"
                )
        return {
            "schema_version": "seq_align_runtime_identity.v1",
            "backend": self.eval_backend,
            "runtime": "host",
            "binaries": _stock_binary_fingerprint(self.hmmer_src_root),
        }

    def reuse_fingerprint(
        self,
        *,
        stage: str,
        runtime_identity: dict[str, Any] | None = None,
    ) -> str:
        """Identify every input that makes a cached timing result reusable."""
        data_dir = self._data_dir_for_stage(stage)
        uniref_db = self._uniref_for_stage(stage, data_dir)
        max_iter = self.full_max_iter if stage == "full" else self.max_iter
        eval_config = self._eval_config(
            stage=stage,
            max_iter=max_iter,
            work_root=None,
            data_dir=data_dir,
            uniref_db=uniref_db,
        )
        eval_config.pop("work_root", None)
        payload = {
            "schema_version": "seq_align_eval_reuse.v2",
            "source_hashes": _stock_source_hashes(self.hmmer_src_root),
            "runtime_identity": runtime_identity or self.runtime_identity(),
            "dataset_manifest": self._dataset_manifest(data_dir=data_dir, uniref_db=uniref_db),
            "eval_config": eval_config,
            "host": {
                "cpu_count": os.cpu_count(),
                "machine": os.uname().machine if hasattr(os, "uname") else None,
                "sysname": os.uname().sysname if hasattr(os, "uname") else None,
            },
        }
        return sha256_text(canonical_json(payload))

    def _dataset_manifest(self, *, data_dir: Path | None = None, uniref_db: Path | None = None) -> dict[str, Any]:
        actual_data_dir = data_dir or self.data_dir
        actual_uniref = uniref_db or self.uniref_db
        query_files = [Path(p) for p in run_jack.query_fastas(actual_data_dir)]
        return {
            "data_dir": str(actual_data_dir),
            "query_count": len(query_files),
            "queries": [p.name for p in query_files[:50]],
            "query_files": [_file_info(path, sha256=True) for path in query_files],
            "astral_db": _file_info(self.astral_db, sha256=True),
            "uniref_db": _file_info(
                actual_uniref,
                digest_cache_dir=self.store.root / "digest_cache",
            ),
            "scop_txt": _file_info(self.scop_txt, sha256=True),
        }

    def _data_dir_for_stage(self, stage: str) -> Path:
        if stage != "smoke":
            return self.data_dir
        smoke_dir = self.data_dir / "smoke"
        if not smoke_dir.is_dir():
            raise ValueError(f"smoke query directory is missing: {smoke_dir}")
        return self._screened_query_dir(smoke_dir)

    def _screened_query_dir(self, smoke_dir: Path) -> Path:
        """Cap the screen to ``screen_queries`` queries without touching the source set.

        A materialised subset keeps the run reproducible and auditable: the same
        first N query files are used on resume, so a screened slot can be replayed
        and compared. Accepted only when a labelled part-file layout is present
        (``all_samples.part_NNN.fasta``); anything else is used as-is.
        """
        limit = self.screen_queries
        parts = sorted(smoke_dir.glob("all_samples.part_*.fasta"))
        if len(parts) <= limit:
            return smoke_dir
        subset = smoke_dir.parent / f"smoke_screen_{limit}"
        subset.mkdir(parents=True, exist_ok=True)
        for part in parts[:limit]:
            target = subset / part.name
            if not target.exists():
                shutil.copy2(part, target)
        # The screen compares against the smoke baseline, so it must resolve the
        # same smoke database; the runner looks for it beside the query files.
        for extra in ("uniref_smoke.fasta", "uniref.fasta"):
            source = smoke_dir / extra
            if source.is_file() and not (subset / extra).exists():
                shutil.copy2(source, subset / extra)
        return subset

    def _uniref_for_stage(self, stage: str, data_dir: Path) -> Path:
        if stage == "smoke":
            if data_dir.name == "smoke":
                candidates = [
                    data_dir / "uniref_smoke.fasta",
                    data_dir.parent / "uniref_smoke.fasta",
                ]
            else:
                candidates = [
                    data_dir / "uniref_smoke.fasta",
                    data_dir / "smoke" / "uniref_smoke.fasta",
                ]
            for cand in candidates:
                if cand.is_file():
                    return cand
            raise ValueError(
                f"smoke UniRef database is missing; tried: {[str(c) for c in candidates]}"
            )
        return self.uniref_db

    def _source_hashes(self, snapshot: ContractSnapshot) -> dict[str, str]:
        out: dict[str, str] = {}
        for target in {snapshot.c_target_file, Path(snapshot.c_target_file).name}:
            if not target:
                continue
            candidate = _resolve_patch_source(self.hmmer_src_root, target)
            if candidate.exists():
                out[str(candidate)] = hash_file(candidate)
        return out

    def _source_hashes_for_root(self, root: Path, patches: list[PatchRecord]) -> dict[str, str]:
        out: dict[str, str] = {}
        for patch in patches:
            candidate = _resolve_patch_source(root, patch.c_target_file)
            if candidate.exists():
                out[patch.c_target_file] = hash_file(candidate)
        return out

    def _proposal_diff(
        self,
        proposal: PatchProposal,
        target_file: str,
        *,
        source_root: Path | None = None,
    ) -> tuple[str, list[str]]:
        root = source_root or self.hmmer_src_root
        if proposal.unified_diff.strip():
            # Do not pass through AI-generated line numbers verbatim. Apply the
            # diff to an isolated source copy and regenerate a canonical diff so
            # the hunk coordinates are always relative to the actual source_root.
            normalized_diff, diff_errors = _normalize_unified_diff(root, target_file, proposal.unified_diff)
            if not diff_errors:
                return normalized_diff, []
            if proposal.edits:
                exact_diff, exact_errors = _diff_from_exact_edits(root, target_file, proposal.edits)
                if not exact_errors:
                    return exact_diff, []
                return "", [
                    *[f"unified_diff: {error}" for error in diff_errors],
                    *exact_errors,
                ]
            # The patch text is what the text guardrails judge; the apply failure
            # is reported through ``edit_synthesis_errors``.
            return proposal.unified_diff, diff_errors
        if proposal.edits:
            return _diff_from_exact_edits(root, target_file, proposal.edits)
        return "", ["patch proposal must include either exact edits or a legacy unified_diff"]

    def _algorithmic_patch_check(self, patchset: PatchsetRecord) -> list[str]:
        if not self.require_algorithmic_patch:
            return []
        errors: list[str] = []
        for idx, patch in enumerate(patchset.patches, start=1):
            diff = patch.unified_diff
            if patch.component_name == "domain_def" and _is_default_threshold_only_patch(diff):
                errors.append(
                    f"Patch {idx} for domain_def only changes default threshold/constant assignments in "
                    "p7_domaindef_Create. This loop is configured for algorithmic C changes; "
                    "modify decision logic in p7_domaindef_ByPosteriorHeuristics or add a bounded "
                    "computed heuristic instead of tuning rt1/rt2/rt3/min_posterior defaults."
                )
            elif _is_parameter_only_patch(diff):
                errors.append(
                    f"Patch {idx} for {patch.component_name} appears to only tune literal constants or "
                    "default assignments. This run is configured for concrete C algorithm changes: add or "
                    "modify bounded decision logic, scoring/correction formulas, clipping rules derived from "
                    "runtime values, or DP recurrence behavior instead of changing static parameters."
                )
            errors.extend(_component_guardrail_errors(idx, patch))
        return errors

    def _git_apply_check(self, patchset: PatchsetRecord) -> list[str]:
        if not patchset.patches:
            return []
        check_dir = self.store.work_dir / "static_apply_checks" / (patchset.patchset_id or f"round_{patchset.round_idx:03d}")
        if check_dir.exists():
            shutil.rmtree(check_dir)
        check_dir.mkdir(parents=True, exist_ok=True)
        build_root = check_dir / "hmmer-src"
        try:
            run_hmmer_cli._copy_source(self.hmmer_src_root, build_root)
        except Exception as e:
            return [f"failed to prepare isolated static apply source: {type(e).__name__}: {e}"]

        patch_file = check_dir / "combined.patch"
        parts = [_normalize_patch_paths(patch.unified_diff, patch.c_target_file) for patch in patchset.patches]
        with patch_file.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write("\n".join(parts) + "\n")
        env = run_hmmer_cli._isolated_git_env(build_root)
        proc = subprocess.run(
            ["git", "apply", "--check", "--recount", str(patch_file)],
            cwd=str(build_root),
            capture_output=True,
            text=True,
            env=env,
        )
        if proc.returncode == 0:
            applied = subprocess.run(
                ["git", "apply", "--recount", str(patch_file)],
                cwd=str(build_root),
                capture_output=True,
                text=True,
                env=env,
            )
            if applied.returncode != 0:
                return [
                    "git apply failed after a successful check: "
                    f"{(applied.stderr or applied.stdout or '').strip()[-2000:]}"
                ]
            island_errors: list[str] = []
            for patch in patchset.patches:
                contract = ALL_C_KERNEL_CONTRACTS.get(patch.component_name)
                if contract is None or not contract.c_allow_translation_unit:
                    continue
                source = _resolve_patch_source(build_root, patch.c_target_file)
                island_errors.extend(
                    _translation_unit_guardrail_errors(
                        source,
                        component_name=patch.component_name,
                        frozen_exports=set(contract.c_target_functions or []),
                    )
                )
            return island_errors
        stderr = (proc.stderr or "")[-2000:]
        stdout = (proc.stdout or "")[-1000:]
        components = ",".join(patch.component_name for patch in patchset.patches)
        return [
            f"git apply --check failed for {components} patchset on isolated copied HMMER source: "
            f"stdout={stdout!r}; stderr={stderr!r}"
        ]

    def _write_legacy_pending(self, patchset: PatchsetRecord) -> None:
        mods = [
            {"contract_name": p.component_name, "python_payload": None, "c_patch": p.unified_diff}
            for p in patchset.patches
        ]
        write_pending(self.store.run_dir, patchset.round_idx, mods)

    def _write_legacy_full_eval(self, round_idx: int, result: EvalResult, record: EvaluationRecord) -> None:
        if record.stage != "full":
            return
        rdir = round_dir(self.store.run_dir, round_idx)
        data = result.model_dump(mode="json")
        data.setdefault("meta", {})
        data["meta"].update({"c_patch_loop_eval_id": record.eval_id, "c_patch_loop_patchset_id": record.patchset_id})
        (rdir / "full_eval.json").write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    def commit_evaluation(self, patchset: PatchsetRecord, record: EvaluationRecord) -> None:
        """Publish one outer-loop winner after its complete inner budget."""
        if record.stage != "full" or record.status != "ok":
            raise ValueError("only a successful Full evaluation can be committed")
        if record.patchset_id != patchset.patchset_id:
            raise ValueError("evaluation/patchset mismatch during outer commit")
        self._update_indexes(patchset, record, baseline=False)
        result = dict(record.eval_result or {})
        meta = dict(result.get("meta") or {})
        meta.update({
            "c_patch_loop_eval_id": record.eval_id,
            "c_patch_loop_patchset_id": patchset.patchset_id,
            "hierarchical_outer_commit": True,
        })
        result["meta"] = meta
        mods = [mod.model_dump(mode="json") for mod in _mods_from_patchset(patchset)]
        write_accepted(self.store.run_dir, mods, result)
        self._write_legacy_full_eval(
            patchset.round_idx,
            EvalResult.model_validate(result),
            record,
        )
        # Second -- and last -- post-hoc profiling trigger point: this candidate has
        # just passed the acceptance gate and is now the committed incumbent, so its
        # stage breakdown is what the next outer round should plan against.
        self.run_posthoc_profile(patchset)

    # ------------------------------------------------------------------
    # Post-hoc stage profiling
    #
    # Trigger points, and only these two:
    #   * the round-000 stock baseline  -- evaluate_stock_baseline();
    #   * a patchset that just passed the acceptance gate -- commit_evaluation().
    #
    # Both sit outside the inner trial loop, so no trial ever pays the
    # instrumentation overhead. The pass is a *separate evaluation*: it recompiles
    # HMMER with the p7_evo_prof stage timers compiled in and runs the same patchset
    # through that build. The instrumented run's cost_s is inflated by the timers and
    # is deliberately discarded; only the stage breakdown is kept.
    # ------------------------------------------------------------------

    def _posthoc_profile_mode(self) -> str | None:
        """Resolve the configured policy to a concrete backend, or ``None`` to skip."""
        mode = self.posthoc_profile
        if mode == "off":
            return None
        if mode == "auto":
            return self.eval_backend
        return mode

    def run_posthoc_profile(self, patchset: PatchsetRecord) -> EvaluationRecord | None:
        """Run and store one profiling evaluation of ``patchset``.

        Returns the stored profiling record, or ``None`` when profiling is disabled
        or could not be completed. Never raises: the profiling pass is evidence, and
        losing it must not fail a round that already passed the acceptance gate.
        """
        mode = self._posthoc_profile_mode()
        if mode is None:
            logger.info(
                "posthoc profile skipped for patchset %s: policy=%r resolves to no backend",
                patchset.patchset_id,
                self.posthoc_profile,
            )
            return None
        try:
            record = self._run_profiling_eval(
                patchset, keep_work=bool(getattr(self, "keep_work", False))
            )
        except Exception as exc:  # noqa: BLE001 -- profiling must fail open
            logger.warning(
                "posthoc profile of patchset %s failed to run: %s: %s",
                patchset.patchset_id,
                type(exc).__name__,
                exc,
            )
            _record_profile_failure(self.store, patchset, f"{type(exc).__name__}: {exc}")
            return None

        stage_timing = _profile_stage_timing(record)
        if not stage_timing or not stage_timing.get("stage_time_available"):
            # Name the cause, not just the symptom. The aggregate profile now
            # carries the per-query instrumented-profile diagnostic ("p7_evo_prof.txt
            # not present", "no recognised stage lines", ...), and that is what makes
            # a failed profiling pass repairable instead of a dead end.
            diagnostic = _profile_failure_diagnostic(record)
            reason = (
                "profiling build produced no stage times "
                f"(status={record.status}, time_source={(stage_timing or {}).get('time_source')}"
                + (f", instrumented_profile={json.dumps(diagnostic, sort_keys=True)}" if diagnostic else "")
                + ")"
            )
            logger.warning("posthoc profile of patchset %s unusable: %s", patchset.patchset_id, reason)
            _record_profile_failure(self.store, patchset, reason)
            return None

        logger.info(
            "posthoc profile of patchset %s recorded: eval=%s total=%.4fs stage_time_fraction=%s",
            patchset.patchset_id,
            record.eval_id,
            float(stage_timing.get("measured_stage_total_s") or 0.0),
            json.dumps(stage_timing.get("stage_time_fraction") or {}, sort_keys=True),
        )
        # Merge into this round's experience context. This is the only sink: the
        # digest is regenerated from the experience store, so no intermediate state
        # tracking and no temporary file cache are introduced.
        append_and_compress(
            self.store,
            experience_from_evaluation(
                patchset,
                record,
                lesson=_posthoc_profile_lesson(patchset, record, stage_timing),
            ),
        )
        return record

    def _run_profiling_eval(self, patchset: PatchsetRecord, *, keep_work: bool = False) -> EvaluationRecord:
        """Evaluate ``patchset`` once with the instrumented profiling build.

        ``keep_work`` is forwarded because the stage report (``p7_evo_prof.txt``) is
        written into the per-query work directory and that directory is deleted on
        success when ``keep_work`` is false. Without forwarding it, an operator who
        asked to keep work still could not inspect the file the pass exists to
        produce.
        """
        cfg: Any
        if not patchset.patches:
            cfg = AssembledConfig()
        else:
            cfg = assemble(_mods_from_patchset(patchset), hmmer_src_root=self.hmmer_src_root)
            if getattr(cfg, "errors", None):
                raise RuntimeError(
                    "failed to reconstruct patchset for profiling: " + "; ".join(cfg.errors)
                )
        return self.evaluate(
            patchset,
            cfg,
            stage="full",
            attempt_kind="posthoc_profile",
            # The profiling run is evidence, not a decision: it must not move any
            # index, and its experience record is authored by run_posthoc_profile()
            # with a profiling-specific lesson.
            commit_indexes=False,
            write_legacy_result=False,
            record_experience=False,
            profiling=True,
            keep_work=keep_work,
        )

    def _write_baseline_index(self, name: str, summary: dict[str, Any]) -> None:
        """Write a baseline index once, and never let a re-measurement replace it.

        The authoritative baseline must be a single, stable number: every later
        verdict is a ratio against it, so replacing it retroactively changes what
        those verdicts meant. A second baseline measurement therefore cannot take
        the index over; it is recorded under ``baseline_recalibration*`` instead,
        and the divergence between the two is exactly the drift signal worth
        reading. Re-writing the same ``eval_id`` (a resume that re-published the
        same evaluation) is a no-op and is allowed.
        """
        existing = self.store.read_index(name)
        existing = existing if isinstance(existing, dict) else None
        incoming = summary.get("eval_id")
        if existing and existing.get("eval_id") not in (None, incoming):
            recalibration_name = (
                "baseline_recalibration"
                if name == "baseline"
                else name.replace("baseline", "baseline_recalibration", 1)
            )
            previous = self.store.read_index(recalibration_name)
            history = previous if isinstance(previous, list) else ([previous] if isinstance(previous, dict) else [])
            history.append(summary)
            self.store.write_index(recalibration_name, history)
            logger.warning(
                "baseline index %r already holds eval_id=%s; refusing to overwrite it with %s. "
                "The re-measurement was recorded under %r instead -- a changed baseline invalidates "
                "every verdict computed against the old one.",
                name,
                existing.get("eval_id"),
                incoming,
                recalibration_name,
            )
            return
        self.store.write_index(name, summary)

    def _update_indexes(self, patchset: PatchsetRecord, record: EvaluationRecord, *, baseline: bool = False) -> None:
        result = record.eval_result or {}
        summary = {
            "patchset_id": patchset.patchset_id,
            "eval_id": record.eval_id,
            "round_idx": patchset.round_idx,
            "stage": record.stage,
            "tp": result.get("tp", 0),
            "fp": result.get("fp", 0),
            "tp_fp_ratio": result.get("tp_fp_ratio", 0.0),
            "cost_s": result.get("cost_s", 0.0),
            "decision": record.decision,
            "baseline": baseline,
            "eval_config": dict(record.eval_config or {}),
        }
        paired = (record.metric_delta or {}).get("paired_confirmation")
        if paired:
            summary["paired_confirmation"] = paired
        stage = record.stage
        if baseline:
            # The baseline is established ONCE per run and is then immutable.
            # ``write_index`` is an unconditional overwrite, so without this guard a
            # later re-baseline -- which the read-side cache check triggers whenever
            # the reuse fingerprint changes (a different core count, a different
            # database, a different eval_repeats) -- silently replaced the number
            # every earlier verdict had been computed against. The 2026-10-09 audit
            # found exactly that: a run whose baseline had been overwritten by a
            # profiling-era re-measurement.
            #
            # A re-measurement is not discarded, it is routed to a separate
            # calibration record, so the authoritative index keeps exactly one value
            # and the recalibration stays auditable.
            self._write_baseline_index(f"baseline_{stage}", summary)
            if stage == "full":
                self._write_baseline_index("baseline", summary)

        self.store.write_index(f"current_speed_best_{stage}", summary)
        self.store.write_index(f"current_best_{stage}", summary)
        lineage_name = f"accepted_lineage_{stage}"
        lineage = self.store.read_index(lineage_name) or []
        lineage.append(summary)
        self.store.write_index(lineage_name, lineage)
        leaderboard_name = f"leaderboard_{stage}"
        leaderboard = self.store.read_index(leaderboard_name) or []
        leaderboard.append(summary)
        leaderboard = sorted(leaderboard, key=_leaderboard_key)
        self.store.write_index(leaderboard_name, leaderboard[:50])

        if stage == "full":
            self.store.write_index("current_speed_best", summary)
            self.store.write_index("current_best", summary)
            legacy_lineage = self.store.read_index("accepted_lineage") or []
            legacy_lineage.append(summary)
            self.store.write_index("accepted_lineage", legacy_lineage)
            legacy_leaderboard = self.store.read_index("leaderboard") or []
            legacy_leaderboard.append(summary)
            legacy_leaderboard = sorted(legacy_leaderboard, key=_leaderboard_key)
            self.store.write_index("leaderboard", legacy_leaderboard[:50])


def _profile_stage_timing(record: EvaluationRecord) -> dict[str, Any]:
    """Pull the stage_timing block out of a stored evaluation's profile."""
    profile = ((record.eval_result or {}).get("meta") or {}).get("profile") or {}
    timing = profile.get("stage_timing") if isinstance(profile, dict) else None
    return timing if isinstance(timing, dict) else {}


def _profile_failure_diagnostic(record: EvaluationRecord) -> dict[str, Any] | None:
    """The instrumented-profile diagnostic explaining *why* no stage times appeared.

    Reads the aggregate stage_timing block that ``_aggregate_profiles`` writes
    (``instrumented_profile``), falling back to the first per-query diagnostic for
    records produced before that field existed.
    """
    timing = _profile_stage_timing(record)
    diagnostic = timing.get("instrumented_profile")
    if isinstance(diagnostic, dict) and diagnostic:
        return diagnostic
    for item in timing.get("per_query_instrumented_profile") or []:
        candidate = item.get("instrumented_profile") if isinstance(item, dict) else None
        if isinstance(candidate, dict) and candidate:
            return candidate
    return None


def _posthoc_profile_lesson(
    patchset: PatchsetRecord,
    record: EvaluationRecord,
    stage_timing: dict[str, Any],
) -> str:
    """State the stage attribution in the terms the outer planner acts on."""
    is_stock = not patchset.patches
    label = "stock baseline" if is_stock else (patchset.patches[-1].component_name or "incumbent")
    fraction = stage_timing.get("stage_time_fraction") or {}
    ordered = sorted(fraction.items(), key=lambda kv: float(kv[1]), reverse=True)
    share = ", ".join(f"{name}={float(value):.3f}" for name, value in ordered[:4]) or "n/a"
    total = stage_timing.get("measured_stage_total_s")
    return (
        f"Post-hoc stage profile of the {label} (instrumented build, profiling eval "
        f"{record.eval_id}): measured stage total {total}s; time share {share}. "
        "This is the durable 'which pipeline stage the speed actually goes to' signal "
        "for the next outer plan. Stage time carries instrumentation overhead, so use "
        "it for attribution only -- never compare it to cost_s or gate acceptance on it."
    )


def _record_profile_failure(store: LoopStore, patchset: PatchsetRecord, reason: str) -> None:
    """Record a failed profiling pass as a workflow lesson, never as a fatal error.

    Uses the experience store -- the same sink the digest is built from -- rather
    than inventing a side channel, so a broken profiling build shows up as an
    avoid/repair hint instead of silently vanishing.
    """
    component = patchset.patches[-1].component_name if patchset.patches else None
    append_and_compress(
        store,
        workflow_experience(
            outcome="posthoc_profile_failed",
            lesson=(
                "Post-hoc stage profiling did not produce usable stage times: "
                f"{reason}. The round result stands; treat the stage attribution as "
                "unavailable rather than assuming any stage is free."
            ),
            component_name=component,
            source_patchset_ids=[patchset.patchset_id] if patchset.patchset_id else [],
            metric_summary={"stage": "posthoc_profile", "failure_stage": "posthoc_profile"},
            retrieval_tags=["profiling", "provenance"],
        ),
    )


def _leaderboard_key(item: dict[str, Any]) -> tuple[float, int, int]:
    return (float(item.get("cost_s", 0.0)), -int(item.get("tp", 0)), int(item.get("fp", 0)))



def _diff_from_exact_edits(source_root: Path, target_file: str, edits: list[ExactEditBlock]) -> tuple[str, list[str]]:
    source_file = _resolve_patch_source(source_root, target_file)
    if not source_file.exists():
        return "", [f"exact edit source file not found: {source_file}"]
    original_text = source_file.read_text(encoding="utf-8", errors="replace")
    modified_text = original_text
    errors: list[str] = []
    for idx, edit in enumerate(edits, start=1):
        old = _strip_excerpt_line_prefixes(edit.old)
        new = _strip_excerpt_line_prefixes(edit.new)
        if not old:
            errors.append(f"exact edit {idx} has empty old block")
            continue
        count = modified_text.count(old)
        if count != 1:
            errors.append(f"exact edit {idx} old block matched {count} times in {target_file}; expected exactly once")
            continue
        if old == new:
            errors.append(f"exact edit {idx} old and new blocks are identical")
            continue
        modified_text = modified_text.replace(old, new, 1)
    if errors:
        return "", errors
    if modified_text == original_text:
        return "", ["exact edits produced no source change"]
    rel = Path(target_file.replace("\\", "/").lstrip("./"))
    if not rel.as_posix().startswith("easel/"):
        rel = Path("src") / rel
    diff = "".join(difflib.unified_diff(
        original_text.splitlines(keepends=True),
        modified_text.splitlines(keepends=True),
        fromfile=f"a/{rel.as_posix()}",
        tofile=f"b/{rel.as_posix()}",
    ))
    return diff, []


def _normalize_unified_diff(
    source_root: Path,
    target_file: str,
    unified_diff: str,
) -> tuple[str, list[str]]:
    """Validate an AI-generated unified_diff and regenerate it with correct line numbers.

    The AI may emit a diff whose hunk coordinates reference a wrong source
    version (for example stock line numbers after a parent patchset has already
    shifted the file). Instead of passing that diff through verbatim, apply it
    to an isolated copy of *source_root* and produce a canonical diff relative
    to the actual tree. If the AI diff cannot be applied at all, return the
    git-apply error so the proposal can be repaired.
    """
    if not unified_diff.strip():
        return "", ["empty unified_diff"]

    tmpdir = Path(tempfile.mkdtemp(prefix="diff_normalize_"))
    try:
        build_root = tmpdir / "hmmer-src"
        run_hmmer_cli._copy_source(source_root, build_root)

        patch = PatchRecord(
            component_name="",
            c_target_file=target_file,
            unified_diff=unified_diff,
        )
        errors = _apply_patch_records(build_root, [patch])
        if errors:
            return "", errors

        canonical = _stock_relative_diff(source_root, build_root, target_file)
        if not canonical.strip():
            return "", ["normalized unified_diff produced no source change"]
        return canonical, []
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def _strip_excerpt_line_prefixes(text: str) -> str:
    lines = text.splitlines(keepends=True)
    if not lines:
        return text
    stripped: list[str] = []
    changed = False
    pattern = re.compile(r"^\s*\d+\t")
    for line in lines:
        new_line = pattern.sub("", line, count=1)
        changed = changed or new_line != line
        stripped.append(new_line)
    return "".join(stripped) if changed else text



def _resolve_patch_source(source_root: Path, target_file: str) -> Path:
    target = target_file.replace("\\", "/").strip().lstrip("./")
    # HMMER uses a split checkout: core sources live in src/, while Easel
    # sources (for example easel/esl_sqio_ascii.c) live beside src/.
    candidates = [source_root / target, source_root / "src" / target]
    if source_root.name == "src":
        candidates.extend([source_root.parent / target, source_root.parent / "src" / target])
    if target.startswith("easel/"):
        candidates.append(source_root / target.removeprefix("easel/"))
        if source_root.name == "src":
            candidates.append(source_root.parent / target)
    candidates.extend([
        source_root / "src" / Path(target).name,
        source_root / Path(target).name,
    ])
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return source_root / target if target.startswith("easel/") else source_root / "src" / target


def _apply_patch_records(root: Path, patches: list[PatchRecord]) -> list[str]:
    """Apply stock-relative or incremental records to an isolated source tree."""
    if not patches:
        return []
    patch_file = root.parent / "materialize.patch"
    parts = [
        _normalize_patch_paths(item.unified_diff, item.c_target_file)
        for item in patches
        if item.unified_diff.strip()
    ]
    if not parts:
        return ["patchset contains no non-empty C diff"]
    with patch_file.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("\n".join(parts) + "\n")
    env = run_hmmer_cli._isolated_git_env(root)
    check = subprocess.run(
        ["git", "apply", "--check", "--recount", str(patch_file)],
        cwd=str(root),
        capture_output=True,
        text=True,
        env=env,
    )
    if check.returncode != 0:
        return [
            "git apply --check failed while materializing source: "
            f"{(check.stderr or check.stdout or '').strip()[-2000:]}"
        ]
    applied = subprocess.run(
        ["git", "apply", "--recount", str(patch_file)],
        cwd=str(root),
        capture_output=True,
        text=True,
        env=env,
    )
    if applied.returncode != 0:
        return [
            "git apply failed while materializing source: "
            f"{(applied.stderr or applied.stdout or '').strip()[-2000:]}"
        ]
    return []


def _stock_relative_diff(stock_root: Path, candidate_root: Path, target_file: str) -> str:
    """Return a canonical unified diff from stock to one materialized file."""
    stock_file = _resolve_patch_source(stock_root, target_file)
    candidate_file = _resolve_patch_source(candidate_root, target_file)
    if not stock_file.is_file() or not candidate_file.is_file():
        return ""
    original = stock_file.read_text(encoding="utf-8", errors="replace")
    modified = candidate_file.read_text(encoding="utf-8", errors="replace")
    target = Path(target_file.replace("\\", "/").lstrip("./"))
    if not target.as_posix().startswith("easel/"):
        target = Path("src") / target
    return "".join(
        difflib.unified_diff(
            original.splitlines(keepends=True),
            modified.splitlines(keepends=True),
            fromfile=f"a/{target.as_posix()}",
            tofile=f"b/{target.as_posix()}",
        )
    )


def _translation_unit_guardrail_errors(
    source: Path,
    *,
    component_name: str,
    frozen_exports: set[str],
) -> list[str]:
    """Reject unsafe state/I/O additions in an otherwise open algorithm island."""
    if not source.is_file():
        return [f"algorithm island source is missing after patch application: {source}"]
    text = source.read_text(encoding="utf-8", errors="replace")
    errors: list[str] = []
    forbidden_headers = (
        "stdio.h",
        "unistd.h",
        "pthread.h",
        "mpi.h",
        "fcntl.h",
        "dlfcn.h",
        "sys/",
    )
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#include") and any(name in stripped for name in forbidden_headers):
            errors.append(
                f"{component_name} may not add process, file, thread, MPI, or dynamic-loader APIs: {stripped}"
            )

    # Strip comments and scan top-level semicolon declarations. Function
    # prototypes/helpers contain '(' and are allowed; immutable lookup tables
    # must be explicitly const. State belongs in the prewired opaque object.
    scrubbed = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    scrubbed = re.sub(r"//.*", "", scrubbed)
    depth = 0
    statement: list[str] = []
    for raw_line in scrubbed.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if depth == 0:
            statement.append(line)
        depth += raw_line.count("{") - raw_line.count("}")
        if depth == 0 and statement:
            joined = " ".join(statement).strip()
            statement = []
            if not joined.endswith(";") or "(" in joined:
                continue
            if joined.startswith(("typedef ", "struct ", "enum ", "union ", "extern ")):
                continue
            if " const " in f" {joined} " or joined.startswith("const "):
                continue
            errors.append(
                f"{component_name} declares mutable file-scope state; use its query/worker-local opaque state: "
                f"{joined[:180]}"
            )

    for symbol in sorted(frozen_exports):
        if not re.search(rf"\b{re.escape(symbol)}\s*\(", text):
            errors.append(f"{component_name} removed frozen export {symbol}")
    return errors



def _safe_hash(path: Path) -> str:
    return hash_file(path) if path.is_file() else ""


def _manifest_file_identity_matches(item: dict[str, Any]) -> bool:
    try:
        stat = Path(str(item["query_path"])).stat()
    except (OSError, KeyError, TypeError):
        return False
    return (
        stat.st_size == item.get("size")
        and stat.st_dev == item.get("device")
        and stat.st_ino == item.get("inode")
        and stat.st_mtime_ns == item.get("mtime_ns")
    )


def _fasta_query_id(path: Path) -> str:
    if not path.is_file():
        return path.stem
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith(">"):
            return line[1:].strip().split()[0]
    return path.stem


def _normalize_binary_hashes(value: dict[str, Any]) -> dict[str, str]:
    return {
        "jackhmmer": str(value.get("jackhmmer") or value.get("jackhmmer_sha256") or ""),
        "hmmsearch": str(value.get("hmmsearch") or value.get("hmmsearch_sha256") or ""),
    }


def _unique_strings(values: list[Any]) -> list[str]:
    return list(dict.fromkeys(str(value) for value in values if value))


def _completion_artifact_errors(completion: dict[str, Any]) -> list[str]:
    artifacts = completion.get("artifacts") or {}
    table = artifacts.get("hmmsearch_tbl") if isinstance(artifacts, dict) else None
    if (
        not isinstance(table, dict)
        or table.get("exists") is not True
        or int(table.get("bytes") or table.get("size") or 0) <= 0
        or not str(table.get("sha256") or "")
    ):
        return [
            f"query {int(completion.get('query_idx') or 0)} artifacts are incomplete: hmmsearch_tbl"
        ]
    return []


def _eval_result_from_completion(completion: QueryCompletionRecord, *, stage: str) -> EvalResult:
    result = EvalResult(
        stage=stage,
        status=completion.status,
        tp=completion.tp,
        fp=completion.fp,
        cost_s=completion.cost_s,
        n_seqs=1,
        iterations=completion.iterations,
        errors=list(completion.errors),
        meta={
            "query_id": completion.query_id,
            "query_path": completion.query_path,
            "profile": {"artifacts": dict(completion.artifacts)},
            "resumed_from_query_completion_id": completion.query_completion_id,
        },
    )
    result.ratio()
    return result


def _attempt_evidence(result: EvalResult) -> dict[str, Any]:
    raw = (result.meta or {}).get("evaluation_attempt") or (result.meta or {}).get("_evaluation_attempt") or {}
    return dict(raw) if isinstance(raw, dict) else {}


def _attempt_evidence_errors(evidence: dict[str, Any]) -> list[str]:
    return [str(item) for item in (evidence.get("evidence_errors") or [])]


def _merge_attempt_evidence(
    evidences: list[dict[str, Any]],
    *,
    selected: dict[str, Any],
) -> dict[str, Any]:
    return {
        "checkpoint_enabled": any(bool(item.get("checkpoint_enabled")) for item in evidences),
        "attempt_ids": _unique_strings([
            attempt_id
            for item in evidences
            for attempt_id in (item.get("attempt_ids") or [])
        ]),
        "primary_attempt_id": selected.get("primary_attempt_id"),
        "evaluation_fingerprint": selected.get("evaluation_fingerprint"),
        "query_completion_ids": _unique_strings([
            completion_id
            for item in evidences
            for completion_id in (item.get("query_completion_ids") or [])
        ]),
        "requested_queries": selected.get("requested_queries"),
        "completed_queries": selected.get("completed_queries"),
        "resumed_queries": sum(int(item.get("resumed_queries") or 0) for item in evidences),
        "timing_protocol_fingerprint": selected.get("timing_protocol_fingerprint"),
        "binary_sha256": dict(selected.get("binary_sha256") or {}),
        "evidence_errors": [
            str(error)
            for item in evidences
            for error in (item.get("evidence_errors") or [])
        ],
    }


def _median_eval_result(results: list[EvalResult]) -> EvalResult:
    """Collapse repeated evals to one result using median runtime for decisions."""
    if not results:
        raise ValueError("at least one EvalResult is required")
    best = min(results, key=lambda item: abs(item.cost_s - median([r.cost_s for r in results])))
    result = best.model_copy(deep=True)
    costs = [r.cost_s for r in results]
    statuses = [r.status for r in results]
    selected_repeat_idx = results.index(best) + 1
    result.cost_s = round(float(median(costs)), 4)
    if any(r.status != "ok" for r in results):
        result.status = "ok" if all(status == "ok" for status in statuses) else "run_error"
    repeat_summaries = [
        {
            "repeat_idx": idx,
            "status": r.status,
            "tp": r.tp,
            "fp": r.fp,
            "tp_fp_ratio": r.tp_fp_ratio,
            "cost_s": r.cost_s,
            "profile": _compact_repeat_profile((r.meta or {}).get("profile")),
            "errors": list(r.errors),
        }
        for idx, r in enumerate(results, start=1)
    ]
    result.meta = dict(result.meta)
    result.meta["timing"] = {
        "eval_repeats": len(results),
        "cost_s_values": costs,
        "median_cost_s": result.cost_s,
        "selected_repeat_idx": selected_repeat_idx,
    }
    result.meta["repeat_results"] = repeat_summaries
    selected_profile = result.meta.get("profile")
    if isinstance(selected_profile, dict):
        result.meta["profile"] = _repeat_profile(
            selected_profile,
            repeat_idx=selected_repeat_idx,
            eval_repeats=len(results),
            repeat_summary=repeat_summaries,
            repeat_aggregate=_repeat_aggregate(repeat_summaries),
        )
    result.errors = [error for r in results for error in r.errors]
    if result.errors and result.meta.get("errors") is None:
        result.meta["errors"] = list(result.errors)
    return result


def _compact_paired_result(result: EvalResult) -> dict[str, Any]:
    return {
        "stage": result.stage,
        "status": result.status,
        "tp": result.tp,
        "fp": result.fp,
        "tp_fp_ratio": result.tp_fp_ratio,
        "cost_s": result.cost_s,
        "timing": (result.meta or {}).get("timing") or {},
        "profile": _compact_repeat_profile((result.meta or {}).get("profile")),
        "errors": list(result.errors),
    }



def _repeat_profile(
    profile: dict[str, Any],
    *,
    repeat_idx: int,
    eval_repeats: int,
    repeat_summary: list[dict[str, Any]] | None = None,
    repeat_aggregate: dict[str, Any] | None = None,
) -> dict[str, Any]:
    out = dict(profile)
    out["repeat_strategy"] = "single" if eval_repeats == 1 else "median_nearest_cost_s"
    out["selected_repeat_idx"] = repeat_idx
    out["eval_repeats"] = eval_repeats
    if repeat_summary is not None:
        out["repeat_summary"] = repeat_summary
    if repeat_aggregate is not None:
        out["repeat_aggregate"] = repeat_aggregate
    return out


def _compact_repeat_profile(profile: Any) -> dict[str, Any]:
    if not isinstance(profile, dict):
        return {}
    return {
        "schema_version": profile.get("schema_version"),
        "dominant_command": profile.get("dominant_command"),
        "command_wall_s": profile.get("command_wall_s") or {},
        "command_wall_fraction": profile.get("command_wall_fraction") or {},
        "top_slowest_queries": (profile.get("top_slowest_queries") or [])[:5],
    }


def _repeat_aggregate(repeat_summaries: list[dict[str, Any]]) -> dict[str, Any]:
    command_totals: dict[str, float] = {}
    dominant_counts: dict[str, int] = {}
    for summary in repeat_summaries:
        profile = summary.get("profile") or {}
        dominant = profile.get("dominant_command")
        if dominant:
            dominant_counts[str(dominant)] = dominant_counts.get(str(dominant), 0) + 1
        for name, wall_s in (profile.get("command_wall_s") or {}).items():
            command_totals[str(name)] = round(command_totals.get(str(name), 0.0) + float(wall_s or 0.0), 4)
    return {
        "repeat_count": len(repeat_summaries),
        "dominant_command_counts": dict(sorted(dominant_counts.items())),
        "command_wall_s_total": dict(sorted(command_totals.items())),
    }


def _stock_source_hashes(root: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for contract in ALL_C_KERNEL_CONTRACTS.values():
        target = contract.c_target_file or ""
        if not target:
            continue
        for candidate in (
            root / target,
            root / "src" / target,
            root / "src" / Path(target).name,
            root / Path(target).name,
        ):
            if candidate.exists():
                out[str(candidate)] = hash_file(candidate)
                break
    return out


def _stock_binary_fingerprint(root: Path) -> dict[str, Any]:
    prebuilt = os.getenv("SEQ_ALIGN_HMMER_BIN_DIR")
    binary_root = Path(prebuilt) if prebuilt else root / "src"
    return {
        name: _file_info(binary_root / name, sha256=True)
        for name in ("jackhmmer", "hmmsearch")
    }


_DOCKER_RUNTIME_BINARY_ROOT = "/opt/hmmer-src/src"
_DOCKER_RUNTIME_BINARY_NAMES = ("jackhmmer", "hmmsearch")
_DOCKER_RUNTIME_IDENTITY_SCHEMA = "seq_align_runtime_identity.v1"


def _slurm_docker_runtime_identity(store: LoopStore) -> dict[str, Any]:
    image_ref = os.getenv("SEQ_ALIGN_SLURM_DOCKER_IMAGE", "").strip()
    match = re.fullmatch(r"\S+@sha256:([0-9a-fA-F]{64})", image_ref)
    if match is None:
        raise RuntimeIdentityUnavailable(
            "SEQ_ALIGN_SLURM_DOCKER_IMAGE must be pinned by a full @sha256 digest"
        )
    image_digest = f"sha256:{match.group(1).lower()}"
    cache_path = (
        store.root
        / "runtime_identities"
        / f"{match.group(1).lower()}.json"
    )
    if cache_path.is_file():
        cached = _load_cached_docker_runtime_identity(cache_path, image_digest=image_digest)
        return _docker_runtime_identity_for_image(cached, image_ref=image_ref)

    binary_paths = {
        name: f"{_DOCKER_RUNTIME_BINARY_ROOT}/{name}"
        for name in _DOCKER_RUNTIME_BINARY_NAMES
    }
    command = [
        "docker",
        "run",
        "--pull=never",
        "--rm",
        "--network",
        "none",
        "--read-only",
        "--entrypoint",
        "sha256sum",
        image_ref,
        *(binary_paths[name] for name in _DOCKER_RUNTIME_BINARY_NAMES),
    ]
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=60,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeIdentityUnavailable(
            f"could not probe pinned Slurm Docker image: {type(exc).__name__}"
        ) from exc
    if completed.returncode != 0:
        raise RuntimeIdentityUnavailable(
            "could not probe pinned Slurm Docker image: docker run failed "
            f"with exit status {completed.returncode}"
        )
    binaries = _parse_docker_binary_hashes(
        completed.stdout,
        binary_paths=binary_paths,
    )
    cached = {
        "schema_version": _DOCKER_RUNTIME_IDENTITY_SCHEMA,
        "runtime": "slurm_docker",
        "image_digest": image_digest,
        "binary_root": _DOCKER_RUNTIME_BINARY_ROOT,
        "binaries": binaries,
        "created_at": utc_now(),
    }
    cached["runtime_identity_id"] = record_hash(
        cached,
        id_field="runtime_identity_id",
    )
    try:
        write_json(cache_path, cached)
    except (OSError, ValueError, TypeError, FileExistsError) as exc:
        raise RuntimeIdentityUnavailable(
            f"could not safely cache Slurm Docker runtime identity: {type(exc).__name__}"
        ) from exc
    return _docker_runtime_identity_for_image(cached, image_ref=image_ref)


def _load_cached_docker_runtime_identity(
    cache_path: Path,
    *,
    image_digest: str,
) -> dict[str, Any]:
    try:
        cached = load_json(cache_path)
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        raise RuntimeIdentityUnavailable("cached Slurm Docker runtime identity is unreadable") from exc
    if (
        not isinstance(cached, dict)
        or cached.get("schema_version") != _DOCKER_RUNTIME_IDENTITY_SCHEMA
        or cached.get("runtime") != "slurm_docker"
        or cached.get("image_digest") != image_digest
        or cached.get("binary_root") != _DOCKER_RUNTIME_BINARY_ROOT
        or record_hash(cached, id_field="runtime_identity_id")
        != cached.get("runtime_identity_id")
    ):
        raise RuntimeIdentityUnavailable("cached Slurm Docker runtime identity failed validation")
    _validated_runtime_binary_hashes(cached.get("binaries"))
    return cached


def _parse_docker_binary_hashes(
    output: str,
    *,
    binary_paths: dict[str, str],
) -> dict[str, dict[str, str]]:
    rows = [line.strip() for line in output.splitlines() if line.strip()]
    if len(rows) != len(binary_paths):
        raise RuntimeIdentityUnavailable("Docker runtime identity probe returned incomplete output")
    by_path: dict[str, str] = {}
    for row in rows:
        parts = row.split(maxsplit=1)
        if len(parts) != 2:
            raise RuntimeIdentityUnavailable("Docker runtime identity probe returned malformed output")
        digest, path = parts
        path = path.lstrip("* ")
        if re.fullmatch(r"[0-9a-fA-F]{64}", digest) is None or path in by_path:
            raise RuntimeIdentityUnavailable("Docker runtime identity probe returned malformed hashes")
        by_path[path] = digest.lower()
    if set(by_path) != set(binary_paths.values()):
        raise RuntimeIdentityUnavailable("Docker runtime identity probe returned unexpected paths")
    return {
        name: {"path": path, "sha256": by_path[path]}
        for name, path in binary_paths.items()
    }


def _validated_runtime_binary_hashes(value: Any) -> dict[str, dict[str, str]]:
    if not isinstance(value, dict) or set(value) != set(_DOCKER_RUNTIME_BINARY_NAMES):
        raise RuntimeIdentityUnavailable("runtime identity does not contain both HMMER binaries")
    validated: dict[str, dict[str, str]] = {}
    for name in _DOCKER_RUNTIME_BINARY_NAMES:
        item = value.get(name)
        expected_path = f"{_DOCKER_RUNTIME_BINARY_ROOT}/{name}"
        if (
            not isinstance(item, dict)
            or item.get("path") != expected_path
            or re.fullmatch(r"[0-9a-f]{64}", str(item.get("sha256") or "")) is None
        ):
            raise RuntimeIdentityUnavailable(f"runtime identity for {name} is invalid")
        validated[name] = {"path": expected_path, "sha256": str(item["sha256"])}
    return validated


def _docker_runtime_identity_for_image(
    cached: dict[str, Any],
    *,
    image_ref: str,
) -> dict[str, Any]:
    # Keep cache bookkeeping out of the semantic identity.  In particular,
    # ``runtime_identity_id`` includes ``created_at`` and would make an
    # otherwise identical image produce a different reuse fingerprint after
    # a lost/rebuilt cache file.
    return {
        "schema_version": _DOCKER_RUNTIME_IDENTITY_SCHEMA,
        "backend": "slurm",
        "runtime": "docker",
        "image_ref": image_ref,
        "image_digest": cached["image_digest"],
        "binary_root": _DOCKER_RUNTIME_BINARY_ROOT,
        "binaries": _validated_runtime_binary_hashes(cached.get("binaries")),
    }


def _plan_diff_binding(
    *,
    proposal: PatchProposal,
    contract: Any,
    incremental_diff: str,
    source_root: Path,
    patch_plan: PatchPlanRecord | None,
    algorithm_plan: AlgorithmPlanRecord | None,
    requested_algorithm_plan_id: str | None,
) -> PlanDiffBindingRecord | None:
    """Bind the generated incremental diff to its approved plan artifacts.

    Historical patchsets without a patch plan retain their original replay
    behavior. New hierarchical proposals are checked against the incremental
    diff (not the stock-relative squashed diff), so inherited parent edits do
    not look like function drift by the current inner trial.
    """
    if not incremental_diff.strip():
        return None
    if patch_plan is None and algorithm_plan is None and not requested_algorithm_plan_id:
        return None

    planned_component = patch_plan.component_name if patch_plan is not None else (
        algorithm_plan.target_contracts[0]
        if algorithm_plan is not None and algorithm_plan.target_contracts
        else proposal.component_name
    )
    planned_contract = ALL_C_KERNEL_CONTRACTS.get(planned_component)
    allowed_file = (
        patch_plan.c_target_file
        if patch_plan is not None
        else (planned_contract.c_target_file if planned_contract is not None else contract.c_target_file)
    ) or ""
    inner_functions = set(
        patch_plan.c_target_functions
        if patch_plan is not None
        else (planned_contract.c_target_functions if planned_contract is not None else [])
    )
    outer_functions = set(algorithm_plan.target_functions if algorithm_plan is not None else [])
    errors: list[str] = []

    component_inner = patch_plan is None or proposal.component_name == patch_plan.component_name
    if not component_inner:
        errors.append(
            f"final patch component {proposal.component_name!r} differs from approved inner patch plan "
            f"component {patch_plan.component_name!r}"
        )
    if patch_plan is not None and patch_plan.validation_status != "ok":
        errors.append("final patch references a rejected inner patch plan")

    component_outer = True
    if requested_algorithm_plan_id and algorithm_plan is None:
        component_outer = False
        errors.append(
            f"approved outer algorithm plan {requested_algorithm_plan_id!r} could not be loaded"
        )
    if algorithm_plan is not None:
        if algorithm_plan.validation_status != "ok":
            component_outer = False
            errors.append("final patch references a rejected outer algorithm plan")
        if requested_algorithm_plan_id and algorithm_plan.algorithm_plan_id != requested_algorithm_plan_id:
            component_outer = False
            errors.append("final patch outer algorithm_plan_id does not match the approved record")
        if patch_plan is not None and patch_plan.algorithm_plan_id and (
            patch_plan.algorithm_plan_id != algorithm_plan.algorithm_plan_id
        ):
            component_outer = False
            errors.append("inner patch plan is linked to a different outer algorithm plan")
        if proposal.component_name not in algorithm_plan.target_contracts:
            component_outer = False
            errors.append(
                f"final patch component {proposal.component_name!r} is outside approved outer algorithm plan "
                f"contracts {algorithm_plan.target_contracts}"
            )
        if patch_plan is not None and not set(patch_plan.c_target_functions) <= outer_functions:
            component_outer = False
            errors.append("inner patch plan functions are outside the approved outer algorithm plan")

    footprint = extract_actual_diff_footprint(
        incremental_diff,
        allowed_file,
        hmmer_src_root=source_root,
    )
    file_matches = footprint.extraction_status == "ok"
    if not file_matches:
        errors.extend(
            f"could not bind final diff to approved file {allowed_file!r}: {error}"
            for error in footprint.errors
        )

    actual_functions = set(footprint.actual_functions)
    actual_inner = file_matches and _actual_functions_within_plan(
        actual_functions,
        inner_functions,
        planned_contract,
    )
    if file_matches and not actual_inner:
        errors.append(
            f"actual diff functions {sorted(actual_functions)} escape approved inner patch plan "
            f"functions {sorted(inner_functions)}"
        )

    actual_outer = component_outer and file_matches
    if algorithm_plan is not None and actual_outer:
        actual_outer = _actual_functions_within_plan(
            actual_functions,
            outer_functions,
            planned_contract,
        )
        if not actual_outer:
            errors.append(
                f"actual diff functions {sorted(actual_functions)} escape approved outer algorithm plan "
                f"functions {sorted(outer_functions)}"
            )

    containment = {
        "component_matches_inner_plan": component_inner,
        "component_within_outer_plan": component_outer,
        "file_matches_inner_plan": file_matches,
        "actual_functions_within_inner_plan": actual_inner,
        "actual_functions_within_outer_plan": actual_outer,
        "actual_within_final_proposal": component_inner and file_matches,
        "final_and_actual_within_inner_plan": component_inner and file_matches and actual_inner,
        "final_and_actual_within_outer_plan": component_outer and file_matches and actual_outer,
    }
    return PlanDiffBindingRecord(
        binding_status="ok" if not errors else "rejected",
        final_component_name=proposal.component_name,
        final_allowed_file=allowed_file,
        final_allowed_functions=sorted(inner_functions),
        actual_footprint=footprint,
        containment=containment,
        errors=_dedupe_terms(errors),
    )


def _actual_functions_within_plan(
    actual_functions: set[str],
    allowed_functions: set[str],
    contract: Any | None,
) -> bool:
    if contract is None or not contract.c_allow_translation_unit:
        return actual_functions <= allowed_functions
    # Dedicated algorithm islands permit private translation-unit helpers, but
    # touching another frozen export still constitutes plan drift.
    frozen_exports = set(contract.c_target_functions or [])
    return (actual_functions & frozen_exports) <= allowed_functions


_DEGENERATE_PLAN_MARKERS = ("comment-only", "comment only", "no-op", "noop", "unreachable")
_NEGATED_MARKER_PREFIXES = (
    "no",
    "not",
    "do not",
    "does not",
    "must not",
    "should not",
    "is not",
    "are not",
    "was not",
    "were not",
    "isn't",
    "aren't",
    "avoid",
    "avoids",
    "avoiding",
    "without",
    "instead of",
    "rather than",
    "cannot",
    "reject",
    "failed",
    "failure",
    "failures",
    "differs from",
)
_NEGATED_MARKER_SUFFIXES = (
    "would not satisfy",
    "does not satisfy",
    "is not acceptable",
    "are not acceptable",
    "is explicitly not",
    "are explicitly not",
    "is invalid",
    "are invalid",
    "would fail",
    "does not alter",
    "do not alter",
)

_HIGH_LEVEL_DECISION_MARKERS = (
    "decision",
    "exact fallback",
    "stock fallback",
    "filter cascade",
    "pruning",
    "prune",
    "bounded",
    "bound",
    "ranking",
    "score correction",
    "validation pass",
    "adaptive",
    "sparse",
    "state machine",
    "convergence",
)


_ALGORITHM_ISLAND_BINDINGS: dict[str, str] = {
    "profile_target_comparison": "pipeline_flow",
    "pipeline_flow_strategy": "pipeline_flow",
    # The production workers are AArch64. Keep the historical unversioned
    # contracts SSE-bound for replay, but bind new plans to the implementation
    # that is actually linked into jackhmmer/hmmsearch on this cluster.
    "ssv_filter_strategy": "ssv_filter_neon",
    "dp_recurrence_strategy": "dp_kernel_neon",
    "domain_reporting_strategy": "pipeline_flow",
    "null2_scoring_strategy": "null2_correction_neon",
    "sequence_streaming_strategy": "sequence_io",
    "iterative_search": "jackhmmer_loop",
    "jackhmmer_search_loop_strategy": "jackhmmer_loop",
    "hit_ranking_alignment_strategy": "jackhmmer_loop",
    "profile_construction_strategy": "jackhmmer_loop",
}


def _algorithm_plan_binding(
    proposal: AlgorithmPlanProposal,
) -> tuple[list[str], list[str], list[str]]:
    """Map an abstract family to one controller-owned source boundary."""
    errors: list[str] = []
    component_name = _ALGORITHM_ISLAND_BINDINGS.get(proposal.algorithm_family)
    if component_name is None:
        supported = ", ".join(sorted(_ALGORITHM_ISLAND_BINDINGS))
        return [], [], [
            f"unknown algorithm_family {proposal.algorithm_family!r}; supported families: {supported}"
        ]
    contract = ALL_C_KERNEL_CONTRACTS.get(component_name)
    if contract is None:
        return [], [], [f"algorithm island contract is unavailable: {component_name}"]
    requested_contracts = list(proposal.target_contracts or [])
    requested_functions = list(proposal.target_functions or [])
    if requested_contracts and requested_contracts != [component_name]:
        errors.append(
            "outer plan source binding is controller-owned; supplied target_contracts "
            f"must be exactly [{component_name!r}]"
        )
    allowed_functions = list(contract.c_target_functions or [])
    if requested_functions and not set(requested_functions) <= set(allowed_functions):
        errors.append("outer plan supplied target_functions outside its selected algorithm island")
    return [component_name], allowed_functions, errors


def _is_parameter_only_plan_text(text: str) -> bool:
    parameter_markers = (
        "change f1",
        "change f2",
        "change f3",
        "tune f1",
        "tune f2",
        "tune f3",
        "increase threshold",
        "decrease threshold",
        "change threshold",
        "reduce max iterations",
        "change max iterations",
        "set -n",
        "set ince",
        "set inct",
    )
    mechanism_markers = (
        "algorithm",
        "fallback",
        "bound",
        "cache",
        "reuse",
        "incremental",
        "adaptive",
        "prune",
        "sparse",
        "band",
        "fuse",
        "state machine",
    )
    return any(marker in text for marker in parameter_markers) and not any(
        marker in text for marker in mechanism_markers
    )


def _is_implementation_only_algorithm_plan(
    proposal: AlgorithmPlanProposal,
    combined: str,
) -> bool:
    if proposal.algorithm_family == "pipeline_flow_strategy":
        return "cache" in combined and any(
            marker in combined
            for marker in ("parsed target", "parser", "streaming", "dataflow")
        )
    return False


def _has_high_level_decision_surface(text: str) -> bool:
    return any(
        _contains_unnegated_marker(text, (marker,))
        for marker in _HIGH_LEVEL_DECISION_MARKERS
    )


def _contains_unnegated_marker(text: str, markers: tuple[str, ...]) -> bool:
    """Return True when marker text is asserted, not mentioned as something avoided."""
    text_l = text.lower()
    for marker in markers:
        pattern = re.escape(marker.lower())
        for match in re.finditer(pattern, text_l):
            if not _marker_is_negated(text_l, match.start()):
                return True
    return False


def _marker_is_negated(text_l: str, marker_start: int) -> bool:
    window = text_l[max(0, marker_start - 128):marker_start]
    last_boundary = max(window.rfind(ch) for ch in ".\n;:")
    if last_boundary >= 0:
        window = window[last_boundary + 1:]
    normalized = re.sub(r"[^a-z0-9']+", " ", window).strip()
    if not normalized:
        return False
    for phrase in _NEGATED_MARKER_PREFIXES:
        phrase_l = re.escape(phrase.lower())
        if re.search(rf"(?:^| ){phrase_l}(?: |$)", normalized):
            return True
    suffix = text_l[marker_start:marker_start + 160]
    next_boundary = min(
        (pos for pos in (suffix.find("."), suffix.find("\n"), suffix.find(";")) if pos >= 0),
        default=len(suffix),
    )
    suffix = suffix[:next_boundary]
    if any(phrase in suffix for phrase in _NEGATED_MARKER_SUFFIXES):
        return True
    return False


def _memory_novelty_errors(
    plan: AlgorithmPlanProposal | PatchPlanProposal,
    context_pack: ContextPackRecord,
    *,
    track: str,
) -> list[str]:
    current = (
        algorithm_mechanism_projection(plan)
        if track == "outer_algorithm"
        else implementation_mechanism_projection(plan)
    )
    current_fingerprint = mechanism_fingerprint(current)
    candidates: list[tuple[float, str, bool]] = []

    for summary in context_pack.l2_phase_summaries:
        data = _record_dict(summary)
        source_id = str(data.get("phase_summary_id") or "unknown")
        for key in ("accepted_speedups", "rejected_mechanisms"):
            for item in data.get(key) or []:
                if not isinstance(item, dict) or item.get("scope") != track:
                    continue
                _append_novelty_candidate(
                    candidates,
                    current=current,
                    current_fingerprint=current_fingerprint,
                    stored=item.get("mechanism_projection"),
                    label=f"l2 {source_id}",
                    track=track,
                )

    expected_wisdom_kind = (
        "algorithm_wisdom" if track == "outer_algorithm" else "implementation_wisdom"
    )
    for wisdom in context_pack.l3_durable_wisdom:
        data = _record_dict(wisdom)
        if data.get("confidence") == "deprecated":
            continue
        if data.get("wisdom_kind") != expected_wisdom_kind:
            continue
        source_id = str(data.get("wisdom_id") or "unknown")
        _append_novelty_candidate(
            candidates,
            current=current,
            current_fingerprint=current_fingerprint,
            stored=data.get("mechanism_projection"),
            label=f"l3 {source_id}",
            track=track,
        )

    if not candidates:
        return []
    similarity, label, exact = sorted(
        candidates,
        key=lambda item: (-item[0], item[1]),
    )[0]
    detail = "exact fingerprint" if exact else f"similarity={similarity:.3f}"
    plan_kind = "algorithm plan" if track == "outer_algorithm" else "patch plan"
    return [f"{plan_kind} is not novel against {label}: {detail}"]


def _append_novelty_candidate(
    candidates: list[tuple[float, str, bool]],
    *,
    current: dict[str, Any],
    current_fingerprint: str,
    stored: Any,
    label: str,
    track: str,
) -> None:
    projection = _valid_mechanism_projection(stored, track=track)
    if projection is None or not _projection_identity_matches(current, projection, track=track):
        return
    stored_fingerprint = mechanism_fingerprint(projection)
    if stored_fingerprint == current_fingerprint:
        candidates.append((1.0, label, True))
        return
    similarity = _projection_similarity(current, projection)
    if similarity >= 0.90:
        candidates.append((similarity, label, False))


def _valid_mechanism_projection(value: Any, *, track: str) -> dict[str, Any] | None:
    if not isinstance(value, dict) or value.get("track") != track:
        return None
    required_lists = ("mechanism_terms", "fallback_terms", "target_functions")
    if any(not isinstance(value.get(key), list) for key in required_lists):
        return None
    if track == "outer_algorithm":
        if not isinstance(value.get("algorithm_family"), str):
            return None
        if not isinstance(value.get("target_contracts"), list):
            return None
    elif track == "inner_implementation":
        if not isinstance(value.get("component_name"), str):
            return None
    else:
        return None
    return value


def _projection_identity_matches(
    current: dict[str, Any],
    stored: dict[str, Any],
    *,
    track: str,
) -> bool:
    current_functions = set(str(item) for item in current.get("target_functions") or [])
    stored_functions = set(str(item) for item in stored.get("target_functions") or [])
    if current_functions and stored_functions and not current_functions & stored_functions:
        return False
    if track == "outer_algorithm":
        if current.get("algorithm_family") != stored.get("algorithm_family"):
            return False
        current_contracts = set(str(item) for item in current.get("target_contracts") or [])
        stored_contracts = set(str(item) for item in stored.get("target_contracts") or [])
        return not current_contracts or not stored_contracts or bool(current_contracts & stored_contracts)
    return current.get("component_name") == stored.get("component_name")


def _projection_similarity(current: dict[str, Any], stored: dict[str, Any]) -> float:
    current_terms = {
        str(item)
        for key in ("mechanism_terms", "fallback_terms")
        for item in current.get(key) or []
    }
    stored_terms = {
        str(item)
        for key in ("mechanism_terms", "fallback_terms")
        for item in stored.get(key) or []
    }
    union = current_terms | stored_terms
    return len(current_terms & stored_terms) / len(union) if union else 0.0


def _record_dict(value: Any) -> dict[str, Any]:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    return dict(value) if isinstance(value, dict) else {}


def _negative_algorithm_experience_errors(
    plan: AlgorithmPlanProposal,
    recent_negative_experience: list[dict[str, Any]] | None,
) -> list[str]:
    return _negative_mechanism_errors(
        plan,
        recent_negative_experience,
        track="outer_algorithm",
    )


def _negative_mechanism_errors(
    plan: AlgorithmPlanProposal | PatchPlanProposal,
    recent_negative_experience: list[dict[str, Any]] | None,
    *,
    track: str,
) -> list[str]:
    negatives = recent_negative_experience or []
    current = (
        algorithm_mechanism_projection(plan)
        if track == "outer_algorithm"
        else implementation_mechanism_projection(plan)
    )
    current_fingerprint = mechanism_fingerprint(current)
    errors: list[str] = []
    for item in sorted(negatives, key=lambda value: str(value.get("experience_id") or value.get("lesson") or "")):
        scope = str(item.get("scope") or "inner_implementation")
        if scope != track or not _negative_target_overlaps(plan, item, track=track):
            continue
        stored_raw = item.get("mechanism_projection")
        if stored_raw:
            stored = _valid_mechanism_projection(stored_raw, track=track)
            if stored is None or not _projection_identity_matches(current, stored, track=track):
                continue
            stored_fingerprint = mechanism_fingerprint(stored)
            similarity = _projection_similarity(current, stored)
        else:
            stored_fingerprint = str(item.get("mechanism_fingerprint") or "")
            similarity = 0.0
        if stored_fingerprint == current_fingerprint:
            errors.append("plan repeats recent negative experience with an identical fingerprint")
        elif stored_raw and similarity >= 0.90:
            errors.append(
                f"plan is not novel against recent negative experience: similarity={similarity:.3f}"
            )
        elif not stored_fingerprint:
            errors.append(
                "legacy negative experience overlaps this target without a comparable mechanism fingerprint"
            )
    return errors


def _negative_target_overlaps(
    plan: AlgorithmPlanProposal | PatchPlanProposal,
    item: dict[str, Any],
    *,
    track: str,
) -> bool:
    signature = item.get("patch_signature") or {}
    item_component = item.get("component_name") or signature.get("component_name")
    item_functions = set(
        str(value)
        for value in (item.get("target_functions") or signature.get("target_functions") or [])
    )
    if track == "outer_algorithm":
        components = set(str(value) for value in (plan.target_contracts or []))
    else:
        components = {plan.component_name}
    requested = set(str(value) for value in (plan.target_functions or []))
    if item_component and str(item_component) not in components:
        return False
    return not item_functions or not requested or bool(item_functions & requested)


def _related_experience_errors(
    plan: AlgorithmPlanProposal | PatchPlanProposal,
    store: LoopStore,
    *,
    expected_scope: str,
) -> list[str]:
    related_ids = list(plan.related_experience_ids or [])
    if not related_ids:
        return []
    by_id = {
        str(item.get("experience_id")): item
        for item in store.iter_experience()
        if item.get("experience_id")
    }
    errors: list[str] = []
    requested_components = (
        set(str(value) for value in plan.target_contracts)
        if isinstance(plan, AlgorithmPlanProposal)
        else {plan.component_name}
    )
    requested_functions = set(str(value) for value in plan.target_functions)
    for experience_id in related_ids:
        item = by_id.get(str(experience_id))
        if item is None:
            errors.append(f"related_experience_id does not exist: {experience_id}")
            continue
        scope = str(item.get("scope") or "inner_implementation")
        if scope != expected_scope:
            errors.append(
                f"related_experience_id {experience_id!r} has scope {scope!r}; expected {expected_scope!r}"
            )
            continue
        signature = item.get("patch_signature") or {}
        component = item.get("component_name") or signature.get("component_name")
        target_functions = set(
            str(value)
            for value in (signature.get("target_functions") or item.get("target_functions") or [])
        )
        component_matches = not component or str(component) in requested_components
        functions_match = (
            not target_functions
            or not requested_functions
            or bool(target_functions & requested_functions)
        )
        if not component_matches or not functions_match:
            errors.append(
                f"related_experience_id {experience_id!r} does not overlap the proposed component/contract and functions"
            )
    return errors


def validate_patch_plan(
    plan: PatchPlanProposal,
    *,
    profile_evidence: dict[str, Any] | None = None,
    recent_negative_experience: list[dict[str, Any]] | None = None,
) -> list[str]:
    """Return validation errors for a first-stage patch plan."""
    errors: list[str] = []
    contract = ALL_C_KERNEL_CONTRACTS.get(plan.component_name)
    if contract is None:
        return [f"unknown C component in patch plan: {plan.component_name}"]
    allowed_functions = set(contract.c_target_functions or [])
    requested = set(plan.target_functions or [])
    if not requested:
        errors.append("patch plan must name at least one target function")
    elif not requested <= allowed_functions:
        bad = sorted(requested - allowed_functions)
        errors.append(f"patch plan targets functions outside the component contract: {bad}")
    combined = "\n".join([
        plan.algorithm_hypothesis,
        plan.intended_code_change,
        plan.runtime_effective_argument,
        plan.profile_evidence_citation,
        plan.expected_metric_effect,
        plan.anti_degenerate_check,
    ]).lower()
    if not plan.algorithm_hypothesis.strip() or not plan.intended_code_change.strip():
        errors.append("patch plan must include a non-empty algorithm hypothesis and intended code change")
    if not plan.runtime_effective_argument.strip():
        errors.append("patch plan must explain why rebuilt jackhmmer/hmmsearch executable hashes should change")
    if _contains_unnegated_marker(combined, _DEGENERATE_PLAN_MARKERS):
        errors.append("patch plan appears to rely on a comment/no-op/unreachable change instead of executable algorithm logic")
    if "tp" not in combined and "true positive" not in combined and "recall" not in combined:
        errors.append("patch plan must state how it avoids degenerate TP/recall loss")
    if "executable" not in combined and "binary" not in combined and "runtime" not in combined:
        errors.append("patch plan must include a runtime-effective/binary-changing argument")
    errors.extend(_profile_plan_errors(plan, profile_evidence=profile_evidence))
    errors.extend(_negative_experience_plan_errors(plan, recent_negative_experience=recent_negative_experience))
    return errors



def _profile_plan_errors(plan: PatchPlanProposal, *, profile_evidence: dict[str, Any] | None) -> list[str]:
    evidence = profile_evidence or {}
    if not evidence.get("has_profile"):
        return []
    citation = plan.profile_evidence_citation.strip()
    if not citation:
        return ["patch plan must include profile_evidence_citation citing measured profile evidence"]
    dominant, fraction = _dominant_profile_command(evidence)
    combined = "\n".join([
        plan.component_name,
        " ".join(plan.target_functions or []),
        " ".join(plan.target_symbols or []),
        " ".join(plan.profile_evidence_refs or []),
        plan.algorithm_hypothesis,
        plan.intended_code_change,
        plan.runtime_effective_argument,
        plan.profile_evidence_citation,
        plan.expected_metric_effect,
        plan.negative_experience_citation,
        plan.novelty_argument,
    ]).lower()
    errors: list[str] = []
    if dominant:
        dominant_l = dominant.lower()
        if dominant_l not in combined:
            errors.append(f"profile_evidence_citation must cite measured dominant command '{dominant}'")
        if fraction is not None and fraction > 0.80:
            for other in ("jackhmmer", "hmmsearch"):
                if other == dominant_l:
                    continue
                other_only_markers = (
                    f"{other}-only",
                    f"{other} only",
                    f"only {other}",
                    f"target {other}",
                    f"targets {other}",
                )
                if _contains_unnegated_marker(combined, other_only_markers):
                    errors.append(
                        f"patch plan targets non-dominant command '{other}' while measured dominant command "
                        f"'{dominant}' has wall fraction {fraction:.4f}"
                    )
                    break
    errors.extend(_symbol_profile_plan_errors(plan, evidence=evidence, combined=combined))
    return errors



def _symbol_profile_plan_errors(
    plan: PatchPlanProposal,
    *,
    evidence: dict[str, Any],
    combined: str,
) -> list[str]:
    if not evidence.get("has_symbol_profile"):
        return []
    errors: list[str] = []
    hotspot_terms = _symbol_profile_terms(evidence)
    citation_text = _plan_symbol_citation_text(plan)
    if hotspot_terms and not _text_mentions_any(citation_text, hotspot_terms):
        examples = ", ".join(hotspot_terms[:5])
        errors.append(
            "profile_evidence_citation must cite a measured symbol/function/contract hotspot "
            f"when symbol profile is available; examples: {examples}"
        )
    mapped_components = _symbol_profile_target_components(evidence)
    if mapped_components and plan.component_name not in mapped_components:
        top_component = mapped_components[0]
        if not _has_symbol_target_mismatch_explanation(combined, top_component=top_component, selected=plan.component_name):
            errors.append(
                f"patch plan targets component '{plan.component_name}' while measured symbol profile maps top hotspot "
                f"to '{top_component}'; explain why the selected component still affects the cited dominant hot path"
            )
    return errors



def _symbol_profile_terms(evidence: dict[str, Any], *, limit: int = 16) -> list[str]:
    terms: list[str] = []
    for row in (evidence.get("fine_grained_hotspots") or [])[:8]:
        if not isinstance(row, dict):
            continue
        _extend_terms(terms, row.get("symbol"), row.get("function"), row.get("component_name"))
    for row in (evidence.get("function_hotspots") or [])[:8]:
        if not isinstance(row, dict):
            continue
        _extend_terms(terms, row.get("function"), row.get("component_name"))
        _extend_terms(terms, *(row.get("functions") or []), *(row.get("symbols") or []))
    for row in (evidence.get("contract_hotspots") or [])[:5]:
        if not isinstance(row, dict):
            continue
        _extend_terms(terms, row.get("component_name"), *(row.get("functions") or []))
    return _dedupe_terms(terms)[:limit]



def _symbol_profile_target_components(evidence: dict[str, Any]) -> list[str]:
    components: list[str] = []
    for key in ("contract_hotspots", "function_hotspots"):
        for row in evidence.get(key) or []:
            if isinstance(row, dict) and row.get("component_name"):
                component_name = str(row["component_name"])
                components.append(component_name)
                runtime_name = _runtime_component_name(component_name)
                if runtime_name != component_name:
                    components.append(runtime_name)
    return _dedupe_terms(components)


def _runtime_component_name(component_name: str) -> str:
    """Map historical profiler labels to the active AArch64 contracts.

    Stored profile evidence predates the versioned NEON contracts and names
    the equivalent SSE-bound compatibility components. The measured symbols
    are still useful, but new proposals must target the runtime implementation.
    """
    return {
        "dp_kernel": "dp_kernel_neon",
        "ssv_filter": "ssv_filter_neon",
        "null2_correction": "null2_correction_neon",
    }.get(component_name, component_name)



def _extend_terms(out: list[str], *values: Any) -> None:
    for value in values:
        if value is None:
            continue
        if isinstance(value, (list, tuple, set)):
            _extend_terms(out, *value)
            continue
        text = str(value).strip()
        if len(text) >= 3:
            out.append(text)



def _dedupe_terms(values: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        key = value.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(value)
    return out



def _plan_symbol_citation_text(plan: PatchPlanProposal) -> str:
    return "\n".join([
        plan.profile_evidence_citation,
        " ".join(plan.profile_evidence_refs or []),
        " ".join(plan.target_symbols or []),
    ]).lower()



def _text_mentions_any(text: str, terms: list[str]) -> bool:
    text_l = text.lower()
    return any(term.lower() in text_l for term in terms)



def _has_symbol_target_mismatch_explanation(combined: str, *, top_component: str, selected: str) -> bool:
    if top_component.lower() not in combined or selected.lower() not in combined:
        return False
    explanation_markers = (
        "although",
        "despite",
        "indirect",
        "downstream",
        "upstream",
        "feeds",
        "affects",
        "coupled",
        "same hot path",
        "dominant hot path",
        "secondary hotspot",
        "not the top",
    )
    return any(marker in combined for marker in explanation_markers)



def _negative_experience_plan_errors(
    plan: PatchPlanProposal,
    *,
    recent_negative_experience: list[dict[str, Any]] | None,
) -> list[str]:
    return _negative_mechanism_errors(
        plan,
        recent_negative_experience,
        track="inner_implementation",
    )



def _dominant_profile_command(evidence: dict[str, Any]) -> tuple[str | None, float | None]:
    for key in ("current_best", "baseline"):
        record = evidence.get(key) or {}
        profile = record.get("profile") or {}
        dominant = profile.get("dominant_command")
        if dominant:
            fractions = profile.get("command_wall_fraction") or {}
            fraction = fractions.get(dominant)
            return str(dominant), float(fraction) if fraction is not None else None
    for item in evidence.get("recent_profiles") or []:
        profile = item.get("profile") or {}
        dominant = profile.get("dominant_command")
        if dominant:
            fractions = profile.get("command_wall_fraction") or {}
            fraction = fractions.get(dominant)
            return str(dominant), float(fraction) if fraction is not None else None
    return None, None



def eval_summary_from_record(record: EvaluationRecord | dict[str, Any] | None) -> EvalSummary | None:
    if record is None:
        return None
    data = record.model_dump(mode="json") if isinstance(record, EvaluationRecord) else record
    result = data.get("eval_result") or data
    meta = result.get("meta") if isinstance(result.get("meta"), dict) else {}
    eval_config = data.get("eval_config") if isinstance(data.get("eval_config"), dict) else {}
    return EvalSummary(
        eval_id=data.get("eval_id"),
        patchset_id=data.get("patchset_id"),
        stage=result.get("stage") or data.get("stage"),
        status=result.get("status") or data.get("status"),
        tp=result.get("tp", 0),
        fp=result.get("fp", 0),
        tp_fp_ratio=result.get("tp_fp_ratio", 0.0),
        cost_s=result.get("cost_s", 0.0),
        n_seqs=result.get("n_seqs", 0),
        iterations=result.get("iterations", 0),
        errors=result.get("errors", data.get("errors", [])),
        # Measured-conditions fingerprint, read back from where the evaluation
        # stored it. Absent on records written before the gate existed, and absent
        # is a first-class answer: those numbers are unverifiable.
        comparability=(
            (meta or {}).get("comparability")
            or eval_config.get("comparability")
            or None
        ),
    )


def _mods_from_patchset(patchset: PatchsetRecord) -> list[ComponentModification]:
    return [
        ComponentModification(contract_name=p.component_name, python_payload=None, c_patch=p.unified_diff)
        for p in patchset.patches
    ]


def _mods_from_cfg(cfg: AssembledConfig) -> list[dict[str, Any]]:
    """Serialize an assembled variant for an isolated Slurm worker."""
    mods: list[dict[str, Any]] = []
    for name, spec in cfg.specs.items():
        if spec.c_patch is not None:
            mod = ComponentModification(contract_name=name, c_patch=spec.c_patch)
        else:
            mod = ComponentModification(contract_name=name, python_payload=dict(spec.payload))
        mods.append(mod.model_dump(mode="json"))
    return mods


_DEFAULT_THRESHOLD_NAMES = {"rt1", "rt2", "rt3", "nsamples", "min_overlap", "max_diagdiff", "min_posterior", "min_endpointp"}
_DEFAULT_THRESHOLD_RE = re.compile(r"^\s*ddef->([A-Za-z_][A-Za-z0-9_]*)\s*=")


def _is_default_threshold_only_patch(diff: str) -> bool:
    """Return True for pure tuning of p7_domaindef_Create default assignments."""
    changed = _changed_payload_lines(diff)
    if not changed:
        return False
    names: list[str] = []
    for line in changed:
        match = _DEFAULT_THRESHOLD_RE.match(line)
        if not match:
            return False
        names.append(match.group(1))
    return bool(names) and all(name in _DEFAULT_THRESHOLD_NAMES for name in names)


def _is_parameter_only_patch(diff: str) -> bool:
    changed = _changed_payload_lines(diff)
    if not changed:
        return False
    executable = [line.strip() for line in changed if line.strip() and not line.strip().startswith(("/*", "*", "//"))]
    if not executable:
        return False
    assignment_or_return_literal = re.compile(
        r"^(?:[A-Za-z_][A-Za-z0-9_]*(?:->|\.)?)?[A-Za-z_][A-Za-z0-9_]*\s*=\s*[-+]?\d+(?:\.\d+)?f?;?$|"
        r"^return\s+[-+]?\d+(?:\.\d+)?f?;?$"
    )
    return all(assignment_or_return_literal.match(line) for line in executable)


def _component_guardrail_errors(idx: int, patch: PatchRecord) -> list[str]:
    if patch.component_name == "sequence_io":
        return _sequence_io_guardrail_errors(
            idx,
            patch.unified_diff,
            component_name=patch.component_name,
        )
    if patch.component_name in {"ssv_filter", "ssv_filter_neon"}:
        return _ssv_filter_guardrail_errors(
            idx,
            patch.unified_diff,
            component_name=patch.component_name,
        )
    return []


def _sequence_io_guardrail_errors(
    idx: int,
    diff: str,
    *,
    component_name: str = "sequence_io",
) -> list[str]:
    errors: list[str] = []
    added, removed = _added_removed_payload_lines(diff)
    added_text = "\n".join(added).lower()
    removed_text = "\n".join(removed).lower()
    if _adds_easel_simd_intrinsics(added_text):
        errors.append(
            f"Patch {idx} for {component_name} introduces SIMD/SSE intrinsic code "
            "(including ARM NEON) into the Easel FASTA parser. "
            f"The {component_name} contract is a parser hot path; avoid new intrinsic dependencies/includes unless the "
            "build contract explicitly supports them. Use scalar parser-preserving optimizations instead."
        )
    if _removes_sequence_parser_safety_check(added, removed):
        errors.append(
            f"Patch {idx} for {component_name} removes or weakens parser safety/classification checks. Preserve FASTA "
            "residue conversion, ignored/EOL/EOD handling, and checks such as x <= 127 / isascii / eslDSQ_* handling."
        )
    if "ascii->bpos" in removed_text and "ascii->bpos" not in added_text:
        errors.append(
            f"Patch {idx} for {component_name} removes ascii->bpos state updates. FASTA parser line/buffer position "
            "semantics must be preserved; hoisting bpos is allowed only with an explicit write-back."
        )
    if "header_fasta" in diff and _removes_name_or_description_parse(added_text, removed_text):
        errors.append(
            f"Patch {idx} for {component_name} appears to remove FASTA name/description parsing logic in header_fasta. "
            "Preserve sq->name/sq->desc semantics and parser error behavior."
        )
    return errors


def _ssv_filter_guardrail_errors(
    idx: int,
    diff: str,
    *,
    component_name: str = "ssv_filter",
) -> list[str]:
    added, removed = _added_removed_payload_lines(diff)
    added_text = "\n".join(added)
    removed_text = "\n".join(removed)
    errors: list[str] = []
    if _mentions_high_calc_band(added_text) and not _has_max_bands_guard(added_text):
        errors.append(
            f"Patch {idx} for {component_name} references calc_band_15..18 without a visible MAX_BANDS preprocessor guard. "
            "Preserve the original #if MAX_BANDS > 14 contract around high band functions."
        )
    if "#if MAX_BANDS > 14" in removed_text and "#if MAX_BANDS > 14" not in added_text:
        errors.append(
            f"Patch {idx} for {component_name} removes the MAX_BANDS > 14 preprocessor guard. Keep calc_band_15..18 "
            "guarded so builds with smaller MAX_BANDS do not reference undeclared functions."
        )
    return errors


def _adds_easel_simd_intrinsics(added_text: str) -> bool:
    markers = (
        "__m128",
        "__m128i",
        "__m128d",
        "_mm_",
        "emmintrin.h",
        "xmmintrin.h",
        "smmintrin.h",
        "tmmintrin.h",
        "immintrin.h",
        "arm_neon.h",
        "uint8x8_t",
        "uint8x16_t",
        "int8x8_t",
        "int8x16_t",
        "uint16x4_t",
        "uint16x8_t",
        "int16x4_t",
        "int16x8_t",
        "uint32x2_t",
        "uint32x4_t",
        "int32x2_t",
        "int32x4_t",
        "float32x2_t",
        "float32x4_t",
        "vld1q_",
        "vst1q_",
        "vreinterpretq_",
    )
    return any(marker.lower() in added_text for marker in markers)


def _removes_sequence_parser_safety_check(added: list[str], removed: list[str]) -> bool:
    added_text = "\n".join(added)
    safety_re = re.compile(r"x\s*(?:==|!=|<=|>=)\s*(?:127|eslDSQ_[A-Z_]+)|isascii\s*\(")
    removed_checks = [line for line in removed if safety_re.search(line)]
    if not removed_checks:
        return False
    added_checks = [line for line in added if safety_re.search(line)]
    if not added_checks:
        return True
    return len(added_checks) < len(removed_checks) and "seebuf" not in added_text


def _removes_name_or_description_parse(added_text: str, removed_text: str) -> bool:
    markers = ("sq->name", "sq->desc", "esl_sq_setname", "esl_sq_setdesc", "name", "desc")
    removed_count = sum(1 for marker in markers if marker in removed_text)
    added_count = sum(1 for marker in markers if marker in added_text)
    return removed_count > added_count


def _mentions_high_calc_band(text: str) -> bool:
    return any(re.search(rf"\bcalc_band_{idx}\b", text) for idx in range(15, 19))


def _has_max_bands_guard(text: str) -> bool:
    return "MAX_BANDS > 14" in text or "MAX_BANDS>14" in text.replace(" ", "")


def _added_removed_payload_lines(diff: str) -> tuple[list[str], list[str]]:
    added: list[str] = []
    removed: list[str] = []
    for line in diff.splitlines():
        if line.startswith(("---", "+++", "@@", "diff --git", "index ")):
            continue
        if line.startswith("+") and not line.startswith("+++"):
            added.append(line[1:])
        elif line.startswith("-") and not line.startswith("---"):
            removed.append(line[1:])
    return added, removed


def _changed_payload_lines(diff: str) -> list[str]:
    changed: list[str] = []
    for line in diff.splitlines():
        if line.startswith(("---", "+++", "@@", "diff --git", "index ")):
            continue
        if line.startswith("-") and not line.startswith("---"):
            changed.append(line[1:])
        elif line.startswith("+") and not line.startswith("+++"):
            changed.append(line[1:])
    return changed


def _file_info(
    path: Path,
    *,
    sha256: bool = True,
    digest_cache_dir: Path | None = None,
) -> dict[str, Any]:
    if not path.exists():
        return {"path": str(path), "exists": False}
    try:
        stat = path.stat()
        info: dict[str, Any] = {
            "path": str(path),
            "exists": True,
            "size": stat.st_size,
            "device": stat.st_dev,
            "inode": stat.st_ino,
            "mtime_ns": stat.st_mtime_ns,
        }
        if sha256 or digest_cache_dir is not None:
            identity = {
                "path": str(path.resolve()),
                "size": stat.st_size,
                "device": stat.st_dev,
                "inode": stat.st_ino,
                "mtime_ns": stat.st_mtime_ns,
            }
            digest: str | None = None
            cache_path: Path | None = None
            if digest_cache_dir is not None:
                digest_cache_dir.mkdir(parents=True, exist_ok=True)
                cache_path = digest_cache_dir / f"{sha256_text(str(path.resolve()))}.json"
                if cache_path.is_file():
                    try:
                        cached = json.loads(cache_path.read_text(encoding="utf-8"))
                        if cached.get("identity") == identity:
                            digest = str(cached.get("sha256") or "") or None
                    except (OSError, ValueError, TypeError):
                        digest = None
            if digest is None:
                digest = hash_file(path)
                if cache_path is not None:
                    cache_path.write_text(
                        json.dumps({"identity": identity, "sha256": digest}, sort_keys=True) + "\n",
                        encoding="utf-8",
                    )
            info["sha256"] = digest
        else:
            info["sha256"] = None
        return info
    except OSError as exc:
        return {"path": str(path), "exists": True, "error": str(exc)}
