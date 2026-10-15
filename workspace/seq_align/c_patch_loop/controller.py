"""Command-line controller for deterministic seq_align C-patch rounds."""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any

from .. import run_hmmer_cli, run_jack
from . import wisdom as durable_wisdom
from .context import build_context_pack
from .experience import append_and_compress, workflow_experience
from .models import (
    AlgorithmPlanProposal,
    AlgorithmPlanRecord,
    EvalSummary,
    EvaluationRecord,
    InvocationRecord,
    InvocationSummaryRecord,
    PatchPlanProposal,
    PatchProposal,
    PatchsetRecord,
    RunSummaryRecord,
    SlotEventRecord,
)
from .proposer import (
    BaseProposer,
    FatalProposalError,
    FixtureProposer,
    OpenAICompatibleProposer,
    ProposalDeferred,
    ProposalError,
)
from .runner import (
    PatchRunner,
    _file_info,
    _stock_source_hashes,
    eval_summary_from_record,
)
from .store import (
    LoopStore,
    canonical_json,
    hash_file,
    load_json,
    record_hash,
    sanitize_sensitive_data,
    sha256_text,
    utc_now,
    write_json,
)

LOGGER = logging.getLogger(__name__)


def run_loop(
    *,
    run_dir: str | Path,
    rounds: int,
    start_round: int = 1,
    proposer: BaseProposer,
    stage: str = "smoke",
    cpus: int = 1,
    max_iter: int = 1,
    full_max_iter: int | None = None,
    hmmer_src_root: str | Path | None = None,
    data_dir: str | Path | None = None,
    astral_db: str | Path | None = None,
    uniref_db: str | Path | None = None,
    scop_txt: str | Path | None = None,
    keep_work: bool = False,
    max_repair_attempts: int = 2,
    max_plan_reproposal_attempts: int = 0,
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
    posthoc_profile: str = "auto",
    pyhmmer_profile_prepass: bool = False,
    pyhmmer_profile_prepass_stage: str = "full",
    pyhmmer_profile_max_target_bytes: int | None = None,
    wisdom_dir: str | Path | None = None,
    disable_durable_wisdom: bool = False,
    import_seed_wisdom: bool = False,
    promote_wisdom: bool = False,
) -> list[dict[str, Any]]:
    """Run deterministic C-patch proposal/evaluation rounds."""
    if start_round < 1:
        raise ValueError("start_round must be at least 1")
    if max_plan_reproposal_attempts < 0:
        raise ValueError("max_plan_reproposal_attempts must be non-negative")
    store = LoopStore(run_dir)
    runner = PatchRunner(
        store,
        hmmer_src_root=hmmer_src_root,
        data_dir=data_dir,
        astral_db=astral_db,
        uniref_db=uniref_db,
        scop_txt=scop_txt,
        cpus=cpus,
        max_iter=max_iter,
        full_max_iter=full_max_iter,
        accept_min_tp_fraction=accept_min_tp_fraction,
        min_speedup_fraction=min_speedup_fraction,
        eval_repeats=eval_repeats,
        paired_current_best_repeats=paired_current_best_repeats,
        near_speedup_band_fraction=near_speedup_band_fraction,
        require_algorithmic_patch=require_algorithmic_patch,
        external_profiler=external_profiler,
        perf_top_queries=perf_top_queries,
        perf_top_symbols=perf_top_symbols,
        perf_frequency=perf_frequency,
        perf_command=perf_command,
        query_timeout=query_timeout,
        eval_backend=eval_backend,
        posthoc_profile=posthoc_profile,
    )
    wisdom_root = Path(wisdom_dir) if wisdom_dir is not None else store.root / "wisdom"
    if import_seed_wisdom:
        durable_wisdom.import_seed_wisdom(wisdom_root)
    context_wisdom_dir = None if disable_durable_wisdom else wisdom_root
    if pyhmmer_profile_prepass:
        runner.evaluate_pyhmmer_profile_prepass(
            stage=pyhmmer_profile_prepass_stage,
            max_target_bytes=pyhmmer_profile_max_target_bytes,
        )
    results: list[dict[str, Any]] = []
    quality_reference = _baseline_summary(store, runner=runner, stage=stage)
    current_best = _current_best_summary(
        store,
        runner=runner,
        stage=stage,
        accept_min_tp_fraction=accept_min_tp_fraction,
    )
    if quality_reference is None:
        baseline_record = runner.evaluate_stock_baseline(stage=stage, keep_work=keep_work)
        quality_reference = eval_summary_from_record(baseline_record)
        current_best = quality_reference if baseline_record.status == "ok" else current_best
        results.append({
            "round_idx": 0,
            "patchset_id": baseline_record.patchset_id,
            "final_eval_id": baseline_record.eval_id,
            "stage": baseline_record.stage,
            "status": baseline_record.status,
            "decision": baseline_record.decision,
            "baseline": True,
            "errors": list(baseline_record.errors or []),
            "data_dir": str((baseline_record.eval_config or {}).get("data_dir") or ""),
            "astral_db": str((baseline_record.eval_config or {}).get("astral_db") or ""),
            "uniref_db": str((baseline_record.eval_config or {}).get("uniref_db") or ""),
            "scop_txt": str((baseline_record.eval_config or {}).get("scop_txt") or ""),
        })
        if baseline_record.status != "ok":
            store.write_index("run_summary", results)
            if promote_wisdom:
                durable_wisdom.promote_wisdom_from_run(store, wisdom_dir=wisdom_root)
            return results
    elif current_best is None:
        current_best = quality_reference
    for idx in range(start_round, start_round + rounds):
        pack = build_context_pack(
            store,
            round_idx=idx,
            hmmer_src_root=hmmer_src_root,
            current_best_patchset_id=current_best.patchset_id if current_best else None,
            baseline=quality_reference,
            current_best=current_best,
            accept_min_tp_fraction=accept_min_tp_fraction,
            min_speedup_fraction=min_speedup_fraction,
            eval_repeats=eval_repeats,
            wisdom_dir=context_wisdom_dir,
        )
        round_results, current_best, stop_all = _run_legacy_round(
            idx=idx,
            store=store,
            runner=runner,
            proposer=proposer,
            pack=pack,
            stage=stage,
            keep_work=keep_work,
            max_repair_attempts=max_repair_attempts,
            max_plan_reproposal_attempts=max_plan_reproposal_attempts,
            quality_reference=quality_reference,
            current_best=current_best,
        )
        results.extend(round_results)
        if stop_all:
            break
    store.write_index("run_summary", results)
    if promote_wisdom:
        durable_wisdom.promote_wisdom_from_run(store, wisdom_dir=wisdom_root)
    return results


def _run_legacy_round(
    *,
    idx: int,
    store: LoopStore,
    runner: PatchRunner,
    proposer: BaseProposer,
    pack: Any,
    stage: str,
    keep_work: bool,
    max_repair_attempts: int,
    max_plan_reproposal_attempts: int,
    quality_reference: EvalSummary,
    current_best: EvalSummary | None,
) -> tuple[list[dict[str, Any]], EvalSummary | None, bool]:
    """Run one legacy round with bounded plan-level recovery."""
    rows: list[dict[str, Any]] = []
    replan_feedback: str | None = None
    replan_parent_patch_plan_id: str | None = None
    for plan_reproposal_idx in range(max_plan_reproposal_attempts + 1):
        try:
            plan_proposal, patch_plan = _propose_valid_plan(
                proposer,
                runner,
                pack,
                parent_patchset_id=current_best.patchset_id if current_best else None,
                max_repair_attempts=max_repair_attempts,
                initial_feedback=replan_feedback,
                initial_parent_patch_plan_id=replan_parent_patch_plan_id,
                plan_reproposal_idx=plan_reproposal_idx,
            )
        except FatalProposalError as exc:
            record = {
                "round_idx": idx,
                "status": "fatal_proposal_error",
                "error": str(exc),
                "context_pack_id": pack.context_pack_id,
                "plan_reproposal_idx": plan_reproposal_idx,
            }
            store.write_index(f"round_{idx:03d}_proposal_error", record)
            _append_proposal_error_experience(store, record)
            rows.append(record)
            return rows, current_best, True
        except ProposalError as exc:
            record = {
                "round_idx": idx,
                "status": "plan_proposal_error",
                "error": str(exc),
                "context_pack_id": pack.context_pack_id,
                "plan_reproposal_idx": plan_reproposal_idx,
            }
            store.write_index(f"round_{idx:03d}_proposal_error", record)
            _append_proposal_error_experience(store, record)
            rows.append(record)
            return rows, current_best, False

        approved_algorithm_plan = _legacy_algorithm_plan_from_patch_plan(plan_proposal)
        try:
            proposal, metadata = _propose_patch(
                proposer,
                pack,
                plan=plan_proposal,
                algorithm_plan=approved_algorithm_plan,
            )
        except FatalProposalError as exc:
            record = {
                "round_idx": idx,
                "status": "fatal_proposal_error",
                "error": str(exc),
                "context_pack_id": pack.context_pack_id,
                "patch_plan_id": patch_plan.patch_plan_id,
                "plan_reproposal_idx": plan_reproposal_idx,
            }
            store.write_index(f"round_{idx:03d}_proposal_error", record)
            _append_proposal_error_experience(store, record)
            rows.append(record)
            return rows, current_best, True
        except ProposalError as exc:
            record = {
                "round_idx": idx,
                "status": "proposal_error",
                "error": str(exc),
                "context_pack_id": pack.context_pack_id,
                "patch_plan_id": patch_plan.patch_plan_id,
                "plan_reproposal_idx": plan_reproposal_idx,
            }
            store.write_index(f"round_{idx:03d}_proposal_error", record)
            _append_proposal_error_experience(store, record)
            rows.append(record)
            return rows, current_best, False

        attempt_idx = 0
        while True:
            patch_metadata = {
                **dict(metadata),
                "attempt_idx": attempt_idx,
                "plan_reproposal_idx": plan_reproposal_idx,
            }
            patchset = runner.build_patchset(
                proposal,
                pack,
                patch_metadata,
                patch_plan=patch_plan,
            )
            static_record, cfg = runner.static_verify(patchset)
            if cfg is None:
                if attempt_idx < max_repair_attempts:
                    try:
                        proposal, metadata = _repair_patch(
                            proposer,
                            pack,
                            previous=proposal,
                            feedback=_repair_feedback(static_record),
                            attempt_idx=attempt_idx + 1,
                            plan=plan_proposal,
                            algorithm_plan=approved_algorithm_plan,
                        )
                    except ProposalError as exc:
                        record = {
                            "round_idx": idx,
                            "context_pack_id": pack.context_pack_id,
                            "patch_plan_id": patch_plan.patch_plan_id,
                            "patchset_id": patchset.patchset_id,
                            "static_eval_id": static_record.eval_id,
                            "status": "proposal_error",
                            "error": str(exc),
                            "attempt_idx": attempt_idx + 1,
                            "plan_reproposal_idx": plan_reproposal_idx,
                        }
                        store.write_index(f"round_{idx:03d}_repair_error", record)
                        _append_proposal_error_experience(store, record, outcome="repair_proposal_error")
                        rows.append(record)
                        return rows, current_best, False
                    attempt_idx += 1
                    continue
                row = {
                    "round_idx": idx,
                    "context_pack_id": pack.context_pack_id,
                    "patch_plan_id": patch_plan.patch_plan_id,
                    "patchset_id": patchset.patchset_id,
                    "static_eval_id": static_record.eval_id,
                    "stage": static_record.stage,
                    "status": static_record.status,
                    "decision": static_record.decision,
                    "failure_kind": _failure_kind_for_feedback(static_record),
                    "attempt_idx": attempt_idx,
                    "repair_attempts_used": attempt_idx,
                    "plan_reproposal_idx": plan_reproposal_idx,
                }
                if (
                    _repairable_final_failure_kind(static_record) is not None
                    and plan_reproposal_idx < max_plan_reproposal_attempts
                ):
                    row["replanned_after_exhaustion"] = True
                    rows.append(row)
                    replan_feedback = _plan_reproposal_feedback(static_record)
                    replan_parent_patch_plan_id = patch_plan.patch_plan_id
                    break
                rows.append(row)
                return rows, current_best, False

            smoke_record = runner.evaluate(
                patchset,
                cfg,
                stage="smoke",
                current_best=None if stage == "full" else current_best,
                quality_reference=None if stage == "full" else quality_reference,
                keep_work=keep_work,
            )
            final_record = smoke_record
            if stage == "full" and smoke_record.status == "ok":
                final_record = runner.evaluate(
                    patchset,
                    cfg,
                    stage="full",
                    current_best=current_best,
                    quality_reference=quality_reference,
                    keep_work=keep_work,
                )
            failure_kind = _repairable_final_failure_kind(final_record)
            if failure_kind is not None and attempt_idx < max_repair_attempts:
                try:
                    proposal, metadata = _repair_patch(
                        proposer,
                        pack,
                        previous=proposal,
                        feedback=_repair_feedback(static_record, final_record),
                        attempt_idx=attempt_idx + 1,
                        plan=plan_proposal,
                        algorithm_plan=approved_algorithm_plan,
                    )
                except ProposalError as exc:
                    record = {
                        "round_idx": idx,
                        "context_pack_id": pack.context_pack_id,
                        "patch_plan_id": patch_plan.patch_plan_id,
                        "patchset_id": patchset.patchset_id,
                        "static_eval_id": static_record.eval_id,
                        "final_eval_id": final_record.eval_id,
                        "status": "proposal_error",
                        "error": str(exc),
                        "attempt_idx": attempt_idx + 1,
                        "plan_reproposal_idx": plan_reproposal_idx,
                    }
                    store.write_index(f"round_{idx:03d}_repair_error", record)
                    _append_proposal_error_experience(store, record, outcome="repair_proposal_error")
                    rows.append(record)
                    return rows, current_best, False
                attempt_idx += 1
                continue

            row = {
                "round_idx": idx,
                "context_pack_id": pack.context_pack_id,
                "patch_plan_id": patch_plan.patch_plan_id,
                "patchset_id": patchset.patchset_id,
                "static_eval_id": static_record.eval_id,
                "final_eval_id": final_record.eval_id,
                "stage": final_record.stage,
                "status": final_record.status,
                "decision": final_record.decision,
                "attempt_idx": attempt_idx,
                "repair_attempts_used": attempt_idx,
                "plan_reproposal_idx": plan_reproposal_idx,
            }
            if failure_kind is not None:
                row["failure_kind"] = failure_kind
            elif final_record.status in {"compile_error", "run_error"}:
                row["failure_kind"] = "environmental_failure"
            if failure_kind is not None and plan_reproposal_idx < max_plan_reproposal_attempts:
                row["replanned_after_exhaustion"] = True
                rows.append(row)
                replan_feedback = _plan_reproposal_feedback(static_record, final_record)
                replan_parent_patch_plan_id = patch_plan.patch_plan_id
                break
            if final_record.decision.startswith("accepted"):
                current_best = eval_summary_from_record(final_record)
            rows.append(row)
            return rows, current_best, False
    return rows, current_best, False


def run_hierarchical_loop(
    *,
    run_dir: str | Path,
    outer_rounds: int,
    proposer: BaseProposer,
    inner_trials: int = 10,
    cpus: int = 1,
    max_iter: int = 1,
    full_max_iter: int | None = None,
    hmmer_src_root: str | Path | None = None,
    data_dir: str | Path | None = None,
    astral_db: str | Path | None = None,
    uniref_db: str | Path | None = None,
    scop_txt: str | Path | None = None,
    keep_work: bool = False,
    max_repair_attempts: int = 2,
    min_commit_speedup_fraction: float = 0.03,
    eval_repeats: int = 1,
    paired_current_best_repeats: int = 0,
    require_algorithmic_patch: bool = True,
    eval_backend: str | None = None,
    runner: PatchRunner | None = None,
    session_id: str | None = None,
    posthoc_profile: str = "auto",
    screen_inner_trials: bool = False,
    screen_min_speedup_fraction: float = 0.5,
    # WIDE-BAND hard reject for the inner screen. 0 (default) = ordering only.
    # A candidate is discarded by the screen only when it is slower than the
    # reference by more than this fraction; set out-of-band from the measured
    # zero-effect distribution, never from a guess.
    screen_reject_speedup_fraction: float = 0.0,
    screen_eval_repeats: int = 1,
) -> list[dict[str, Any]]:
    """Run the official outer-algorithm/inner-implementation workflow.

    The controller and both model calls run locally.  Only ``PatchRunner``'s
    Full evaluation backend may submit work to Slurm.  An outer plan is frozen
    for all of its inner trials; each valid faster inner result becomes the
    next trial's parent, but global indexes are updated only after the complete
    inner budget and only for a quality-valid speedup of at least 3 percent.

    ``paired_current_best_repeats`` (2026-10-08). Every comparison this loop
    makes is "freshly measured candidate" vs "a cost stored by an earlier
    evaluation", so run-to-run timing drift -- node load, page cache, HMMER
    oversubscription -- is charged to the patch. A deliberately score-neutral
    probe patch once "won" 76% that way and was committed. Setting this to >0
    makes ``PatchRunner`` re-measure the current best interleaved with the
    candidate and honour the paired verdict (``accepted_speedup_paired`` /
    ``rejected_not_faster_paired``), which retires the drift. It is 0 by
    default because it costs ``repeats x 2`` extra full evaluations **per
    triggering evaluation** -- the hook sits in ``PatchRunner.evaluate``, which
    the inner trials share -- so it multiplies cluster usage.
    """
    if outer_rounds < 0:
        raise ValueError("outer_rounds must be non-negative")
    if inner_trials <= 0:
        raise ValueError("inner_trials must be positive")
    if min_commit_speedup_fraction < 0.0 or min_commit_speedup_fraction >= 1.0:
        raise ValueError("min_commit_speedup_fraction must be in [0, 1)")
    if paired_current_best_repeats < 0:
        raise ValueError("paired_current_best_repeats must be non-negative")

    store = LoopStore(run_dir)
    active_runner = runner or PatchRunner(
        store,
        hmmer_src_root=hmmer_src_root,
        data_dir=data_dir,
        astral_db=astral_db,
        uniref_db=uniref_db,
        scop_txt=scop_txt,
        cpus=cpus,
        max_iter=max_iter,
        full_max_iter=full_max_iter,
        accept_min_tp_fraction=1.0,
        # Inner candidates only need to be strictly faster to become the
        # provisional parent. The 3% gate is applied once, after all inner
        # trials, against the frozen outer parent.
        min_speedup_fraction=0.0,
        eval_repeats=eval_repeats,
        paired_current_best_repeats=int(paired_current_best_repeats),
        require_algorithmic_patch=require_algorithmic_patch,
        eval_backend=eval_backend,
        # Post-hoc profiling is a control-layer policy: the controller decides
        # whether to spend an extra job on the baseline and on each committed
        # winner. The runner decides how. It never runs inside the inner loop.
        posthoc_profile=posthoc_profile,
        # Inner-trial screening: cheap small-subset single measurement per trial,
        # full repeats+paired measurement only for candidates that could clear the
        # commit gate. An explicitly supplied runner keeps its own configuration.
        screen_inner_trials=screen_inner_trials,
        screen_min_speedup_fraction=screen_min_speedup_fraction,
        screen_reject_speedup_fraction=screen_reject_speedup_fraction,
        screen_eval_repeats=screen_eval_repeats,
    )
    if Path(active_runner.store.run_dir).resolve() != Path(store.run_dir).resolve():
        raise ValueError("runner store and hierarchical run_dir must refer to the same run")
    active_runner.min_speedup_fraction = 0.0
    # This used to hard-reset to 0 (paired confirmation was unreachable on the
    # hierarchical path even though the runner implements it). It now follows
    # the caller: an explicitly supplied runner is still overridden, so the
    # parameter remains the single source of truth.
    active_runner.paired_current_best_repeats = int(paired_current_best_repeats)
    arguments = {
        "mode": "hierarchical_full_v1",
        "rounds": outer_rounds,
        "inner_rounds": inner_trials,
        "stage": "full",
        "min_commit_speedup_fraction": min_commit_speedup_fraction,
        "max_repair_attempts": max_repair_attempts,
        "eval_repeats": eval_repeats,
        # Explicit session identity. When present it -- not the environment
        # fingerprint -- decides which invocation a call re-attaches to, so a
        # callback bridge keeps its slot grid across interruptions, reconnects and
        # environment drift. Omitted => historical behaviour, unchanged.
        "session_id": session_id,
    }
    with store.single_writer():
        invocation = _open_invocation(store, active_runner, arguments)
        active_runner.invocation_id = invocation.invocation_id
        try:
            results = _run_hierarchical_invocation(
                store=store,
                runner=active_runner,
                invocation=invocation,
                proposer=proposer,
                outer_rounds=outer_rounds,
                inner_trials=inner_trials,
                hmmer_src_root=hmmer_src_root,
                keep_work=keep_work,
                max_repair_attempts=max_repair_attempts,
                min_commit_speedup_fraction=min_commit_speedup_fraction,
                eval_repeats=eval_repeats,
                # Screening travels with the runner (it owns the query subset and
                # repeat policy), so the two cannot disagree about the subset size.
                screen_inner_trials=bool(getattr(active_runner, "screen_inner_trials", False)),
                screen_min_speedup_fraction=float(
                    getattr(active_runner, "screen_min_speedup_fraction", 0.5)
                ),
                screen_reject_speedup_fraction=float(
                    getattr(active_runner, "screen_reject_speedup_fraction", 0.0)
                ),
                screen_eval_repeats=int(getattr(active_runner, "screen_eval_repeats", 1)),
            )
        except BaseException:
            _finalize_invocation(store, invocation, terminal_state="interrupted")
            _rebuild_run_summary(store)
            raise
        terminal = _invocation_terminal_state(store, invocation)
        _finalize_invocation(store, invocation, terminal_state=terminal)
        _rebuild_run_summary(store)
        store.write_index("hierarchical_run_summary", results)
        return results


def _run_hierarchical_invocation(
    *,
    store: LoopStore,
    runner: PatchRunner,
    invocation: InvocationRecord,
    proposer: BaseProposer,
    outer_rounds: int,
    inner_trials: int,
    hmmer_src_root: str | Path | None,
    keep_work: bool,
    max_repair_attempts: int,
    min_commit_speedup_fraction: float,
    eval_repeats: int,
    screen_inner_trials: bool = False,
    screen_min_speedup_fraction: float = 0.5,
    # WIDE-BAND hard reject for the inner screen. 0 (default) = ordering only.
    # A candidate is discarded by the screen only when it is slower than the
    # reference by more than this fraction; set out-of-band from the measured
    # zero-effect distribution, never from a guess.
    screen_reject_speedup_fraction: float = 0.0,
    screen_eval_repeats: int = 1,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    baseline = _baseline_summary(store, runner=runner, stage="full")
    if baseline is None:
        baseline_record = runner.evaluate_stock_baseline(stage="full", keep_work=keep_work)
        baseline = eval_summary_from_record(baseline_record)
        results.append(_evaluation_result_row(baseline_record, baseline=True))
        if baseline_record.status != "ok":
            return results

    # A run begins at stock.  On resume, only an explicitly committed outer
    # winner in this run can replace that parent through the speed-best index.
    global_parent = _current_best_summary(
        store,
        runner=runner,
        stage="full",
        accept_min_tp_fraction=1.0,
    ) or baseline
    if not _quality_valid(global_parent, baseline):
        global_parent = baseline

    fatal = False
    for outer_idx in range(1, outer_rounds + 1):
        outer_head = _slot_head(store, invocation, outer_idx, None)
        if outer_head is not None and outer_head.state in {"committed", "not_committed"}:
            committed_eval_id = outer_head.result.get("committed_eval_id")
            if committed_eval_id:
                global_parent = eval_summary_from_record(store.get_evaluation(str(committed_eval_id)))
            results.append(dict(outer_head.result))
            continue
        if outer_head is not None and outer_head.state == "blocked":
            results.append(dict(outer_head.result))
            continue

        outer_parent = global_parent.model_copy(deep=True)
        outer_parent_record = _evaluation_for_summary(store, outer_parent)
        timeout_reference = _timeout_reference_path(store, outer_parent_record)
        source_root = _materialized_source(runner, outer_parent.patchset_id, hmmer_src_root)
        outer_pack = build_context_pack(
            store,
            round_idx=outer_idx,
            hmmer_src_root=source_root,
            parent_patchset_id=outer_parent.patchset_id,
            current_best_patchset_id=outer_parent.patchset_id,
            baseline=outer_parent,
            current_best=outer_parent,
            max_recent=max(32, inner_trials * 3 + 4),
            accept_min_tp_fraction=1.0,
        # Inner acceptance means "strictly faster and quality-valid".  The
        # 3% threshold is intentionally enforced only once, at outer commit.
        min_speedup_fraction=0.0,
            eval_repeats=eval_repeats,
        )
        outer_pack = _focus_full_trajectory(store, outer_pack, None)
        try:
            algorithm_proposal, algorithm_plan = _resume_or_propose_algorithm_plan(
                store,
                runner,
                invocation,
                proposer,
                outer_pack,
                outer_idx=outer_idx,
                comparison_patchset_id=outer_parent.patchset_id,
                max_repair_attempts=max_repair_attempts,
            )
        except ProposalDeferred as exc:
            # This round is not blocked: the proposal simply has to be authored
            # out-of-process.  ``awaiting_proposal`` is deliberately absent from
            # every resume-terminal set, so a later invocation re-enters proposal
            # authoring for this round instead of replaying it as finished.
            _transition_slot(
                store,
                invocation,
                outer_idx=outer_idx,
                inner_idx=None,
                state="awaiting_proposal",
                result={
                    "outer_idx": outer_idx,
                    "status": "awaiting_proposal",
                    "reason": str(exc),
                },
            )
            raise
        except FatalProposalError as exc:
            row = {
                "outer_idx": outer_idx,
                "status": "fatal_algorithm_plan_proposal_error",
                "error": str(exc),
                "outer_parent_eval_id": outer_parent.eval_id,
            }
            _block_outer_and_inner(store, invocation, outer_idx, inner_trials, row)
            _append_proposal_error_experience(store, {**row, "round_idx": outer_idx})
            results.append(row)
            fatal = True
            break
        except ProposalError as exc:
            row = {
                "outer_idx": outer_idx,
                "status": "algorithm_plan_proposal_error",
                "error": str(exc),
                "outer_parent_eval_id": outer_parent.eval_id,
            }
            _block_outer_and_inner(store, invocation, outer_idx, inner_trials, row)
            _append_proposal_error_experience(store, {**row, "round_idx": outer_idx})
            results.append(row)
            continue

        provisional = outer_parent
        provisional_eval = outer_parent_record
        # Reconstruct the provisional chain from completed slots.  This is also
        # what makes restart deterministic after a controller interruption.
        for inner_idx in range(1, inner_trials + 1):
            head = _slot_head(store, invocation, outer_idx, inner_idx)
            if head is None or head.state != "evaluated":
                continue
            eval_id = head.eval_id or head.result.get("final_eval_id")
            if not eval_id:
                continue
            record = store.get_evaluation(str(eval_id))
            candidate = eval_summary_from_record(record)
            if _candidate_eligible(record, candidate, outer_parent) and _strictly_faster(candidate, provisional):
                provisional = candidate
                provisional_eval = record

        for inner_idx in range(1, inner_trials + 1):
            head = _slot_head(store, invocation, outer_idx, inner_idx)
            if head is not None and head.state in {"evaluated", "blocked"}:
                if head.result:
                    results.append(dict(head.result))
                continue

            try:
                row, record = _run_inner_full_slot(
                    store=store,
                    runner=runner,
                    invocation=invocation,
                    proposer=proposer,
                    algorithm_proposal=algorithm_proposal,
                    algorithm_plan=algorithm_plan,
                    outer_idx=outer_idx,
                    inner_idx=inner_idx,
                    outer_parent=outer_parent,
                    provisional=provisional,
                    hmmer_src_root=hmmer_src_root,
                    keep_work=keep_work,
                    max_repair_attempts=max_repair_attempts,
                    min_commit_speedup_fraction=min_commit_speedup_fraction,
                    eval_repeats=eval_repeats,
                    query_timeout_reference=timeout_reference,
                    screen_inner_trials=screen_inner_trials,
                    screen_min_speedup_fraction=screen_min_speedup_fraction,
                    screen_reject_speedup_fraction=screen_reject_speedup_fraction,
                    screen_eval_repeats=screen_eval_repeats,
                )
            except BaseException as exc:
                current = _slot_head(store, invocation, outer_idx, inner_idx)
                interrupted = dict(current.result) if current is not None else {}
                # A deferral is recoverable waiting, not a crash: record it as
                # ``awaiting_proposal``.  Crucially this state keeps the slot out
                # of the patchset-resume branch in ``_run_inner_full_slot``, so a
                # rejected-then-deferred patch is re-authored rather than replayed
                # (replaying it raised "failed to reconstruct current-best config").
                state = (
                    "awaiting_proposal"
                    if isinstance(exc, ProposalDeferred)
                    else "interrupted"
                )
                interrupted.update({
                    "outer_idx": outer_idx,
                    "inner_idx": inner_idx,
                    "status": state,
                    "error": f"{type(exc).__name__}: {exc}",
                })
                _transition_slot(
                    store,
                    invocation,
                    outer_idx=outer_idx,
                    inner_idx=inner_idx,
                    state=state,
                    result=interrupted,
                )
                raise
            results.append(row)
            if record is None:
                continue
            candidate = eval_summary_from_record(record)
            if _candidate_eligible(record, candidate, outer_parent) and _strictly_faster(candidate, provisional):
                provisional = candidate
                provisional_eval = record

        # The outer commit gate is the ONE place a verdict is rendered, so it is the
        # one place a comparability failure must be loud: an incomparable pair never
        # commits and the mismatch is written into the round row.
        speedup, incomparable = _try_speedup_fraction(outer_parent, provisional)
        committed = (
            provisional.patchset_id != outer_parent.patchset_id
            and incomparable is None
            and _quality_valid(provisional, outer_parent)
            and speedup is not None
            and speedup + 1e-12 >= min_commit_speedup_fraction
            and provisional_eval is not None
        )
        if committed:
            winning_patchset = store.get_patchset(provisional.patchset_id or "")
            if hasattr(runner, "commit_evaluation"):
                runner.commit_evaluation(winning_patchset, provisional_eval)
            else:
                runner._update_indexes(winning_patchset, provisional_eval, baseline=False)
            global_parent = provisional
        else:
            global_parent = outer_parent

        outer_row = {
            "outer_idx": outer_idx,
            "status": "committed" if committed else "not_committed",
            "algorithm_plan_id": algorithm_plan.algorithm_plan_id,
            "outer_parent_patchset_id": outer_parent.patchset_id,
            "outer_parent_eval_id": outer_parent.eval_id,
            "winner_patchset_id": provisional.patchset_id,
            "winner_eval_id": provisional.eval_id,
            "committed_eval_id": provisional.eval_id if committed else None,
            "speedup_fraction": speedup,
            "required_speedup_fraction": min_commit_speedup_fraction,
            "inner_trials": inner_trials,
        }
        if incomparable is not None:
            # Make the refusal readable from the round row and the ledger: the round
            # did not commit *because the numbers were not comparable*, which is a
            # different fact from "the candidate was not faster".
            outer_row["status"] = "not_committed_incomparable"
            outer_row["incomparable_measurement"] = incomparable
        _transition_slot(
            store,
            invocation,
            outer_idx=outer_idx,
            inner_idx=None,
            state="committed" if committed else "not_committed",
            result=outer_row,
        )
        results.append(outer_row)

    if fatal:
        for outer_idx in range(1, outer_rounds + 1):
            head = _slot_head(store, invocation, outer_idx, None)
            if head is not None and head.state == "pending":
                _block_outer_and_inner(
                    store,
                    invocation,
                    outer_idx,
                    inner_trials,
                    {"outer_idx": outer_idx, "status": "blocked_after_fatal_proposal_error"},
                )
    return results


def _run_inner_full_slot(
    *,
    store: LoopStore,
    runner: PatchRunner,
    invocation: InvocationRecord,
    proposer: BaseProposer,
    algorithm_proposal: AlgorithmPlanProposal,
    algorithm_plan: AlgorithmPlanRecord,
    outer_idx: int,
    inner_idx: int,
    outer_parent: EvalSummary,
    provisional: EvalSummary,
    hmmer_src_root: str | Path | None,
    keep_work: bool,
    max_repair_attempts: int,
    min_commit_speedup_fraction: float,
    eval_repeats: int,
    query_timeout_reference: Path | None,
    screen_inner_trials: bool = False,
    screen_min_speedup_fraction: float = 0.5,
    # WIDE-BAND hard reject for the inner screen. 0 (default) = ordering only.
    # A candidate is discarded by the screen only when it is slower than the
    # reference by more than this fraction; set out-of-band from the measured
    # zero-effect distribution, never from a guess.
    screen_reject_speedup_fraction: float = 0.0,
    screen_eval_repeats: int = 1,
) -> tuple[dict[str, Any], EvaluationRecord | None]:
    head = _slot_head(store, invocation, outer_idx, inner_idx)
    pack = None
    plan_proposal: PatchPlanProposal | None = None
    proposal: PatchProposal | None = None
    metadata: dict[str, Any] = {}
    static_record: EvaluationRecord | None = None
    repair_idx = 0
    resumed_patchset_id = None
    if head is not None and head.state in {"full_running", "interrupted", "repairing", "static_verified"}:
        resumed_patchset_id = head.patchset_id or head.result.get("patchset_id")

    if resumed_patchset_id:
        patchset = store.get_patchset(str(resumed_patchset_id))
        cfg = runner._config_for_patchset_id(patchset.patchset_id)
        if cfg is None:
            static_record, cfg = runner.static_verify(patchset)
            if cfg is None:
                row = _blocked_inner_row(
                    outer_idx,
                    inner_idx,
                    algorithm_plan.algorithm_plan_id,
                    "resume_static_validation_failed",
                    static_eval_id=static_record.eval_id,
                    patchset_id=patchset.patchset_id,
                )
                _transition_slot(store, invocation, outer_idx=outer_idx, inner_idx=inner_idx, state="blocked", result=row)
                return row, None
        patch_plan_id = patchset.patch_plan_id
        static_eval_id = head.result.get("static_eval_id") if head else None
        if static_eval_id:
            static_record = store.get_evaluation(str(static_eval_id))
        repair_idx = int(patchset.proposal_metadata.get("attempt_idx", 0))
    else:
        source_root = _materialized_source(runner, provisional.patchset_id, hmmer_src_root)
        pack = build_context_pack(
            store,
            round_idx=outer_idx,
            hmmer_src_root=source_root,
            parent_patchset_id=provisional.patchset_id,
            current_best_patchset_id=provisional.patchset_id,
            baseline=outer_parent,
            current_best=provisional,
            # Static and Full records are stored separately.  Read enough raw
            # entries to retain all prior Full trials before focusing below.
            max_recent=max(32, inner_idx * 3 + 4),
            accept_min_tp_fraction=1.0,
            min_speedup_fraction=min_commit_speedup_fraction,
            eval_repeats=eval_repeats,
        )
        pack = _focus_full_trajectory(store, pack, algorithm_plan.algorithm_plan_id)
        try:
            plan_proposal, patch_plan = _propose_valid_inner_plan(
                proposer,
                runner,
                pack,
                algorithm_proposal=algorithm_proposal,
                algorithm_plan=algorithm_plan,
                parent_patchset_id=provisional.patchset_id,
                inner_idx=inner_idx,
                max_repair_attempts=max_repair_attempts,
            )
        except ProposalError as exc:
            row = _blocked_inner_row(
                outer_idx,
                inner_idx,
                algorithm_plan.algorithm_plan_id,
                "inner_plan_proposal_error",
                error=str(exc),
            )
            _transition_slot(store, invocation, outer_idx=outer_idx, inner_idx=inner_idx, state="blocked", result=row)
            _append_proposal_error_experience(store, {**row, "round_idx": outer_idx})
            return row, None

        proposal_error: ProposalError | None = None
        proposal = None
        metadata = {}
        for proposal_attempt_idx in range(max_repair_attempts + 1):
            try:
                proposal, metadata = _propose_inner_patch(
                    proposer,
                    pack,
                    plan=plan_proposal,
                    algorithm_plan=algorithm_proposal,
                )
                metadata = {
                    **metadata,
                    "patch_proposal_attempt_idx": proposal_attempt_idx,
                }
                proposal_error = None
                break
            except FatalProposalError:
                raise
            except ProposalError as exc:
                proposal_error = exc
        if proposal is None:
            row = _blocked_inner_row(
                outer_idx,
                inner_idx,
                algorithm_plan.algorithm_plan_id,
                "patch_proposal_error",
                patch_plan_id=patch_plan.patch_plan_id,
                error=str(proposal_error or "patch proposal failed"),
            )
            _transition_slot(store, invocation, outer_idx=outer_idx, inner_idx=inner_idx, state="blocked", result=row)
            _append_proposal_error_experience(store, {**row, "round_idx": outer_idx})
            return row, None

        implementation = runner.build_implementation_trial(
            round_idx=outer_idx,
            inner_idx=inner_idx,
            algorithm_plan=algorithm_plan,
            parent_version_id=provisional.patchset_id or "stock",
            context_pack_id=pack.context_pack_id,
            patch_plan_id=patch_plan.patch_plan_id,
            proposal_metadata=metadata,
            implementation_approach=patch_plan.intended_code_change,
            trajectory_analysis=(
                f"Trial {inner_idx} follows {len(pack.recent_history)} recorded trajectory entries; "
                f"parent={provisional.patchset_id}, parent_cost_s={provisional.cost_s}."
            ),
            interface_preservation_argument=patch_plan.anti_degenerate_check,
            fallback_argument=algorithm_plan.exact_fallback_strategy,
            thread_safety_argument="Keep all mutable optimization state worker-local.",
            expected_profile_effect=patch_plan.expected_metric_effect,
        )
        _transition_slot(
            store,
            invocation,
            outer_idx=outer_idx,
            inner_idx=inner_idx,
            state="planned",
            result={
                "outer_idx": outer_idx,
                "inner_idx": inner_idx,
                "status": "planned",
                "algorithm_plan_id": algorithm_plan.algorithm_plan_id,
                "patch_plan_id": patch_plan.patch_plan_id,
                "implementation_trial_id": implementation.implementation_trial_id,
                "parent_patchset_id": provisional.patchset_id,
            },
        )

        patchset = None
        cfg = None
        static_record = None
        for repair_idx in range(max_repair_attempts + 1):
            patch_metadata = {
                **metadata,
                "attempt_idx": repair_idx,
                "outer_idx": outer_idx,
                "inner_idx": inner_idx,
                "algorithm_plan_id": algorithm_plan.algorithm_plan_id,
                "implementation_trial_id": implementation.implementation_trial_id,
            }
            patchset = runner.build_patchset(
                proposal,
                pack,
                patch_metadata,
                parent_patchset_id=provisional.patchset_id,
                patch_plan=patch_plan,
                algorithm_plan_id=algorithm_plan.algorithm_plan_id,
                implementation_trial_id=implementation.implementation_trial_id,
                parent_version_id=provisional.patchset_id,
            )
            static_record, cfg = runner.static_verify(patchset)
            if cfg is not None:
                break
            _transition_slot(
                store,
                invocation,
                outer_idx=outer_idx,
                inner_idx=inner_idx,
                state="repairing",
                result={
                    "status": "repairing",
                    "patchset_id": patchset.patchset_id,
                    "static_eval_id": static_record.eval_id,
                    "repair_attempt_idx": repair_idx,
                },
            )
            if repair_idx >= max_repair_attempts:
                break
            try:
                proposal, metadata = _repair_inner_patch(
                    proposer,
                    pack,
                    previous=proposal,
                    feedback=_repair_feedback(static_record),
                    attempt_idx=repair_idx + 1,
                    plan=plan_proposal,
                    algorithm_plan=algorithm_proposal,
                )
            except ProposalError:
                cfg = None
                break

        if patchset is None or static_record is None or cfg is None:
            row = _blocked_inner_row(
                outer_idx,
                inner_idx,
                algorithm_plan.algorithm_plan_id,
                "static_validation_failed",
                patch_plan_id=patch_plan.patch_plan_id,
                patchset_id=patchset.patchset_id if patchset else None,
                static_eval_id=static_record.eval_id if static_record else None,
            )
            _transition_slot(store, invocation, outer_idx=outer_idx, inner_idx=inner_idx, state="blocked", result=row)
            return row, None
        patch_plan_id = patch_plan.patch_plan_id
        static_eval_id = static_record.eval_id
        _transition_slot(
            store,
            invocation,
            outer_idx=outer_idx,
            inner_idx=inner_idx,
            state="static_verified",
            result={
                "status": "static_verified",
                "patchset_id": patchset.patchset_id,
                "static_eval_id": static_record.eval_id,
            },
        )

    while True:
        running = {
            "outer_idx": outer_idx,
            "inner_idx": inner_idx,
            "status": "full_running",
            "algorithm_plan_id": algorithm_plan.algorithm_plan_id,
            "patch_plan_id": patch_plan_id,
            "implementation_trial_id": patchset.implementation_trial_id,
            "patchset_id": patchset.patchset_id,
            "static_eval_id": static_eval_id,
            "parent_patchset_id": provisional.patchset_id,
            "quality_reference_eval_id": outer_parent.eval_id,
            "query_timeout_reference": str(query_timeout_reference) if query_timeout_reference else None,
            "repair_attempt_idx": repair_idx,
        }
        if screen_inner_trials:
            # Cheap pre-screen: one small-subset single measurement, used to ORDER
            # candidates for the outer gate -- not to judge them.
            #
            # The screen runs on a tiny separate database and a handful of queries,
            # so its per-pair noise is far larger than the commit gate. Treating it
            # as a verdict therefore discards real candidates on timing noise. It
            # may only hard-reject when the candidate is slower than the round parent
            # by a WIDE margin that is set explicitly out-of-band
            # (SEQ_ALIGN_SCREEN_REJECT_SPEEDUP_FRACTION, disabled by default); at any
            # other margin the trial proceeds to the authoritative full measurement
            # and the screen result is recorded as an ordering hint.
            #
            # Fails open: a screen that cannot run (no smoke baseline, cluster error,
            # incomparable numbers) must not fail the trial -- it just forfeits the
            # saving.
            try:
                screen_record, screen_reference = _screen_inner_candidate(
                    store=store,
                    runner=runner,
                    patchset=patchset,
                    cfg=cfg,
                    provisional=provisional,
                    outer_parent=outer_parent,
                    keep_work=keep_work,
                    query_timeout_reference=query_timeout_reference,
                )
            except Exception as exc:  # noqa: BLE001 -- screening is an optimisation
                LOGGER.warning(
                    "inner screening unavailable for patchset %s, falling through to the full "
                    "measurement: %s: %s",
                    patchset.patchset_id,
                    type(exc).__name__,
                    exc,
                )
                screen_record = None
                screen_reference = None
            if screen_record is not None and screen_reference is not None:
                screen_speedup, screen_incomparable = _try_speedup_fraction(
                    screen_reference, eval_summary_from_record(screen_record)
                )
                screen_floor = min_commit_speedup_fraction * min(max(screen_min_speedup_fraction, 0.0), 0.95)
                # 0 (the default) disables the hard reject entirely: the screen then
                # only orders. A negative value would be nonsense, so clamp to "off".
                wide_band_reject = screen_reject_speedup_fraction or 0.0
                if screen_incomparable is not None:
                    LOGGER.warning(
                        "inner screen for patchset %s is not comparable to its reference; "
                        "ordering signal discarded, proceeding to the full measurement: %s",
                        patchset.patchset_id,
                        screen_incomparable.get("message"),
                    )
                if (
                    wide_band_reject > 0.0
                    and screen_incomparable is None
                    and screen_speedup is not None
                    and screen_record.status == "ok"
                    and screen_speedup < -wide_band_reject
                ):
                    decision = "rejected_screening_not_faster"
                    screen_record.decision = decision
                    candidate = eval_summary_from_record(screen_record)
                    quality_valid = _quality_valid(candidate, screen_reference)
                    row = {
                        **running,
                        "status": screen_record.status,
                        "decision": decision,
                        "final_eval_id": screen_record.eval_id,
                        "screening": True,
                        "screen_speedup_fraction": round(screen_speedup, 6) if screen_speedup is not None else None,
                        "screen_required_speedup_fraction": round(screen_floor, 6),
                        "screen_hard_reject_fraction": round(wide_band_reject, 6),
                        "screen_ordering_only": True,
                        "screen_reference_eval_id": screen_reference.eval_id,
                        "tp": candidate.tp,
                        "fp": candidate.fp,
                        "cost_s": candidate.cost_s,
                        "quality_valid_against_screen_reference": quality_valid,
                        "implementation_valid": False,
                        "provisional_incumbent": False,
                        "query_results": ((screen_record.eval_result or {}).get("meta") or {}).get("query_results", []),
                    }
                    if resumed_patchset_id:
                        row.update({"resumed": True, "resumed_patchset_id": str(resumed_patchset_id)})
                    _transition_slot(
                        store, invocation, outer_idx=outer_idx, inner_idx=inner_idx, state="evaluated", result=row
                    )
                    return row, screen_record
        _transition_slot(store, invocation, outer_idx=outer_idx, inner_idx=inner_idx, state="full_running", result=running)
        record = _evaluate_inner_full(
            runner,
            patchset,
            cfg,
            current_best=provisional,
            quality_reference=outer_parent,
            keep_work=keep_work,
            query_timeout_reference=query_timeout_reference,
        )
        failure_kind = _repairable_final_failure_kind(record)
        if failure_kind is None or repair_idx >= max_repair_attempts:
            break

        _transition_slot(
            store,
            invocation,
            outer_idx=outer_idx,
            inner_idx=inner_idx,
            state="repairing",
            result={
                **running,
                "status": "repairing",
                "failure_kind": failure_kind,
                "failed_eval_id": record.eval_id,
                "repair_attempt_idx": repair_idx + 1,
            },
        )

        # Resume may enter here with only immutable artifact IDs. Rehydrate the
        # exact frozen plan and failed diff before asking the same proposer to
        # repair it; no new outer plan or inner slot is consumed.
        if pack is None:
            source_root = _materialized_source(runner, provisional.patchset_id, hmmer_src_root)
            pack = build_context_pack(
                store,
                round_idx=outer_idx,
                hmmer_src_root=source_root,
                parent_patchset_id=provisional.patchset_id,
                current_best_patchset_id=provisional.patchset_id,
                baseline=outer_parent,
                current_best=provisional,
                max_recent=max(32, inner_idx * 3 + 4),
                accept_min_tp_fraction=1.0,
                min_speedup_fraction=min_commit_speedup_fraction,
                eval_repeats=eval_repeats,
            )
            pack = _focus_full_trajectory(store, pack, algorithm_plan.algorithm_plan_id)
        if plan_proposal is None:
            if not patch_plan_id:
                raise ProposalError("cannot repair Full failure without a patch_plan_id")
            plan_proposal = _patch_plan_proposal_from_record(store.get_patch_plan(patch_plan_id))
        if proposal is None:
            proposal = _patch_proposal_from_patchset(patchset)
        if not metadata:
            metadata = dict(patchset.proposal_metadata)

        while True:
            next_repair_idx = repair_idx + 1
            try:
                proposal, metadata = _repair_inner_patch(
                    proposer,
                    pack,
                    previous=proposal,
                    feedback=_repair_feedback(*(item for item in (static_record, record) if item is not None)),
                    attempt_idx=next_repair_idx,
                    plan=plan_proposal,
                    algorithm_plan=algorithm_proposal,
                )
            except FatalProposalError:
                raise
            except ProposalError as exc:
                row = _blocked_inner_row(
                    outer_idx,
                    inner_idx,
                    algorithm_plan.algorithm_plan_id,
                    "full_repair_proposal_error",
                    patch_plan_id=patch_plan_id,
                    patchset_id=patchset.patchset_id,
                    final_eval_id=record.eval_id,
                    error=str(exc),
                )
                _transition_slot(store, invocation, outer_idx=outer_idx, inner_idx=inner_idx, state="blocked", result=row)
                _append_proposal_error_experience(store, {**row, "round_idx": outer_idx})
                return row, None

            repair_idx = next_repair_idx
            patchset = runner.build_patchset(
                proposal,
                pack,
                {
                    **metadata,
                    "attempt_idx": repair_idx,
                    "outer_idx": outer_idx,
                    "inner_idx": inner_idx,
                    "algorithm_plan_id": algorithm_plan.algorithm_plan_id,
                    "implementation_trial_id": patchset.implementation_trial_id,
                    "repaired_from_eval_id": record.eval_id,
                    "repair_failure_kind": failure_kind,
                },
                parent_patchset_id=provisional.patchset_id,
                patch_plan=store.get_patch_plan(patch_plan_id),
                algorithm_plan_id=algorithm_plan.algorithm_plan_id,
                implementation_trial_id=patchset.implementation_trial_id,
                parent_version_id=provisional.patchset_id,
            )
            static_record, cfg = runner.static_verify(patchset)
            static_eval_id = static_record.eval_id
            if cfg is not None:
                _transition_slot(
                    store,
                    invocation,
                    outer_idx=outer_idx,
                    inner_idx=inner_idx,
                    state="static_verified",
                    result={
                        "status": "static_verified",
                        "patchset_id": patchset.patchset_id,
                        "static_eval_id": static_eval_id,
                        "repair_attempt_idx": repair_idx,
                        "repaired_from_eval_id": record.eval_id,
                    },
                )
                break
            if repair_idx >= max_repair_attempts:
                row = _blocked_inner_row(
                    outer_idx,
                    inner_idx,
                    algorithm_plan.algorithm_plan_id,
                    "full_repair_static_validation_failed",
                    patch_plan_id=patch_plan_id,
                    patchset_id=patchset.patchset_id,
                    static_eval_id=static_eval_id,
                    final_eval_id=record.eval_id,
                )
                _transition_slot(store, invocation, outer_idx=outer_idx, inner_idx=inner_idx, state="blocked", result=row)
                return row, None
            _transition_slot(
                store,
                invocation,
                outer_idx=outer_idx,
                inner_idx=inner_idx,
                state="repairing",
                result={
                    "status": "repairing",
                    "patchset_id": patchset.patchset_id,
                    "static_eval_id": static_eval_id,
                    "failed_eval_id": record.eval_id,
                    "repair_attempt_idx": repair_idx + 1,
                    "failure_kind": "static_validation_failure_after_full_repair",
                },
            )

    candidate = eval_summary_from_record(record)
    quality_valid = _quality_valid(candidate, outer_parent)
    implementation_valid = _candidate_eligible(record, candidate, outer_parent)
    provisional_update = implementation_valid and _strictly_faster(candidate, provisional)
    row = {
        **running,
        "status": record.status,
        "decision": record.decision,
        "final_eval_id": record.eval_id,
        "tp": candidate.tp,
        "fp": candidate.fp,
        "cost_s": candidate.cost_s,
        "quality_valid_against_outer_parent": quality_valid,
        "implementation_valid": implementation_valid,
        "provisional_incumbent": provisional_update,
        "query_results": ((record.eval_result or {}).get("meta") or {}).get("query_results", []),
    }
    if resumed_patchset_id:
        row.update({
            "resumed": True,
            "resumed_patchset_id": str(resumed_patchset_id),
        })
    _transition_slot(store, invocation, outer_idx=outer_idx, inner_idx=inner_idx, state="evaluated", result=row)
    return row, record


def _screen_inner_candidate(
    *,
    store: LoopStore,
    runner: PatchRunner,
    patchset: PatchsetRecord,
    cfg: Any,
    provisional: EvalSummary,
    outer_parent: EvalSummary,
    keep_work: bool,
    query_timeout_reference: Path | None,
) -> tuple[EvaluationRecord, EvalSummary]:
    """Run the cheap inner-trial screen and return ``(screen_record, screen_reference)``.

    The reference must come from the *same* query subset and stage as the screen,
    otherwise the comparison mixes a small-subset candidate cost with a
    full-subset baseline cost. The smoke baseline is therefore measured once per
    run (stage ``smoke`` is not otherwise evaluated by the hierarchical loop) and
    cached as the ``baseline_smoke`` index.
    """
    # The reference must come from the *same* query subset and stage as the screen,
    # otherwise the comparison mixes a small-subset candidate cost with a
    # full-subset baseline cost. The smoke baseline is measured once per run
    # (stage ``smoke`` is not otherwise evaluated by the hierarchical loop) and
    # cached as the ``baseline_smoke`` index.
    # Read-only reuse: a previously recorded smoke reference may be used, but this
    # path never CREATES one in the index. An inner path that writes an index can
    # publish state the outer round never chose, so the smoke reference is measured
    # with ``commit_indexes=False`` and held only for this screening decision.
    reference = eval_summary_from_record(store.read_index("baseline_smoke"))
    if reference is None or not reference.cost_s:
        smoke_record = runner.evaluate_stock_baseline(
            stage="smoke", keep_work=keep_work, commit_indexes=False, write_legacy_result=False
        )
        reference = eval_summary_from_record(smoke_record)
        if reference is None or reference.status != "ok":
            raise ProposalError(
                "inner screening needs a smoke baseline, but the smoke baseline evaluation "
                f"did not return a usable summary (status={getattr(reference, 'status', None)})"
            )
    screen_record = runner.screen_candidate(
        patchset,
        cfg,
        current_best=provisional,
        quality_reference=reference,
        keep_work=keep_work,
        query_timeout_reference=query_timeout_reference,
    )
    return screen_record, reference


def _evaluate_inner_full(
    runner: PatchRunner,
    patchset: PatchsetRecord,
    cfg: Any,
    *,
    current_best: EvalSummary,
    quality_reference: EvalSummary,
    keep_work: bool,
    query_timeout_reference: Path | None,
) -> EvaluationRecord:
    kwargs = {
        "stage": "full",
        "current_best": current_best,
        "quality_reference": quality_reference,
        # The VERDICT reference is the round parent, not the in-round ranking
        # incumbent. On later trials ``current_best`` is a faster sibling from the
        # same round, and taking the verdict against a sibling would ask "is this
        # better than the other candidate I just made" instead of "is this better
        # than what the round started from".
        "paired_reference": quality_reference,
        "keep_work": keep_work,
        "commit_indexes": False,
        "write_legacy_result": False,
        "query_timeout_reference": query_timeout_reference,
    }
    try:
        return runner.evaluate(patchset, cfg, **kwargs)
    except TypeError as exc:
        # Compatibility for test doubles and older runners. Never hide an
        # unrelated TypeError raised from inside an evaluation backend.
        if not any(
            name in str(exc)
            for name in (
                "commit_indexes",
                "write_legacy_result",
                "query_timeout_reference",
                "paired_reference",
            )
        ):
            raise
        # ``commit_indexes`` / ``write_legacy_result`` are SAFETY flags, not
        # conveniences: dropping them restores their permissive defaults and lets an
        # inner trial write the authoritative indexes the outer round owns. A runner
        # that cannot be told not to commit must not run an inner trial at all, so
        # this is a hard error rather than a silent downgrade.
        if "commit_indexes" in str(exc) or "write_legacy_result" in str(exc):
            raise ProposalError(
                "the evaluation runner does not accept commit_indexes/write_legacy_result, so an "
                "inner trial cannot be prevented from writing the authoritative indexes; refusing "
                f"to run the inner measurement ({exc})"
            ) from exc
        kwargs.pop("query_timeout_reference", None)
        # Dropping ``paired_reference`` is safe: the paired confirmation then falls
        # back to ``current_best``, which is the pre-change behaviour. Dropping the
        # two safety flags above is not, which is why they raise instead.
        kwargs.pop("paired_reference", None)
        return runner.evaluate(patchset, cfg, **kwargs)


def _propose_valid_algorithm_plan(
    proposer: BaseProposer,
    runner: PatchRunner,
    pack,
    *,
    comparison_patchset_id: str | None,
    max_repair_attempts: int,
) -> tuple[AlgorithmPlanProposal, AlgorithmPlanRecord]:
    feedback: str | None = None
    rejected_parent: str | None = None
    for attempt_idx in range(max_repair_attempts + 1):
        try:
            proposal, metadata = proposer.propose_algorithm_plan(
                pack,
                feedback=feedback,
                attempt_idx=attempt_idx if attempt_idx else None,
            )
        except TypeError as exc:
            if "feedback" not in str(exc) and "attempt_idx" not in str(exc):
                raise
            if feedback is not None:
                raise ProposalError("legacy proposer cannot repair rejected algorithm plan") from exc
            proposal, metadata = proposer.propose_algorithm_plan(pack)
        record = runner.build_algorithm_plan(
            proposal,
            pack,
            {**metadata, "algorithm_plan_attempt_idx": attempt_idx},
            parent_version_id=comparison_patchset_id,
            parent_algorithm_plan_id=rejected_parent,
        )
        if record.validation_status == "ok":
            # The outer model selects an abstract family.  The controller owns
            # the concrete one-island source binding, so all inner prompts use
            # the validated/bound record rather than untrusted target hints.
            return _algorithm_proposal_from_record(record), record
        append_and_compress(store=runner.store, record=workflow_experience(
            outcome="algorithm_plan_validation_failure",
            lesson=(
                f"Outer algorithm plan {record.algorithm_plan_id} was rejected before any Full slot: "
                f"{record.validation_errors}."
            ),
            severity="medium",
            metric_summary={
                "round_idx": record.round_idx,
                "algorithm_plan_id": record.algorithm_plan_id,
                "validation_errors": list(record.validation_errors),
            },
            patch_signature={
                "algorithm_family": record.algorithm_family,
                "target_contracts": list(record.target_contracts),
                "algorithm_plan_id": record.algorithm_plan_id,
            },
            negative_patterns=["algorithm_plan_validation_failure", *record.validation_errors],
            retrieval_tags=["outer_algorithm", "plan_validation"],
        ))
        rejected_parent = record.algorithm_plan_id
        feedback = json.dumps({
            "algorithm_plan_id": record.algorithm_plan_id,
            "validation_errors": record.validation_errors,
            "repair_instruction": (
                "Return a corrected high-level algorithm plan for exactly one active island. "
                "Do not return code, a diff, parameter tuning, or multiple islands."
            ),
        }, sort_keys=True)
    raise ProposalError(
        f"algorithm plan validation failed after {max_repair_attempts} repairs: {feedback}"
    )


def _resume_or_propose_algorithm_plan(
    store: LoopStore,
    runner: PatchRunner,
    invocation: InvocationRecord,
    proposer: BaseProposer,
    pack,
    *,
    outer_idx: int,
    comparison_patchset_id: str | None,
    max_repair_attempts: int,
) -> tuple[AlgorithmPlanProposal, AlgorithmPlanRecord]:
    head = _slot_head(store, invocation, outer_idx, None)
    plan_id = None if head is None else (head.algorithm_plan_id or head.result.get("algorithm_plan_id"))
    if plan_id and head.state in {"validated", "running", "interrupted"}:
        record = store.get_algorithm_plan(str(plan_id))
        return _algorithm_proposal_from_record(record), record
    proposal, record = _propose_valid_algorithm_plan(
        proposer,
        runner,
        pack,
        comparison_patchset_id=comparison_patchset_id,
        max_repair_attempts=max_repair_attempts,
    )
    _transition_slot(
        store,
        invocation,
        outer_idx=outer_idx,
        inner_idx=None,
        state="validated",
        result={
            "outer_idx": outer_idx,
            "status": "validated",
            "algorithm_plan_id": record.algorithm_plan_id,
            "comparison_patchset_id": comparison_patchset_id,
        },
    )
    return _algorithm_proposal_from_record(record), record


def _propose_valid_inner_plan(
    proposer: BaseProposer,
    runner: PatchRunner,
    pack,
    *,
    algorithm_proposal: AlgorithmPlanProposal,
    algorithm_plan: AlgorithmPlanRecord,
    parent_patchset_id: str | None,
    inner_idx: int,
    max_repair_attempts: int,
) -> tuple[PatchPlanProposal, Any]:
    feedback: str | None = None
    rejected_parent: str | None = None
    for attempt_idx in range(max_repair_attempts + 1):
        try:
            proposal, metadata = proposer.propose_plan(
                pack,
                feedback=feedback,
                attempt_idx=attempt_idx if attempt_idx else None,
                algorithm_plan=algorithm_proposal,
            )
        except TypeError as exc:
            if "algorithm_plan" not in str(exc):
                raise
            proposal, metadata = proposer.propose_plan(
                pack,
                feedback=feedback,
                attempt_idx=attempt_idx if attempt_idx else None,
            )
        record = runner.build_patch_plan(
            proposal,
            pack,
            {**metadata, "inner_plan_attempt_idx": attempt_idx},
            parent_patch_plan_id=rejected_parent,
            parent_patchset_id=parent_patchset_id,
            algorithm_plan=algorithm_plan,
            inner_idx=inner_idx,
        )
        if record.validation_status == "ok":
            return proposal, record
        rejected_parent = record.patch_plan_id
        feedback = _plan_repair_feedback(record)
    raise ProposalError(
        f"inner patch plan validation failed after {max_repair_attempts} repairs: {feedback}"
    )


def _propose_inner_patch(
    proposer: BaseProposer,
    pack,
    *,
    plan: PatchPlanProposal,
    algorithm_plan: AlgorithmPlanProposal,
):
    try:
        return proposer.propose(pack, plan=plan, algorithm_plan=algorithm_plan)
    except TypeError as exc:
        if "algorithm_plan" not in str(exc):
            raise
        return _propose_patch(proposer, pack, plan=plan)


def _repair_inner_patch(
    proposer: BaseProposer,
    pack,
    *,
    previous,
    feedback: str,
    attempt_idx: int,
    plan: PatchPlanProposal,
    algorithm_plan: AlgorithmPlanProposal,
):
    try:
        return proposer.repair(
            pack,
            previous=previous,
            feedback=feedback,
            attempt_idx=attempt_idx,
            plan=plan,
            algorithm_plan=algorithm_plan,
        )
    except TypeError as exc:
        if "algorithm_plan" not in str(exc):
            raise
        return _repair_patch(
            proposer,
            pack,
            previous=previous,
            feedback=feedback,
            attempt_idx=attempt_idx,
            plan=plan,
        )


def _algorithm_proposal_from_record(record: AlgorithmPlanRecord) -> AlgorithmPlanProposal:
    fields = AlgorithmPlanProposal.model_fields
    data = record.model_dump(mode="json")
    return AlgorithmPlanProposal(**{name: data[name] for name in fields if name in data})


def _patch_plan_proposal_from_record(record) -> PatchPlanProposal:
    data = record.model_dump(mode="json")
    data["target_functions"] = data.get("c_target_functions", [])
    return PatchPlanProposal(**{
        name: data[name]
        for name in PatchPlanProposal.model_fields
        if name in data
    })


def _patch_proposal_from_patchset(patchset: PatchsetRecord) -> PatchProposal:
    if len(patchset.patches) != 1:
        raise ProposalError(
            f"cannot repair patchset {patchset.patchset_id}: expected exactly one patch, "
            f"found {len(patchset.patches)}"
        )
    patch = patchset.patches[0]
    return PatchProposal(
        component_name=patch.component_name,
        rationale=patch.rationale,
        unified_diff=patch.unified_diff,
        expected_metric_effect=patch.expected_metric_effect,
        risk_notes=patch.risk_notes,
    )


def _focus_full_trajectory(store: LoopStore, pack, algorithm_plan_id: str | None):
    """Keep prior Full results, prioritizing the frozen plan when supplied.

    Static records remain available through workflow experience.  Removing them
    from the raw trajectory prevents one static + one Full artifact per trial
    from evicting half of a ten-trial trajectory.
    """
    full = [
        item for item in pack.recent_history
        if ((item.get("evaluation") or {}).get("stage") == "full")
    ]
    same_plan: list[dict[str, Any]] = []
    other: list[dict[str, Any]] = []
    for item in full:
        plan = item.get("algorithm_plan") or {}
        if algorithm_plan_id and plan.get("algorithm_plan_id") == algorithm_plan_id:
            same_plan.append(item)
        else:
            other.append(item)
    trajectory = [*same_plan[:10], *other[: max(0, 10 - len(same_plan))]]
    working = dict(pack.l1_raw_working_memory)
    working["trajectory"] = trajectory
    focused = pack.model_copy(update={
        "context_pack_id": None,
        "recent_history": trajectory,
        "l1_raw_working_memory": working,
    })
    focused.context_pack_id = record_hash(focused, id_field="context_pack_id")
    store.put_context_pack(focused)
    return focused


def _quality_valid(candidate: EvalSummary, reference: EvalSummary) -> bool:
    return (
        candidate.status == "ok"
        and candidate.tp >= reference.tp
        and candidate.fp <= reference.fp
    )


def _candidate_eligible(
    record: EvaluationRecord,
    candidate: EvalSummary,
    quality_reference: EvalSummary,
) -> bool:
    if not _quality_valid(candidate, quality_reference):
        return False
    if record.build.get("runtime_effective") is False:
        return False
    ineligible_decisions = {
        "rejected_not_runtime_effective",
        "rejected_parameter_tuning",
        "rejected_plan_diff_binding",
        "rejected_paired_quality_regression",
        # Paired-confirmation verdicts must count as ineligible, or the knob is
        # decorative: the commit gate recomputes its speedup from *stored*
        # costs, so a candidate the paired re-measurement refused to confirm
        # would still be committed on the same stale numbers. Failing closed on
        # an inconclusive confirmation is the point of asking for one.
        "rejected_not_faster_paired",
        "rejected_paired_confirmation_error",
        "rejected_paired_incomplete_evidence",
        # A candidate rejected by the cheap pre-screen never earned the full
        # measurement, so it must not be committable on the screen's own numbers.
        "rejected_screening_not_faster",
    }
    return record.decision not in ineligible_decisions


def _strictly_faster(candidate: EvalSummary, reference: EvalSummary) -> bool:
    """True only when the candidate is faster AND the two numbers are comparable.

    Refusing is not faster: an incomparable pair must never advance the provisional
    incumbent, so the refusal degrades to ``False`` here rather than crashing a
    long-running loop. The mismatch itself is recorded on the summaries.
    """
    from .comparability import IncomparableMeasurement, compare_summaries

    if not (candidate.cost_s > 0.0 and reference.cost_s > 0.0):
        return False
    try:
        compare_summaries(reference, candidate)
    except IncomparableMeasurement:
        return False
    return candidate.cost_s + 1e-12 < reference.cost_s


def _speedup_fraction(reference: EvalSummary, candidate: EvalSummary) -> float:
    """Speedup of ``candidate`` over ``reference``, or raise if they are not comparable.

    Raises :class:`~.comparability.IncomparableMeasurement` instead of returning a
    number. Callers that must keep running (the inner-loop ordering) use
    :func:`_try_speedup_fraction`; the outer commit gate lets it propagate to a
    non-commit. There is deliberately no path that returns a ratio for two numbers
    measured under different conditions.
    """
    from .comparability import compare_summaries

    return compare_summaries(reference, candidate)


def _try_speedup_fraction(reference: EvalSummary, candidate: EvalSummary) -> tuple[float | None, dict[str, Any] | None]:
    """``(speedup, None)`` or ``(None, mismatch_detail)`` -- never a silent ratio."""
    from .comparability import IncomparableMeasurement, compare_summaries

    try:
        return compare_summaries(reference, candidate), None
    except IncomparableMeasurement as exc:
        return None, {"message": str(exc), "mismatches": dict(exc.mismatches)}


def _evaluation_for_summary(store: LoopStore, summary: EvalSummary) -> EvaluationRecord | None:
    if summary.eval_id:
        path = store.evaluations_dir / f"{summary.eval_id}.json"
        if path.is_file():
            return store.get_evaluation(summary.eval_id)
    return None


def _timeout_reference_path(store: LoopStore, record: EvaluationRecord | None) -> Path | None:
    if record is None or not record.eval_id or not record.eval_result:
        return None
    path = store.root / "timeout_references" / f"{record.eval_id}.json"
    write_json(path, record.eval_result)
    return path


def _materialized_source(
    runner: PatchRunner,
    patchset_id: str | None,
    fallback: str | Path | None,
) -> Path:
    if patchset_id:
        try:
            return runner.materialize_patchset(patchset_id)
        except (FileNotFoundError, ValueError):
            pass
    return Path(fallback or runner.hmmer_src_root).resolve()


def _evaluation_result_row(record: EvaluationRecord, *, baseline: bool = False) -> dict[str, Any]:
    result = record.eval_result or {}
    return {
        "outer_idx": 0,
        "baseline": baseline,
        "status": record.status,
        "decision": record.decision,
        "patchset_id": record.patchset_id,
        "final_eval_id": record.eval_id,
        "tp": result.get("tp", 0),
        "fp": result.get("fp", 0),
        "cost_s": result.get("cost_s", 0.0),
    }


def _blocked_inner_row(
    outer_idx: int,
    inner_idx: int,
    algorithm_plan_id: str | None,
    status: str,
    **details: Any,
) -> dict[str, Any]:
    return {
        "outer_idx": outer_idx,
        "inner_idx": inner_idx,
        "algorithm_plan_id": algorithm_plan_id,
        "status": status,
        **details,
    }


def _block_outer_and_inner(
    store: LoopStore,
    invocation: InvocationRecord,
    outer_idx: int,
    inner_trials: int,
    result: dict[str, Any],
) -> None:
    _transition_slot(store, invocation, outer_idx=outer_idx, inner_idx=None, state="blocked", result=result)
    for inner_idx in range(1, inner_trials + 1):
        head = _slot_head(store, invocation, outer_idx, inner_idx)
        if head is None or head.state == "pending":
            _transition_slot(
                store,
                invocation,
                outer_idx=outer_idx,
                inner_idx=inner_idx,
                state="blocked",
                result={**result, "inner_idx": inner_idx},
            )


def _configuration_payload(arguments: dict[str, Any], runner: Any) -> dict[str, Any]:
    options = {key: value for key, value in sorted(arguments.items()) if value is not None}
    stage = str(options.get("stage") or "full")
    if hasattr(runner, "reuse_fingerprint"):
        fingerprint = runner.reuse_fingerprint(stage=stage)
    else:
        fingerprint = canonical_json({
            "stage": stage,
            "eval_backend": getattr(runner, "eval_backend", None),
            "hmmer_src_root": str(getattr(runner, "hmmer_src_root", "")),
            "data_dir": str(getattr(runner, "data_dir", "")),
            "uniref_db": str(getattr(runner, "uniref_db", "")),
            "cpus": getattr(runner, "cpus", None),
            "full_max_iter": getattr(runner, "full_max_iter", None),
        })
    payload = {
        "schema_version": "hierarchical_controller.v1",
        "options": options,
        "evaluation_reuse_fingerprint": fingerprint,
    }
    session_id = options.get("session_id")
    if session_id:
        payload["session_id"] = str(session_id)
    return payload


def _configuration_identity(configuration: dict[str, Any]) -> dict[str, Any]:
    """Return the subset of a configuration that decides *invocation identity*.

    Invocation identity is what lets a later call re-attach to an interrupted run
    and pick up its slot grid. The environment fingerprint answers a different
    question -- "is a cached evaluation still reusable?" -- and the two must not
    be conflated, or any environment drift silently orphans every slot and the
    loop restarts from scratch.

    Without an explicit ``session_id`` the historical behaviour is preserved
    exactly: the fingerprint participates, so an environment change starts a
    fresh invocation. With a ``session_id`` the session *is* the identity, and the
    fingerprint is retained on the record for audit and cache-hit decisions only.
    """
    if not configuration.get("session_id"):
        return configuration
    return {
        key: value
        for key, value in configuration.items()
        if key != "evaluation_reuse_fingerprint"
    }


def _open_invocation(
    store: LoopStore,
    runner: Any,
    arguments: dict[str, Any],
) -> InvocationRecord:
    configuration = _configuration_payload(arguments, runner)
    configuration_id = sha256_text(canonical_json(_configuration_identity(configuration)))
    summaries = _latest_invocation_summaries(store)
    candidates: list[InvocationRecord] = []
    for raw in store.iter_invocations():
        invocation = InvocationRecord.model_validate(raw)
        summary = summaries.get(invocation.invocation_id)
        if invocation.configuration_id != configuration_id:
            continue
        if summary is None or summary.terminal_state in {"partial", "interrupted"}:
            candidates.append(invocation)
    if candidates:
        invocation = max(candidates, key=lambda item: (item.created_at or "", item.invocation_id))
    else:
        created_at = utc_now()
        invocation_id = sha256_text(canonical_json({
            "configuration_id": configuration_id,
            "created_at": created_at,
            "ordinal": len(store.iter_invocations()) + 1,
        }))
        invocation = InvocationRecord(
            invocation_id=invocation_id,
            configuration_id=configuration_id,
            configuration=configuration,
            requested_outer=int(arguments.get("rounds") or arguments.get("outer_rounds") or 0),
            inner_budget=int(arguments.get("inner_rounds") or arguments.get("inner_trials") or 1),
            created_at=created_at,
        )
        store.put_invocation(invocation)

    heads = store.slot_heads(invocation_id=invocation.invocation_id)
    for outer_idx in range(1, invocation.requested_outer + 1):
        for inner_idx in (None, *range(1, invocation.inner_budget + 1)):
            slot_id = _slot_id(invocation.configuration_id, outer_idx, inner_idx)
            if slot_id not in heads:
                store.append_slot_event(
                    None,
                    slot_id=slot_id,
                    invocation_id=invocation.invocation_id,
                    configuration_id=invocation.configuration_id,
                    outer_idx=outer_idx,
                    inner_idx=inner_idx,
                    state="pending",
                    created_at=utc_now(),
                )
    runner.invocation_id = invocation.invocation_id
    return invocation


def _slot_id(configuration_id: str, outer_idx: int, inner_idx: int | None) -> str:
    suffix = "outer" if inner_idx is None else f"inner-{inner_idx:03d}"
    return f"{configuration_id}:{outer_idx:03d}:{suffix}"


def _slot_head(
    store: LoopStore,
    invocation: InvocationRecord,
    outer_idx: int,
    inner_idx: int | None,
) -> SlotEventRecord | None:
    return store.slot_heads(invocation_id=invocation.invocation_id).get(
        _slot_id(invocation.configuration_id, outer_idx, inner_idx)
    )


def _transition_slot(
    store: LoopStore,
    invocation: InvocationRecord,
    *,
    outer_idx: int,
    inner_idx: int | None,
    state: str,
    result: dict[str, Any] | None = None,
) -> SlotEventRecord:
    previous = _slot_head(store, invocation, outer_idx, inner_idx)
    merged = dict(previous.result) if previous is not None else {}
    merged.update(result or {})
    return store.append_slot_event(
        previous,
        slot_id=_slot_id(invocation.configuration_id, outer_idx, inner_idx),
        invocation_id=invocation.invocation_id,
        configuration_id=invocation.configuration_id,
        outer_idx=outer_idx,
        inner_idx=inner_idx,
        state=state,
        algorithm_plan_id=merged.get("algorithm_plan_id"),
        patch_plan_id=merged.get("patch_plan_id"),
        patchset_id=merged.get("patchset_id"),
        eval_id=merged.get("final_eval_id") or merged.get("eval_id"),
        details=merged,
        result=merged,
        created_at=utc_now(),
    )


def _latest_invocation_summaries(store: LoopStore) -> dict[str, InvocationSummaryRecord]:
    latest: dict[str, InvocationSummaryRecord] = {}
    for raw in store.iter_invocation_summaries():
        record = InvocationSummaryRecord.model_validate(raw)
        previous = latest.get(record.invocation_id)
        if previous is None or (record.episode_idx, record.created_at or "", record.invocation_summary_id or "") > (
            previous.episode_idx,
            previous.created_at or "",
            previous.invocation_summary_id or "",
        ):
            latest[record.invocation_id] = record
    return latest


def _invocation_terminal_state(store: LoopStore, invocation: InvocationRecord) -> str:
    heads = store.slot_heads(invocation_id=invocation.invocation_id)
    # ``awaiting_proposal`` means the run is paused pending an out-of-process
    # proposal; like ``interrupted`` it must stay re-attachable.
    if any(head.state in {"interrupted", "awaiting_proposal"} for head in heads.values()):
        return "interrupted"
    # ``validated`` is terminal for the legacy ledger, where an outer slot is
    # only the plan slot.  The hierarchical runner advances it to committed or
    # not_committed before normal finalization.
    outer_terminal = {"validated", "committed", "not_committed", "blocked"}
    inner_terminal = {"evaluated", "blocked"}
    for outer_idx in range(1, invocation.requested_outer + 1):
        outer = heads.get(_slot_id(invocation.configuration_id, outer_idx, None))
        if outer is None or outer.state not in outer_terminal:
            return "partial"
        for inner_idx in range(1, invocation.inner_budget + 1):
            inner = heads.get(_slot_id(invocation.configuration_id, outer_idx, inner_idx))
            if inner is None or inner.state not in inner_terminal:
                return "partial"
    return "complete"


def _finalize_invocation(
    store: LoopStore,
    invocation: InvocationRecord,
    *,
    terminal_state: str,
) -> InvocationSummaryRecord:
    heads = list(store.slot_heads(invocation_id=invocation.invocation_id).values())
    previous = _latest_invocation_summaries(store).get(invocation.invocation_id)
    summary = InvocationSummaryRecord(
        invocation_id=invocation.invocation_id,
        parent_invocation_summary_id=previous.invocation_summary_id if previous else None,
        episode_idx=(previous.episode_idx + 1) if previous else 1,
        configuration_id=invocation.configuration_id,
        requested_outer=invocation.requested_outer,
        inner_budget=invocation.inner_budget,
        completed_slots=sum(1 for item in heads if item.inner_idx is not None and item.state == "evaluated"),
        blocked_slots=sum(1 for item in heads if item.inner_idx is not None and item.state == "blocked"),
        terminal_state=terminal_state,
        high_water_mark=store.canonical_high_water(),
        proposal_attempts=sum(1 for item in heads if item.state != "pending"),
        valid_plans=sum(1 for item in heads if item.inner_idx is None and item.algorithm_plan_id),
        evaluated_candidates=sum(1 for item in heads if item.inner_idx is not None and item.state == "evaluated"),
        created_at=utc_now(),
    )
    summary.invocation_summary_id = record_hash(summary, id_field="invocation_summary_id")
    store.put_invocation_summary(summary)
    return summary


def _rebuild_run_summary(store: LoopStore) -> RunSummaryRecord:
    invocations = [InvocationRecord.model_validate(item) for item in store.iter_invocations()]
    heads = list(store.slot_heads().values())
    results = [dict(head.result) for head in heads if head.result]
    latest = _latest_invocation_summaries(store)
    invocation_states = [
        latest[invocation.invocation_id].terminal_state
        if invocation.invocation_id in latest
        else _invocation_terminal_state(store, invocation)
        for invocation in invocations
    ]
    if "interrupted" in invocation_states:
        terminal = "interrupted"
    elif invocation_states and all(state == "complete" for state in invocation_states):
        terminal = "complete"
    else:
        terminal = "partial"
    summary = RunSummaryRecord(
        high_water_mark=store.canonical_high_water(),
        requested=sum(item.requested_outer * item.inner_budget for item in invocations),
        evaluated=sum(1 for item in heads if item.inner_idx is not None and item.state == "evaluated"),
        proposal_attempts=sum(1 for item in heads if item.state != "pending"),
        valid_plans=sum(1 for item in heads if item.inner_idx is None and item.algorithm_plan_id),
        evaluated_candidates=sum(1 for item in heads if item.inner_idx is not None and item.state == "evaluated"),
        blocked_slots=sum(1 for item in heads if item.inner_idx is not None and item.state == "blocked"),
        terminal_state=terminal,
        invocation_ids=sorted(item.invocation_id for item in invocations),
        results=results,
        created_at=utc_now(),
    )
    summary.run_summary_id = record_hash(summary, id_field="run_summary_id")
    store.write_index("run_summary", summary.model_dump(mode="json"))
    return summary


def _read_run_summary(store: LoopStore) -> RunSummaryRecord | None:
    raw = store.read_index("run_summary")
    if raw is None and not store.iter_invocations():
        return None
    if raw is not None:
        try:
            current = RunSummaryRecord.model_validate(raw)
            if current.high_water_mark == store.canonical_high_water():
                return current
        except (TypeError, ValueError):
            pass
    return _rebuild_run_summary(store)


def _write_run_summary(
    store: LoopStore,
    results: list[dict[str, Any]],
    *,
    requested: int,
) -> RunSummaryRecord:
    projected: list[dict[str, Any]] = []
    for item in results:
        row = dict(item)
        if row.get("final_eval_id"):
            outcome = "evaluated"
        elif row.get("static_eval_id"):
            outcome = "static_rejected"
        elif "proposal_error" in str(row.get("status") or ""):
            outcome = "proposal_error"
        else:
            outcome = "unknown"
        row["terminal_outcome"] = outcome
        projected.append(row)
    evaluated = sum(1 for item in projected if item["terminal_outcome"] == "evaluated")
    terminal = "complete" if len(projected) >= requested and all(
        item["terminal_outcome"] in {"evaluated", "static_rejected"} for item in projected
    ) else "partial"
    summary = RunSummaryRecord(
        high_water_mark=store.canonical_high_water(),
        requested=requested,
        evaluated=evaluated,
        proposal_attempts=len(projected),
        evaluated_candidates=evaluated,
        terminal_state=terminal,
        results=projected,
        created_at=utc_now(),
    )
    summary.run_summary_id = record_hash(summary, id_field="run_summary_id")
    store.write_index("run_summary", summary.model_dump(mode="json"))
    return summary


def _append_proposal_error_experience(store: LoopStore, record: dict[str, Any], *, outcome: str | None = None) -> None:
    status = str(record.get("status") or outcome or "proposal_error")
    mapped = outcome or {
        "proposal_error": "patch_proposal_error",
        "plan_proposal_error": "plan_proposal_error",
        "fatal_proposal_error": "fatal_proposal_error",
    }.get(status, status)
    append_and_compress(store, workflow_experience(
        outcome=mapped,
        lesson=(
            f"Round {record.get('round_idx')} failed before evaluation with status={status}: "
            f"{str(record.get('error') or '')[:500]}"
        ),
        severity="high" if mapped == "fatal_proposal_error" else "medium",
        metric_summary={
            "round_idx": record.get("round_idx"),
            "context_pack_id": record.get("context_pack_id"),
            "patch_plan_id": record.get("patch_plan_id"),
            "patchset_id": record.get("patchset_id"),
            "static_eval_id": record.get("static_eval_id"),
            "final_eval_id": record.get("final_eval_id"),
        },
        patch_signature={
            "patch_plan_id": record.get("patch_plan_id"),
            "patchset_id": record.get("patchset_id"),
            "round_idx": record.get("round_idx"),
        },
        negative_patterns=[mapped, status],
        retrieval_tags=[mapped, status],
    ))


def _should_repair_final(record) -> bool:
    return _repairable_final_failure_kind(record) is not None


def _repairable_final_failure_kind(record) -> str | None:
    """Classify deterministic implementation failures eligible for repair."""
    if record.decision == "rejected_plan_diff_binding":
        return "plan_diff_binding_failure"
    if record.status == "compile_error":
        return "compile_error"
    if record.decision == "rejected_not_runtime_effective":
        return "runtime_provenance_failure"
    if record.status != "run_error":
        return None
    errors = _record_errors(record)
    if _looks_like_patch_apply_failure(errors):
        return "patch_apply_failure"
    text = "\n".join(errors).lower()
    environmental_markers = (
        "timeout",
        "timed out",
        "signal 9",
        "out of memory",
        "oom",
        "slurm",
        "node failure",
        "cancelled",
        "filesystem",
        "no space left",
    )
    if any(marker in text for marker in environmental_markers):
        return None
    return "semantic_runtime_failure" if text else None


def _failure_kind_for_feedback(record) -> str:
    return _repairable_final_failure_kind(record) or "environmental_failure"



def _record_errors(record) -> list[str]:
    data = record.model_dump(mode="json") if hasattr(record, "model_dump") else dict(record)
    result = data.get("eval_result") or {}
    errors: list[str] = []
    for value in (data.get("errors"), result.get("errors")):
        if isinstance(value, list):
            errors.extend(str(item) for item in value)
        elif value:
            errors.append(str(value))
    meta = result.get("meta") or {}
    repeat_results = meta.get("repeat_results") or []
    if isinstance(repeat_results, list):
        for repeat in repeat_results:
            if isinstance(repeat, dict):
                repeat_errors = repeat.get("errors") or []
                if isinstance(repeat_errors, list):
                    errors.extend(str(item) for item in repeat_errors)
                elif repeat_errors:
                    errors.append(str(repeat_errors))
    return errors



def _looks_like_patch_apply_failure(errors: list[str]) -> bool:
    text = "\n".join(errors).lower()
    if not text:
        return False
    apply_markers = (
        "git apply --check",
        "git apply",
        "patch does not apply",
        "patch failed",
        "corrupt patch",
        "unrecognized input",
    )
    return any(marker in text for marker in apply_markers)



def _propose_valid_plan(
    proposer: BaseProposer,
    runner: PatchRunner,
    pack,
    *,
    parent_patchset_id: str | None,
    max_repair_attempts: int,
    initial_feedback: str | None = None,
    initial_parent_patch_plan_id: str | None = None,
    plan_reproposal_idx: int = 0,
):
    feedback = initial_feedback
    rejected_parent = initial_parent_patch_plan_id
    for attempt_idx in range(max_repair_attempts + 1):
        try:
            plan_proposal, plan_metadata = proposer.propose_plan(
                pack,
                feedback=feedback,
                attempt_idx=attempt_idx if attempt_idx else None,
            )
        except TypeError as exc:
            if "feedback" not in str(exc) and "attempt_idx" not in str(exc):
                raise
            if feedback is not None:
                raise ProposalError("legacy proposer cannot repair rejected patch plan") from exc
            plan_proposal, plan_metadata = proposer.propose_plan(pack)
        patch_plan = runner.build_patch_plan(
            plan_proposal,
            pack,
            {
                **plan_metadata,
                "plan_attempt_idx": attempt_idx,
                "plan_reproposal_idx": plan_reproposal_idx,
            },
            parent_patch_plan_id=rejected_parent,
            parent_patchset_id=parent_patchset_id,
        )
        if patch_plan.validation_status == "ok":
            return plan_proposal, patch_plan
        rejected_parent = patch_plan.patch_plan_id
        feedback = _plan_repair_feedback(patch_plan)
    raise ProposalError(f"patch plan validation failed after {max_repair_attempts} repairs: {feedback}")



def _plan_repair_feedback(patch_plan) -> str:
    return json.dumps(
        {
            "patch_plan_id": patch_plan.patch_plan_id,
            "component_name": patch_plan.component_name,
            "target_functions": patch_plan.c_target_functions,
            "validation_errors": patch_plan.validation_errors,
            "repair_instruction": (
                "Return a corrected patch-plan JSON object only. Keep component_name in the allowed C "
                "contract, target at least one allowed function, describe executable runtime C logic, "
                "and explicitly state how TP/recall is preserved. Include profile_evidence_citation; "
                "when measured profile evidence is present, cite the dominant command/query hotspot and do not "
                "target a non-dominant command-only path. When symbol/function profile evidence is present, "
                "cite the measured symbol/function/contract hotspot or explain any target mismatch. "
                "Do not include unified_diff or python_payload."
            ),
        },
        sort_keys=True,
    )



def _legacy_algorithm_plan_from_patch_plan(plan: PatchPlanProposal) -> AlgorithmPlanProposal:
    """Expose a validated legacy patch plan through the modern repair API."""
    family = {
        "dp_kernel_neon": "dp_recurrence_strategy",
        "ssv_filter_neon": "ssv_filter_strategy",
        "null2_correction": "null2_scoring_strategy",
        "null2_correction_neon": "null2_scoring_strategy",
        "sequence_io": "sequence_streaming_strategy",
    }.get(plan.component_name, "profile_target_comparison")
    return AlgorithmPlanProposal(
        algorithm_family=family,
        baseline_algorithm_summary=(
            f"Legacy bounded search currently executes the {plan.component_name} path."
        ),
        proposed_algorithm_change=plan.algorithm_hypothesis,
        speed_mechanism=plan.runtime_effective_argument,
        quality_preservation_argument=plan.anti_degenerate_check,
        target_contracts=[plan.component_name],
        target_functions=list(plan.target_functions),
        exact_fallback_strategy=plan.anti_degenerate_check,
        profile_evidence_citation=plan.profile_evidence_citation,
        profile_evidence_refs=list(plan.profile_evidence_refs),
        implementation_constraints=[plan.intended_code_change],
        risk_notes=plan.risk_notes,
        related_experience_ids=list(plan.related_experience_ids),
    )


def _propose_patch(
    proposer: BaseProposer,
    pack,
    *,
    plan,
    algorithm_plan: AlgorithmPlanProposal | None = None,
):
    """Call the new plan-aware proposer API with a legacy fallback for tests."""
    try:
        return proposer.propose(pack, plan=plan, algorithm_plan=algorithm_plan)
    except TypeError as exc:
        if "algorithm_plan" not in str(exc) and "plan" not in str(exc):
            raise
    try:
        return proposer.propose(pack, plan=plan)
    except TypeError as exc:
        if "plan" not in str(exc):
            raise
        return proposer.propose(pack)


def _repair_patch(
    proposer: BaseProposer,
    pack,
    *,
    previous,
    feedback: str,
    attempt_idx: int,
    plan,
    algorithm_plan: AlgorithmPlanProposal | None = None,
):
    """Call the new plan-aware repair API with a legacy fallback for tests."""
    try:
        return proposer.repair(
            pack,
            previous=previous,
            feedback=feedback,
            attempt_idx=attempt_idx,
            plan=plan,
            algorithm_plan=algorithm_plan,
        )
    except TypeError as exc:
        if "algorithm_plan" not in str(exc) and "plan" not in str(exc):
            raise
    try:
        return proposer.repair(pack, previous=previous, feedback=feedback, attempt_idx=attempt_idx, plan=plan)
    except TypeError as exc:
        if "plan" not in str(exc):
            raise
        return proposer.repair(pack, previous=previous, feedback=feedback, attempt_idx=attempt_idx)



def _baseline_summary(
    store: LoopStore,
    *,
    runner: PatchRunner | None = None,
    stage: str = "full",
):
    if runner is not None and hasattr(runner, "runtime_identity"):
        # An unavailable runtime identity is not a cache mismatch. Resolve it
        # before deciding whether an automatic Full baseline is necessary.
        runner.runtime_identity()
    current = store.read_index(f"baseline_{stage}") or (store.read_index("baseline") if stage == "full" else None)
    if current and current.get("stage") == stage and _cache_matches_runner(current, runner, stage=stage):
        return eval_summary_from_record(current)
    for path in sorted(store.evaluations_dir.glob("*.json")):
        data = load_json(path)
        if (
            data.get("stage") == stage
            and data.get("status") == "ok"
            and data.get("decision") == "accepted_baseline"
            and _cache_matches_runner(data, runner, stage=stage)
        ):
            return eval_summary_from_record(data)
    return None


def _current_best_summary(
    store: LoopStore,
    *,
    runner: PatchRunner | None = None,
    stage: str = "full",
    accept_min_tp_fraction: float = 1.0,
):
    current = store.read_index(f"current_speed_best_{stage}") or store.read_index(f"current_best_{stage}")
    if current and current.get("stage") == stage and _cache_matches_runner(current, runner, stage=stage):
        return eval_summary_from_record(current)
    if stage == "full":
        current = store.read_index("current_speed_best") or store.read_index("current_best")
        if current and current.get("stage") == "full" and _cache_matches_runner(current, runner, stage=stage):
            return eval_summary_from_record(current)
    evals = []
    for path in sorted(store.evaluations_dir.glob("*.json")):
        data = load_json(path)
        if (
            data.get("stage") == stage
            and data.get("status") == "ok"
            and str(data.get("decision", "")).startswith("accepted")
            and _cache_matches_runner(data, runner, stage=stage)
        ):
            evals.append(data)
    if evals:
        max_tp = max(int((item.get("eval_result") or {}).get("tp", 0)) for item in evals)
        min_tp = max_tp * accept_min_tp_fraction
        eligible = [item for item in evals if int((item.get("eval_result") or {}).get("tp", 0)) + 1e-12 >= min_tp]
        best = min(eligible or evals, key=lambda item: _speed_eval_sort_key(item.get("eval_result") or {}))
        return eval_summary_from_record(best)
    return None


def _cache_matches_runner(data: dict[str, Any], runner: PatchRunner | None, *, stage: str) -> bool:
    if runner is None or not hasattr(runner, "reuse_fingerprint"):
        return True
    cached = (data.get("eval_config") or {}).get("reuse_fingerprint")
    if cached:
        return str(cached) == str(runner.reuse_fingerprint(stage=stage))
    return _backfill_legacy_baseline_reuse(data, runner, stage=stage)


def _backfill_legacy_baseline_reuse(
    data: dict[str, Any],
    runner: PatchRunner,
    *,
    stage: str,
) -> bool:
    """Attest one pre-fingerprint Slurm baseline before reusing it.

    Evaluation records remain immutable.  A successful migration writes the
    derived fingerprint only to the mutable baseline index, alongside hashes
    of every archived query/result artifact used for the decision.  Any
    missing or inconsistent legacy evidence fails closed and forces a fresh
    baseline instead.
    """
    if (
        data.get("stage") != stage
        or data.get("status") != "ok"
        or data.get("decision") != "accepted_baseline"
    ):
        return False
    eval_id = str(data.get("eval_id") or "")
    if not eval_id:
        return False
    eval_path = runner.store.evaluations_dir / f"{eval_id}.json"
    if not eval_path.is_file():
        return False
    try:
        record = load_json(eval_path)
        if not isinstance(record, dict) or record_hash(record, id_field="eval_id") != eval_id:
            return False
        for key in ("eval_id", "patchset_id", "stage", "status", "decision"):
            if data.get(key) != record.get(key):
                return False
        attestation = _legacy_slurm_baseline_attestation(record, runner, stage=stage)
        if attestation is None:
            return False
        fingerprint = runner.reuse_fingerprint(
            stage=stage,
            runtime_identity=attestation["runtime_identity"],
        )
        attestation.update({
            "schema_version": "legacy_baseline_reuse_attestation.v2",
            "eval_id": eval_id,
            "reuse_fingerprint": fingerprint,
            "evaluation_record_sha256": hash_file(eval_path),
            "created_at": utc_now(),
        })
        runner.store.write_index(f"legacy_baseline_reuse_{stage}", attestation)
        for index_name in (
            f"baseline_{stage}",
            "baseline" if stage == "full" else "",
        ):
            if not index_name:
                continue
            current = runner.store.read_index(index_name)
            if not isinstance(current, dict) or current.get("eval_id") != eval_id:
                continue
            migrated = dict(current)
            migrated["eval_config"] = {
                **dict(record.get("eval_config") or {}),
                "reuse_fingerprint": fingerprint,
            }
            migrated["legacy_reuse_attestation"] = {
                "schema_version": attestation["schema_version"],
                "eval_id": eval_id,
                "reuse_fingerprint": fingerprint,
                "evaluation_record_sha256": attestation["evaluation_record_sha256"],
            }
            runner.store.write_index(index_name, migrated)
        return True
    except (OSError, TypeError, ValueError, KeyError, json.JSONDecodeError):
        return False


def _legacy_slurm_baseline_attestation(
    record: dict[str, Any],
    runner: PatchRunner,
    *,
    stage: str,
) -> dict[str, Any] | None:
    result = record.get("eval_result") or {}
    meta = result.get("meta") or {}
    if result.get("status") != "ok" or meta.get("execution") != "slurm_array":
        return None
    legacy_config = record.get("eval_config") or {}
    data_dir = runner._data_dir_for_stage(stage)
    uniref_db = runner._uniref_for_stage(stage, data_dir)
    max_iter = runner.full_max_iter if stage == "full" else runner.max_iter
    expected_config = runner._eval_config(
        stage=stage,
        max_iter=max_iter,
        work_root=None,
        data_dir=data_dir,
        uniref_db=uniref_db,
    )
    for key, expected in expected_config.items():
        if key == "work_root" or (key not in legacy_config and expected is None):
            continue
        if legacy_config.get(key) != expected:
            return None
    if record.get("source_hashes") != _stock_source_hashes(runner.hmmer_src_root):
        return None
    current_runtime_identity = runner.runtime_identity()
    current_binary_fingerprint = current_runtime_identity.get("binaries") or {}
    current_binary_hashes = tuple(
        str((current_binary_fingerprint.get(name) or {}).get("sha256") or "")
        for name in ("jackhmmer", "hmmsearch")
    )
    if any(len(digest) != 64 for digest in current_binary_hashes):
        return None

    manifest = record.get("dataset_manifest") or {}
    current_queries = runner._query_manifest(data_dir)
    requested = int(manifest.get("query_count") or 0)
    if requested <= 0 or requested != len(current_queries) or int(result.get("n_seqs") or 0) != requested:
        return None
    if manifest.get("data_dir") != str(data_dir):
        return None
    if list(manifest.get("queries") or []) != [Path(item["query_path"]).name for item in current_queries]:
        return None
    for label, path in (("astral_db", runner.astral_db), ("scop_txt", runner.scop_txt)):
        stored = manifest.get(label) or {}
        if (
            stored.get("path") != str(path)
            or int(stored.get("size") or -1) != path.stat().st_size
            or stored.get("sha256") != hash_file(path)
        ):
            return None
    stored_uniref = manifest.get("uniref_db") or {}
    current_uniref = _file_info(uniref_db, sha256=False)
    uniref_identity_fields = ("path", "exists", "size", "device", "inode", "mtime_ns")
    if any(
        key not in stored_uniref or stored_uniref.get(key) != current_uniref.get(key)
        for key in uniref_identity_fields
    ):
        return None

    job_dir = Path(str(meta.get("job_dir") or ""))
    request_path = job_dir / "request.json"
    submission_path = job_dir / "submission.json"
    if not job_dir.is_dir() or not request_path.is_file() or not submission_path.is_file():
        return None
    request = load_json(request_path)
    submission = load_json(submission_path)
    if not isinstance(request, dict) or not isinstance(submission, dict):
        return None
    if request.get("stage") != stage or int(request.get("max_iter") or -1) != max_iter:
        return None
    if request.get("mods") not in ([], None):
        return None
    source_paths = request.get("source_paths") or {}
    for key in ("data_dir", "astral_db", "uniref_db", "scop_txt"):
        if source_paths.get(key) != legacy_config.get(key):
            return None
    request_queries = list(request.get("query_paths") or [])
    if len(request_queries) != requested:
        return None

    rows = list(meta.get("query_results") or [])
    costs = list(meta.get("query_cost_s") or [])
    if len(rows) != requested or len(costs) != requested:
        return None
    rows_by_idx = {int(row.get("query_index", -1)): row for row in rows if isinstance(row, dict)}
    if sorted(rows_by_idx) != list(range(requested)):
        return None
    artifact_hashes = {
        "request.json": hash_file(request_path),
        "submission.json": hash_file(submission_path),
    }
    binary_hashes: set[tuple[str, str]] = set()
    total_tp = total_fp = 0
    total_cost = 0.0
    archived_queries: list[Path] = []
    for idx, current in enumerate(current_queries):
        archived_query = job_dir / "queries" / f"query_{idx:03d}.fasta"
        result_path = job_dir / "results" / f"result_{idx:03d}.json"
        if not archived_query.is_file() or not result_path.is_file():
            return None
        archived_queries.append(archived_query)
        if hash_file(archived_query) != current["query_sha256"]:
            return None
        query_result = load_json(result_path)
        row = rows_by_idx[idx]
        query_meta = query_result.get("meta") or {}
        if (
            query_result.get("stage") != stage
            or query_result.get("status") != "ok"
            or int(query_result.get("n_seqs") or 0) != 1
            or int(query_meta.get("query_index", -1)) != idx
            or int(query_meta.get("max_iter") or -1) != max_iter
            or (query_meta.get("build") or {}).get("runtime_effective") is not True
        ):
            return None
        for key in ("status", "tp", "fp", "iterations", "cost_s"):
            if row.get(key) != query_result.get(key):
                return None
        if float(costs[idx]) != float(query_result.get("cost_s") or 0.0):
            return None
        jack_hash = str(query_meta.get("jackhmmer_sha256") or "")
        search_hash = str(query_meta.get("hmmsearch_sha256") or "")
        if len(jack_hash) != 64 or len(search_hash) != 64:
            return None
        binary_hashes.add((jack_hash, search_hash))
        total_tp += int(query_result.get("tp") or 0)
        total_fp += int(query_result.get("fp") or 0)
        total_cost += float(query_result.get("cost_s") or 0.0)
        artifact_hashes[f"queries/query_{idx:03d}.fasta"] = hash_file(archived_query)
        artifact_hashes[f"results/result_{idx:03d}.json"] = hash_file(result_path)
    run_jack.validate_query_fastas([str(path) for path in archived_queries])
    if len(binary_hashes) != 1:
        return None
    recorded_binary_hashes = next(iter(binary_hashes))
    if recorded_binary_hashes != current_binary_hashes:
        return None
    if (
        total_tp != int(result.get("tp") or 0)
        or total_fp != int(result.get("fp") or 0)
        or abs(round(total_cost, 4) - float(result.get("cost_s") or 0.0)) > 1e-9
    ):
        return None
    return {
        "stage": stage,
        "query_count": requested,
        "binary_sha256": {
            "jackhmmer": recorded_binary_hashes[0],
            "hmmsearch": recorded_binary_hashes[1],
        },
        "runtime_identity": current_runtime_identity,
        "artifact_hashes": artifact_hashes,
    }



def _speed_eval_sort_key(result: dict[str, Any]) -> tuple[float, int, int]:
    return (float(result.get("cost_s", 0.0)), -int(result.get("tp", 0)), int(result.get("fp", 0)))



def _repair_feedback(*records: Any) -> str:
    """Return compact, structured, sanitized validation feedback for repair."""
    payload: list[dict[str, Any]] = []
    for record in records:
        data = record.model_dump(mode="json") if hasattr(record, "model_dump") else dict(record)
        result = data.get("eval_result") or {}
        build = data.get("build") or {}
        comparisons = build.get("comparisons") or {}
        payload.append({
            "eval_id": data.get("eval_id"),
            "patchset_id": data.get("patchset_id"),
            "stage": data.get("stage"),
            "status": data.get("status"),
            "decision": data.get("decision"),
            "failure_kind": _failure_kind_for_feedback(record),
            "violations": data.get("violations") or [],
            "errors": _record_errors(record),
            "result_status": result.get("status"),
            "metrics": {
                key: result[key]
                for key in ("tp", "fp", "tp_fp_ratio", "cost_s", "n_seqs", "iterations")
                if key in result
            },
            "metric_delta": data.get("metric_delta") or {},
            "runtime_effective": build.get("runtime_effective"),
            "plan_diff_binding": data.get("plan_diff_binding"),
            "build_comparisons": {
                "target_sources": _comparison_changed(comparisons.get("target_sources")),
                "target_objects": _comparison_changed(comparisons.get("target_objects")),
                "libraries": _comparison_changed(comparisons.get("libraries")),
                "binaries": _comparison_changed(comparisons.get("binaries")),
            },
        })
    feedback = {
        "feedback_kind": "patch_repair",
        "instructions": (
            "Fix only the unified_diff and remain within the approved bounded C-patch plan. "
            "If runtime_effective is false, make an executable-affecting C logic change, "
            "not a comment, unreachable guard, or optimized-away no-op."
        ),
        "validation_records": payload,
    }
    return json.dumps(
        sanitize_sensitive_data(feedback, replacement="<redacted>"),
        ensure_ascii=False,
        sort_keys=True,
        default=str,
    )


def _plan_reproposal_feedback(*records: Any) -> str:
    """Escalate exhausted patch-level evidence to a fresh bounded plan."""
    failure_kind = _failure_kind_for_feedback(records[-1]) if records else "implementation_failure"
    feedback = {
        "feedback_kind": "implementation_path_failed_after_patch_repairs",
        "failure_kind": failure_kind,
        "instructions": (
            "The approved implementation path exhausted its bounded patch repairs. "
            "Propose a materially different implementation plan within the same context; "
            "do not merely restate or cosmetically rewrite the failed patch."
        ),
        "failed_implementation": json.loads(_repair_feedback(*records)),
    }
    return json.dumps(
        sanitize_sensitive_data(feedback, replacement="<redacted>"),
        ensure_ascii=False,
        sort_keys=True,
        default=str,
    )



def _comparison_changed(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: v.get("changed") for k, v in value.items() if isinstance(v, dict)}
    if isinstance(value, list):
        return [item.get("changed") for item in value if isinstance(item, dict)]
    return None


def make_proposer(args: argparse.Namespace) -> BaseProposer:
    if args.proposer == "fixture":
        return FixtureProposer(hmmer_src_root=args.hmmer_src_root)
    return OpenAICompatibleProposer(
        model=args.model,
        base_url=args.base_url,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        timeout=args.timeout,
    )


class _DeferredProposer:
    """Resolve CLI credentials only when a run actually needs a proposal."""

    def __init__(self, factory):
        self._factory = factory
        self._resolved: BaseProposer | None = None

    def __getattr__(self, name: str) -> Any:
        if self._resolved is None:
            self._resolved = self._factory()
        return getattr(self._resolved, name)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Deterministic seq_align C-patch optimization loop")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--start-round", type=int, default=1)
    parser.add_argument("--proposer", choices=["openai", "fixture"], default="openai")
    parser.add_argument("--stage", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--cpus", type=int, default=1)
    parser.add_argument("--max-iter", type=int, default=1)
    parser.add_argument("--full-max-iter", type=int, default=None)
    parser.add_argument("--hmmer-src-root", default=None)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--astral-db", default=None)
    parser.add_argument("--uniref-db", default=None)
    parser.add_argument("--scop-txt", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument(
        "--query-timeout",
        type=float,
        default=None,
        help="Optional hard wall-clock limit in seconds for each benchmark query.",
    )
    parser.add_argument("--max-repair-attempts", type=int, default=2)
    parser.add_argument("--max-plan-reproposal-attempts", type=int, default=0)
    parser.add_argument("--accept-min-tp-fraction", type=float, default=1.0)
    parser.add_argument("--min-speedup-fraction", type=float, default=0.03)
    parser.add_argument("--eval-repeats", type=int, default=1)
    parser.add_argument("--paired-current-best-repeats", type=int, default=0)
    parser.add_argument("--near-speedup-band-fraction", type=float, default=0.02)
    parser.add_argument(
        "--external-profiler",
        choices=["off", "perf", "gprof"],
        default="off",
        help=(
            "Symbol-level profiler for the post-hoc profiling pass only. It is no "
            "longer applied to every evaluation, so it costs the inner loop nothing."
        ),
    )
    parser.add_argument(
        "--posthoc-profile",
        choices=["off", "auto", "local", "slurm"],
        default=os.getenv("SEQ_ALIGN_POSTHOC_PROFILE", "auto"),
        help=(
            "Run one instrumented stage-timing evaluation for the round-000 baseline "
            "and for each committed winner. 'auto' follows --eval-backend."
        ),
    )
    parser.add_argument("--perf-top-queries", type=int, default=0)
    parser.add_argument("--perf-top-symbols", type=int, default=20)
    parser.add_argument("--perf-frequency", type=int, default=99)
    parser.add_argument("--perf-command", choices=["dominant", "jackhmmer", "hmmsearch", "both"], default="dominant")
    parser.add_argument("--pyhmmer-profile-prepass", action="store_true")
    parser.add_argument("--pyhmmer-profile-prepass-stage", choices=["smoke", "full"], default="full")
    parser.add_argument("--pyhmmer-profile-max-target-bytes", type=int, default=None)
    parser.add_argument("--wisdom-dir", default=None)
    parser.add_argument("--disable-durable-wisdom", action="store_true")
    parser.add_argument("--import-seed-wisdom", action="store_true")
    parser.add_argument("--promote-wisdom", action="store_true")
    parser.add_argument(
        "--eval-backend",
        choices=["local", "slurm"],
        default=os.getenv("SEQ_ALIGN_EVAL_BACKEND", "slurm"),
        help="Run expensive evaluations locally or as isolated Slurm query arrays.",
    )
    parser.add_argument("--allow-parameter-only-patches", action="store_true")
    parser.add_argument("--keep-work", action="store_true")
    args = parser.parse_args(argv)

    try:
        results = run_loop(
            run_dir=args.run_dir,
            rounds=args.rounds,
            start_round=args.start_round,
            proposer=_DeferredProposer(lambda: make_proposer(args)),  # type: ignore[arg-type]
            stage=args.stage,
            cpus=args.cpus,
            max_iter=args.max_iter,
            full_max_iter=args.full_max_iter,
            hmmer_src_root=args.hmmer_src_root,
            data_dir=args.data_dir,
            astral_db=args.astral_db,
            uniref_db=args.uniref_db,
            scop_txt=args.scop_txt,
            keep_work=args.keep_work,
            max_repair_attempts=args.max_repair_attempts,
            max_plan_reproposal_attempts=args.max_plan_reproposal_attempts,
            accept_min_tp_fraction=args.accept_min_tp_fraction,
            min_speedup_fraction=args.min_speedup_fraction,
            eval_repeats=args.eval_repeats,
            paired_current_best_repeats=args.paired_current_best_repeats,
            near_speedup_band_fraction=args.near_speedup_band_fraction,
            require_algorithmic_patch=not args.allow_parameter_only_patches,
            external_profiler=args.external_profiler,
            perf_top_queries=args.perf_top_queries,
            perf_top_symbols=args.perf_top_symbols,
            perf_frequency=args.perf_frequency,
            perf_command=args.perf_command,
            query_timeout=args.query_timeout,
            eval_backend=args.eval_backend,
            posthoc_profile=args.posthoc_profile,
            pyhmmer_profile_prepass=args.pyhmmer_profile_prepass,
            pyhmmer_profile_prepass_stage=args.pyhmmer_profile_prepass_stage,
            pyhmmer_profile_max_target_bytes=args.pyhmmer_profile_max_target_bytes,
            wisdom_dir=args.wisdom_dir,
            disable_durable_wisdom=args.disable_durable_wisdom,
            import_seed_wisdom=args.import_seed_wisdom,
            promote_wisdom=args.promote_wisdom,
        )
    except run_hmmer_cli.EvaluationInterrupted as exc:
        parser.exit(2, f"evaluation interrupted: {exc}\n")
    for result in results:
        print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
