"""Experience records and lightweight compression for C-patch loop evidence."""

from __future__ import annotations

import json
import re
from collections import Counter
from typing import Any

from .models import AlgorithmPlanRecord, EvaluationRecord, ExperienceRecord, PatchsetRecord
from .store import LoopStore, canonical_json, record_hash, sha256_text, utc_now


WORKFLOW_FAILURE_OUTCOMES = {
    "compile_failure",
    "patch_apply_failure",
    "runtime_failure",
    "runtime_provenance_failure",
    "scope_violation",
    "plan_validation_failure",
    "plan_proposal_error",
    "patch_proposal_error",
    "repair_proposal_error",
    "fatal_proposal_error",
}
POSITIVE_OUTCOMES = {"accepted_speedup", "accepted_speedup_paired", "accepted_quality_preserving_candidate", "stock_baseline"}

_PROJECTION_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "because", "by", "for",
    "from", "in", "into", "is", "it", "of", "on", "or", "that", "the",
    "this", "to", "use", "uses", "using", "while", "with", "without",
    "should", "would", "could", "must", "will", "same", "stock",
}


def algorithm_mechanism_projection(value: Any) -> dict[str, Any]:
    """Return a stable outer-algorithm identity projection.

    The projection intentionally excludes prose about novelty, profile citations,
    proposal metadata, and evidence IDs.  Those fields describe why a mechanism
    was selected, not what the mechanism is.
    """
    family = _field(value, "algorithm_family") or "unknown"
    contracts = _string_list(_field(value, "target_contracts"))
    if not contracts:
        component = _field(value, "component_name")
        contracts = [str(component)] if component else []
    functions = _string_list(_field(value, "target_functions"))
    change = (
        _field(value, "proposed_algorithm_change")
        or _field(value, "mechanism")
        or _field(value, "lesson")
        or ""
    )
    speed = _field(value, "speed_mechanism") or _field(value, "runtime_effective_argument") or ""
    fallback = _field(value, "exact_fallback_strategy") or _field(value, "quality_preservation_argument") or ""
    return {
        "schema_version": "1.0",
        "track": "outer_algorithm",
        "algorithm_family": str(family).strip().lower(),
        "target_contracts": sorted(set(contracts)),
        "target_functions": sorted(set(functions)),
        "mechanism_terms": _normalized_terms(f"{change} {speed}"),
        "fallback_terms": _normalized_terms(str(fallback)),
    }


def implementation_mechanism_projection(value: Any) -> dict[str, Any]:
    """Return a stable inner implementation identity projection."""
    component = _field(value, "component_name") or "unknown"
    functions = _string_list(
        _field(value, "target_functions")
        or _field(value, "c_target_functions")
    )
    symbols = _string_list(_field(value, "target_symbols"))
    hypothesis = _field(value, "algorithm_hypothesis") or ""
    change = (
        _field(value, "intended_code_change")
        or _field(value, "rationale")
        or _field(value, "lesson")
        or ""
    )
    runtime = _field(value, "runtime_effective_argument") or ""
    fallback = _field(value, "anti_degenerate_check") or ""
    diff_sha = _field(value, "diff_sha256")
    return {
        "schema_version": "1.0",
        "track": "inner_implementation",
        "component_name": str(component).strip().lower(),
        "target_functions": sorted(set(functions)),
        "target_symbols": sorted(set(symbols)),
        "mechanism_terms": _normalized_terms(f"{hypothesis} {change} {runtime}"),
        "fallback_terms": _normalized_terms(str(fallback)),
        "diff_sha256": str(diff_sha) if diff_sha else None,
    }


def mechanism_fingerprint(projection: dict[str, Any]) -> str:
    """Content hash for one canonical mechanism projection."""
    return sha256_text(canonical_json(projection))


def workflow_mechanism_projection(
    *,
    outcome: str,
    failure_stage: str | None,
    failure_class: str | None,
    component_name: str | None,
) -> dict[str, Any]:
    """Keep infrastructure failures out of algorithm/implementation identity."""
    return {
        "schema_version": "1.0",
        "track": "workflow",
        "outcome": outcome,
        "failure_stage": failure_stage,
        "failure_class": failure_class,
        "component_name": component_name,
    }


def _field(value: Any, name: str) -> Any:
    if isinstance(value, dict):
        return value.get(name)
    return getattr(value, name, None)


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return [str(item) for item in value if item is not None]


def _normalized_terms(text: str) -> list[str]:
    words = re.findall(r"[a-z][a-z0-9_]+", text.lower())
    normalized: set[str] = set()
    for word in words:
        if word in _PROJECTION_STOPWORDS:
            continue
        # Minimal morphology makes reordered/rephrased mechanism prose compare
        # sensibly without pretending to be a semantic embedding model.
        for suffix in ("ing", "ed", "es", "s"):
            if word.endswith(suffix) and len(word) > len(suffix) + 3:
                word = word[:-len(suffix)]
                break
        normalized.add(word)
    return sorted(normalized)


def experience_from_evaluation(
    patchset: PatchsetRecord,
    evaluation: EvaluationRecord,
    *,
    lesson: str | None = None,
) -> ExperienceRecord:
    """Create one evidence-linked lesson from an evaluation outcome."""
    patch = patchset.patches[-1] if patchset.patches else None
    result = evaluation.eval_result or {}
    outcome = _outcome(evaluation)
    scope = "workflow" if outcome in WORKFLOW_FAILURE_OUTCOMES else "inner_implementation"
    failure_stage = evaluation.stage if scope == "workflow" else None
    failure_class = outcome if scope == "workflow" else None
    if scope == "workflow":
        projection = workflow_mechanism_projection(
            outcome=outcome,
            failure_stage=failure_stage,
            failure_class=failure_class,
            component_name=patch.component_name if patch else None,
        )
    else:
        projection = implementation_mechanism_projection(patch or {})
    if lesson is None:
        lesson = _default_lesson(patch.component_name if patch else None, evaluation, result)
    record = ExperienceRecord(
        source_eval_ids=[evaluation.eval_id] if evaluation.eval_id else [],
        source_patchset_ids=[patchset.patchset_id] if patchset.patchset_id else [],
        source_algorithm_plan_ids=[patchset.algorithm_plan_id] if patchset.algorithm_plan_id else [],
        source_patch_plan_ids=[patchset.patch_plan_id] if patchset.patch_plan_id else [],
        component_name=patch.component_name if patch else None,
        scope=scope,
        polarity=_polarity(outcome),
        failure_stage=failure_stage,
        failure_class=failure_class,
        outcome=outcome,
        lesson_kind=_lesson_kind(outcome),
        severity=_severity(outcome),
        metric_summary=_metric_summary(evaluation, result),
        patch_signature=_patch_signature(patchset),
        lesson=lesson,
        positive_patterns=_positive_patterns(evaluation, result),
        negative_patterns=_negative_patterns(evaluation, result),
        applicability=[patch.component_name] if patch else ["workflow"],
        retrieval_tags=_retrieval_tags(outcome, evaluation, patch.component_name if patch else None),
        mechanism_projection=projection,
        mechanism_fingerprint=mechanism_fingerprint(projection),
        created_at=utc_now(),
    )
    return _finalize_experience(record)


def workflow_experience(
    *,
    outcome: str,
    lesson: str,
    severity: str = "medium",
    component_name: str | None = None,
    source_patchset_ids: list[str] | None = None,
    metric_summary: dict[str, Any] | None = None,
    patch_signature: dict[str, Any] | None = None,
    negative_patterns: list[str] | None = None,
    retrieval_tags: list[str] | None = None,
) -> ExperienceRecord:
    """Create a workflow-level lesson not backed by a completed EvaluationRecord."""
    failure_stage = str((metric_summary or {}).get("stage") or (metric_summary or {}).get("failure_stage") or "proposal")
    projection = workflow_mechanism_projection(
        outcome=outcome,
        failure_stage=failure_stage,
        failure_class=outcome,
        component_name=component_name,
    )
    record = ExperienceRecord(
        source_patchset_ids=source_patchset_ids or [],
        component_name=component_name,
        scope="workflow",
        polarity="negative",
        failure_stage=failure_stage,
        failure_class=outcome,
        outcome=outcome,
        lesson_kind="workflow",
        severity=severity,
        metric_summary=metric_summary or {},
        patch_signature=patch_signature or {},
        lesson=lesson,
        negative_patterns=negative_patterns or [outcome],
        applicability=[component_name] if component_name else ["workflow"],
        retrieval_tags=sorted({*(retrieval_tags or []), outcome, "workflow", "repair"}),
        mechanism_projection=projection,
        mechanism_fingerprint=mechanism_fingerprint(projection),
        created_at=utc_now(),
    )
    return _finalize_experience(record)


def append_and_compress(store: LoopStore, record: ExperienceRecord) -> None:
    """Append raw experience and refresh derived indexes and compact summaries.

    A completed Full evaluation carries two different kinds of evidence.  The
    implementation record identifies the concrete patch mechanism, while the
    derived algorithm record identifies the approved outer plan without leaking
    a diff into the algorithm track.  Smoke evidence remains implementation-only.
    """
    record = _enrich_implementation_projection(store, record)
    _append_experience(store, record)
    outer = _outer_algorithm_experience(store, record)
    if outer is not None:
        _append_experience(store, outer)
    refresh_experience_indexes(store)
    refresh_compressed_summary(store)


def _finalize_experience(record: ExperienceRecord) -> ExperienceRecord:
    """Attach deterministic mechanism and semantic identities to a record.

    ``semantic_id`` is attached defensively so this module remains compatible
    with older ``ExperienceRecord`` schemas.  Persistence always includes it,
    even when the installed model predates that field.
    """
    if not record.mechanism_projection:
        if record.scope == "workflow":
            record.mechanism_projection = workflow_mechanism_projection(
                outcome=record.outcome,
                failure_stage=record.failure_stage,
                failure_class=record.failure_class,
                component_name=record.component_name,
            )
        else:
            record.mechanism_projection = {
                "schema_version": "1.0",
                "track": record.scope,
                "component_name": record.component_name,
            }
    record.mechanism_fingerprint = mechanism_fingerprint(record.mechanism_projection)
    semantic_id = sha256_text(canonical_json({
        "schema_version": "1.0",
        "scope": record.scope,
        "polarity": record.polarity,
        "outcome": record.outcome,
        "mechanism_fingerprint": record.mechanism_fingerprint,
    }))
    # Pydantic v2 intentionally ignores this compatibility attribute in
    # model_dump when the older model is installed; _append_experience adds it.
    object.__setattr__(record, "semantic_id", semantic_id)
    record.experience_id = record_hash(record, id_field="experience_id")
    return record


def _append_experience(store: LoopStore, record: ExperienceRecord) -> None:
    payload = record.model_dump(mode="json")
    payload["semantic_id"] = str(getattr(record, "semantic_id"))
    existing = next(
        (
            item for item in store.iter_experience()
            if item.get("experience_id") == record.experience_id
        ),
        None,
    )
    if existing is not None:
        # A pre-dual-track run can contain the same content-addressed record
        # without semantic_id. Treat that as the same immutable evidence.
        old = dict(existing)
        new = dict(payload)
        old.pop("semantic_id", None)
        new.pop("semantic_id", None)
        if canonical_json(old, exclude_fields={"created_at"}) == canonical_json(
            new, exclude_fields={"created_at"}
        ):
            return
    store.append_experience(payload)


def _enrich_implementation_projection(
    store: LoopStore,
    record: ExperienceRecord,
) -> ExperienceRecord:
    if record.scope != "inner_implementation" or not record.source_patch_plan_ids:
        return _finalize_experience(record)
    try:
        patch_plan = store.get_patch_plan(record.source_patch_plan_ids[-1])
    except (FileNotFoundError, OSError, TypeError, ValueError):
        return _finalize_experience(record)
    record.mechanism_projection = implementation_mechanism_projection(patch_plan)
    return _finalize_experience(record)


def _outer_algorithm_experience(
    store: LoopStore,
    implementation: ExperienceRecord,
) -> ExperienceRecord | None:
    metric = implementation.metric_summary or {}
    if (
        implementation.scope != "inner_implementation"
        or str(metric.get("stage") or "").lower() != "full"
        or not implementation.source_algorithm_plan_ids
    ):
        return None
    try:
        plan = store.get_algorithm_plan(implementation.source_algorithm_plan_ids[-1])
    except (FileNotFoundError, OSError, TypeError, ValueError):
        return None

    projection = algorithm_mechanism_projection(plan)
    target_contracts = list(plan.target_contracts)
    component = target_contracts[0] if target_contracts else implementation.component_name
    outer = ExperienceRecord(
        source_eval_ids=list(implementation.source_eval_ids),
        source_patchset_ids=list(implementation.source_patchset_ids),
        source_algorithm_plan_ids=[plan.algorithm_plan_id] if plan.algorithm_plan_id else [],
        source_patch_plan_ids=[],
        component_name=component,
        scope="outer_algorithm",
        polarity=implementation.polarity,
        outcome=implementation.outcome,
        lesson_kind="algorithm",
        severity=implementation.severity,
        metric_summary=dict(metric),
        patch_signature={
            "kind": "algorithm_plan",
            "algorithm_plan_id": plan.algorithm_plan_id,
            "algorithm_family": plan.algorithm_family,
            "target_contracts": target_contracts,
            "target_functions": list(plan.target_functions),
            "round_idx": plan.round_idx,
        },
        lesson=(
            f"Full evaluation outcome={implementation.outcome} for "
            f"{plan.algorithm_family}: {plan.proposed_algorithm_change}"
        ),
        positive_patterns=list(implementation.positive_patterns),
        negative_patterns=list(implementation.negative_patterns),
        applicability=sorted({*target_contracts, *plan.target_functions}),
        retrieval_tags=sorted({
            *implementation.retrieval_tags,
            "outer_algorithm",
            plan.algorithm_family,
        }),
        mechanism_projection=projection,
        mechanism_fingerprint=mechanism_fingerprint(projection),
        created_at=implementation.created_at,
    )
    return _finalize_experience(outer)


def _l2_promotion_candidate(item: dict[str, Any]) -> dict[str, Any] | None:
    metric = item.get("metric_summary") or {}
    if (
        item.get("outcome") not in {"accepted_speedup", "accepted_speedup_paired"}
        or str(metric.get("stage") or "").lower() != "full"
        or item.get("scope") not in {"outer_algorithm", "inner_implementation"}
    ):
        return None
    timing = metric.get("timing") or {}
    repeat_aggregate = (metric.get("profile") or {}).get("repeat_aggregate") or {}
    repeats = int(
        timing.get("eval_repeats")
        or timing.get("completed_repeats")
        or repeat_aggregate.get("count")
        or 1
    )
    confirmed = repeats >= 2
    candidate = _compact_items([item])[0]
    candidate.update({
        "promotion_confidence": "confirmed" if confirmed else "candidate",
        "promotion_confirmation": {
            "reason": "repeated_full_evaluation" if confirmed else "single_full_eval_only",
            "eval_repeats": repeats,
            "required_eval_repeats": 2,
        },
    })
    return candidate


def refresh_experience_indexes(store: LoopStore, *, max_records: int = 200) -> None:
    """Write deterministic indexes used for retrieval and auditing.

    Raw ``experience.jsonl`` remains the evidence source. The indexes are derived
    projections: component stats, workflow failure stats, and accepted speed
    lineage. They are safe to regenerate at any time.
    """
    items = _read_experience(store, max_records=max_records)
    by_component: dict[str, list[dict[str, Any]]] = {}
    workflow_items: list[dict[str, Any]] = []
    accepted: list[dict[str, Any]] = []
    profile_items: list[dict[str, Any]] = []
    negative_items: list[dict[str, Any]] = []
    for item in items:
        component = item.get("component_name") or "workflow"
        by_component.setdefault(component, []).append(item)
        if item.get("lesson_kind") == "workflow":
            workflow_items.append(item)
        if item.get("outcome") in POSITIVE_OUTCOMES:
            accepted.append(item)
        if _is_recent_negative(item):
            negative_items.append(item)
        profile_item = _profile_index_item(item)
        if profile_item:
            profile_items.append(profile_item)

    component_stats: dict[str, Any] = {}
    for component, component_items in sorted(by_component.items()):
        outcomes = Counter(str(item.get("outcome")) for item in component_items)
        component_stats[component] = {
            "count": len(component_items),
            "outcomes": dict(sorted(outcomes.items())),
            "accepted_speedups": outcomes.get("accepted_speedup", 0),
            "quality_regressions": outcomes.get("rejected_quality_tp_regression", 0) + outcomes.get("rejected_quality_fp_regression", 0),
            "patch_apply_failures": outcomes.get("patch_apply_failure", 0),
            "recent_experience_ids": [item.get("experience_id") for item in component_items[-6:] if item.get("experience_id")],
        }
    store.write_index("experience_component_stats", component_stats)
    store.write_index("experience_workflow_failures", _compact_items(workflow_items[-30:]))
    accepted_sorted = sorted(
        accepted,
        key=lambda item: (
            float((item.get("metric_summary") or {}).get("cost_s", 0.0)),
            str(item.get("created_at") or ""),
        ),
    )
    store.write_index("experience_accepted_speed_lineage", _compact_items(accepted_sorted[:50]))
    store.write_index("experience_profile_evidence", profile_items[-80:])
    store.write_index("experience_recent_negatives", _compact_negative_items(negative_items[-80:])[-30:])
    store.write_index(
        "l2_promotion_candidates",
        [
            candidate
            for item in items
            if (candidate := _l2_promotion_candidate(item)) is not None
        ][-80:],
    )


def refresh_compressed_summary(store: LoopStore, *, max_records: int = 120) -> None:
    """Regenerate model-facing experience summary from raw evidence."""
    items = _read_experience(store, max_records=max_records)
    if not items:
        return
    by_component: dict[str, list[dict[str, Any]]] = {}
    workflow_items: list[dict[str, Any]] = []
    profile_items: list[dict[str, Any]] = []
    negative_items: list[dict[str, Any]] = []
    for item in items:
        component = item.get("component_name") or "workflow"
        by_component.setdefault(component, []).append(item)
        if item.get("lesson_kind") == "workflow":
            workflow_items.append(item)
        if _is_recent_negative(item):
            negative_items.append(item)
        profile_item = _profile_index_item(item)
        if profile_item:
            profile_items.append(profile_item)

    out = [
        "# Compressed C-patch loop experience",
        "",
        "Use this as evidence, not as a substitute for the current context pack.",
        "Each lesson keeps eval/patchset IDs so a later audit can trace the raw record.",
        "",
        "## Retrieval policy",
        "- Prefer the current baseline/current_speed_best from the context pack for decisions.",
        "- Use recent_history for the last few attempts and this summary for durable patterns.",
        "- Treat workflow failures (apply/build/provenance) as diff/repair guidance, not algorithm evidence.",
        "- Reuse accepted speedup patterns only when TP/FP were preserved against the quality reference.",
        "",
    ]
    accepted = [item for item in items if item.get("outcome") == "accepted_speedup"]
    if accepted:
        out.append("## Accepted speed lineage")
        for item in sorted(accepted, key=_cost_key)[:8]:
            out.append(_format_item(item))
        out.append("")

    if profile_items:
        out.append("## Profile-derived algorithm evidence")
        for item in profile_items[-8:]:
            out.append(_format_profile_item(item))
        out.append("")

    quality_slow = [item for item in negative_items if item.get("outcome") == "rejected_not_faster"]
    if quality_slow:
        out.append("## Recent quality-preserving slow rejects")
        for item in _compact_negative_items(quality_slow[-8:]):
            out.append(_format_negative_item(item))
        out.append("")

    avoid_rules = _negative_avoid_rules(negative_items)
    if avoid_rules:
        out.append("## Repeated negative patterns / avoid rules")
        for rule in avoid_rules[:10]:
            out.append(f"- {rule}")
        out.append("")

    workflow_failures = [item for item in negative_items if item.get("lesson_kind") == "workflow"]
    if workflow_failures:
        out.append("## Workflow failures to repair at plan/proposal stage")
        for item in _compact_negative_items(workflow_failures[-8:]):
            out.append(_format_negative_item(item))
        out.append("")

    for component, component_items in sorted(by_component.items()):
        if component == "workflow":
            continue
        outcomes = Counter(str(item.get("outcome")) for item in component_items)
        out.append(f"## Component: {component}")
        out.append("- outcome_counts: " + json.dumps(dict(sorted(outcomes.items())), ensure_ascii=False, sort_keys=True))
        positives = [item for item in component_items if item.get("outcome") in POSITIVE_OUTCOMES]
        negatives = [item for item in component_items if item.get("outcome") not in POSITIVE_OUTCOMES]
        if positives:
            out.append("- useful patterns:")
            for item in positives[-4:]:
                out.append("  " + _format_item(item))
        if negatives:
            out.append("- avoid/repair patterns:")
            for item in negatives[-6:]:
                out.append("  " + _format_item(item))
        out.append("")

    if workflow_items:
        out.append("## Workflow / repair lessons")
        for item in workflow_items[-8:]:
            out.append(_format_item(item))
        out.append("")
    store.experience_dir.mkdir(parents=True, exist_ok=True)
    (store.experience_dir / "compressed_summary.md").write_text("\n".join(out), encoding="utf-8")


def _read_experience(store: LoopStore, *, max_records: int) -> list[dict[str, Any]]:
    raw = store.experience_dir / "experience.jsonl"
    if not raw.exists():
        return []
    out: list[dict[str, Any]] = []
    for line in raw.read_text(encoding="utf-8", errors="replace").splitlines()[-max_records:]:
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(item, dict):
            out.append(item)
    return out


def _outcome(evaluation: EvaluationRecord) -> str:
    if _looks_like_patch_apply_failure(evaluation):
        return "patch_apply_failure"
    if evaluation.stage == "static_verify" and evaluation.status != "ok":
        return "scope_violation"
    if evaluation.decision == "rejected_not_runtime_effective":
        return "runtime_provenance_failure"
    if evaluation.status == "compile_error":
        return "compile_failure"
    if evaluation.status == "run_error":
        return "runtime_failure"
    if evaluation.decision in {"accepted_speedup", "accepted_speedup_paired", "accepted_smoke_speedup"}:
        return "accepted_speedup"
    if evaluation.decision == "accepted_baseline":
        return "stock_baseline"
    if evaluation.decision in {"rejected_not_faster", "rejected_not_faster_paired"}:
        return "rejected_not_faster"
    if evaluation.decision == "rejected_quality_tp_regression":
        return "rejected_quality_tp_regression"
    if evaluation.decision == "rejected_quality_fp_regression":
        return "rejected_quality_fp_regression"
    if evaluation.decision.startswith("accepted"):
        return "accepted_quality_preserving_candidate"
    if evaluation.decision.startswith("rejected"):
        return "metric_regression"
    return "neutral"


def _metric_summary(evaluation: EvaluationRecord, result: dict[str, Any]) -> dict[str, Any]:
    return {
        "stage": evaluation.stage,
        "status": evaluation.status,
        "tp": result.get("tp", 0),
        "fp": result.get("fp", 0),
        "tp_fp_ratio": result.get("tp_fp_ratio", 0.0),
        "cost_s": result.get("cost_s", 0.0),
        "decision": evaluation.decision,
        "metric_delta": evaluation.metric_delta,
        "timing": (result.get("meta") or {}).get("timing") or {},
        "profile": _compact_profile((result.get("meta") or {}).get("profile") or {}),
    }


def _compact_profile(profile: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(profile, dict) or not profile:
        return {}
    return {
        "schema_version": profile.get("schema_version"),
        "dominant_command": profile.get("dominant_command"),
        "command_wall_s": profile.get("command_wall_s") or {},
        "command_wall_fraction": profile.get("command_wall_fraction") or {},
        "query_cost_stats": profile.get("query_cost_stats") or {},
        "top_slowest_queries": (profile.get("top_slowest_queries") or [])[:5],
        "external_profiler": _compact_external_profiler(profile.get("external_profiler") or {}),
        "symbol_hotspots": (profile.get("symbol_hotspots") or [])[:10],
        "command_symbol_hotspots": _compact_command_symbol_hotspots(profile.get("command_symbol_hotspots") or {}),
        # Per-stage evidence for the outer planner: which pipeline stage the time
        # goes to, keyed by the component contract the agent may actually edit.
        # Carried straight through from EvalResult meta -- no separate store.
        "stage_timing": _compact_stage_timing(profile.get("stage_timing") or {}),
        "repeat_strategy": profile.get("repeat_strategy"),
        "selected_repeat_idx": profile.get("selected_repeat_idx"),
        "repeat_aggregate": profile.get("repeat_aggregate") or {},
    }


def _compact_stage_timing(data: dict[str, Any]) -> dict[str, Any]:
    """Carry the stage-attribution picture through to the outer planner.

    Two shapes reach this point and both are reduced to the same headline facts so
    the digest has one shape to read:

    * the session rollup built by ``run_hmmer_cli._aggregate_profiles``
      (``stage_time_s`` / ``stage_time_fraction`` / ``measured_stage_total_s``);
    * the per-query report from ``build_stage_timing`` (``contract_stages``).

    Absent the instrumented profiling build, no stage times exist and only the
    jackhmmer per-iteration elapsed figures are carried -- that is a real signal
    (where the wall clock goes), not a fabricated stage split.
    """
    if not isinstance(data, dict) or not data:
        return {}

    compact: dict[str, Any] = {
        "time_source": data.get("time_source"),
        "stage_time_available": bool(data.get("stage_time_available")),
    }

    stage_time_s = data.get("stage_time_s")
    if not isinstance(stage_time_s, dict) or not stage_time_s:
        # Fall back to the per-query contract_stages shape.
        stage_time_s = {
            name: entry["time_s"]
            for name, entry in (data.get("contract_stages") or {}).items()
            if isinstance(entry, dict) and entry.get("time_s") is not None
        }

    if stage_time_s:
        stage_time_s = dict(sorted(stage_time_s.items()))
        total = round(sum(float(value) for value in stage_time_s.values()), 4)
        compact["stage_time_s"] = stage_time_s
        compact["measured_stage_total_s"] = total
        fraction = data.get("stage_time_fraction")
        compact["stage_time_fraction"] = (
            fraction
            if isinstance(fraction, dict) and fraction
            else (
                {name: round(float(value) / total, 6) for name, value in stage_time_s.items()}
                if total
                else {}
            )
        )
    elif data.get("stage_time_unavailable_reason"):
        compact["stage_time_unavailable_reason"] = data["stage_time_unavailable_reason"]

    summary = data.get("jackhmmer_summary")
    if isinstance(summary, dict):
        compact["jackhmmer"] = {
            "iterations_reported": summary.get("iterations_reported"),
            "elapsed_total_s": summary.get("elapsed_total_s"),
            "slowest_iteration_round": summary.get("slowest_iteration_round"),
            "slowest_iteration_elapsed_s": summary.get("slowest_iteration_elapsed_s"),
        }
    elif isinstance(data.get("jackhmmer_elapsed_stats"), dict):
        compact["jackhmmer_elapsed_stats"] = data["jackhmmer_elapsed_stats"]

    if data.get("slowest_iteration_query_id"):
        compact["slowest_iteration_query_id"] = data["slowest_iteration_query_id"]
    return compact


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


def _patch_signature(patchset: PatchsetRecord) -> dict[str, Any]:
    patch = patchset.patches[-1] if patchset.patches else None
    if patch is None:
        return {"kind": "stock", "round_idx": patchset.round_idx}
    return {
        "component_name": patch.component_name,
        "target_file": patch.c_target_file,
        "target_functions": list(patch.c_target_functions),
        "diff_sha256": patch.diff_sha256,
        "patch_id": patch.patch_id,
        "patch_plan_id": patchset.patch_plan_id,
        "round_idx": patchset.round_idx,
    }


def _lesson_kind(outcome: str) -> str:
    if outcome in WORKFLOW_FAILURE_OUTCOMES:
        return "workflow"
    return "algorithm"


def _severity(outcome: str) -> str:
    if outcome in {"accepted_speedup", "stock_baseline"}:
        return "positive"
    if outcome in {"rejected_quality_tp_regression", "rejected_quality_fp_regression", "patch_apply_failure", "compile_failure"}:
        return "high"
    if outcome.startswith("rejected") or outcome in {"runtime_failure", "runtime_provenance_failure", "scope_violation"}:
        return "medium"
    return "info"


def _polarity(outcome: str) -> str:
    if outcome in POSITIVE_OUTCOMES:
        return "positive"
    if outcome == "neutral":
        return "neutral"
    return "negative"


def _retrieval_tags(outcome: str, evaluation: EvaluationRecord, component: str | None) -> list[str]:
    tags = [outcome, evaluation.stage, evaluation.status]
    if component:
        tags.append(component)
    if outcome in WORKFLOW_FAILURE_OUTCOMES:
        tags.append("repair")
    if outcome == "accepted_speedup":
        tags.extend(["speedup", "quality_preserving"])
    if outcome in {"rejected_quality_tp_regression", "rejected_quality_fp_regression"}:
        tags.append("quality_gate")
    return sorted({tag for tag in tags if tag})


def _looks_like_patch_apply_failure(evaluation: EvaluationRecord) -> bool:
    text = "\n".join(_error_texts(evaluation)).lower()
    return any(
        marker in text
        for marker in (
            "git apply --check",
            "git apply",
            "patch does not apply",
            "patch failed",
            "corrupt patch",
            "unrecognized input",
            "exact edit",
            "old block matched",
        )
    )


def _error_texts(evaluation: EvaluationRecord) -> list[str]:
    result = evaluation.eval_result or {}
    out: list[str] = []
    for value in (evaluation.errors, evaluation.violations, result.get("errors")):
        if isinstance(value, list):
            out.extend(str(item) for item in value)
        elif value:
            out.append(str(value))
    meta = result.get("meta") or {}
    for repeat in meta.get("repeat_results") or []:
        if isinstance(repeat, dict):
            repeat_errors = repeat.get("errors") or []
            if isinstance(repeat_errors, list):
                out.extend(str(item) for item in repeat_errors)
            elif repeat_errors:
                out.append(str(repeat_errors))
    return out


def _default_lesson(component: str | None, evaluation: EvaluationRecord, result: dict[str, Any]) -> str:
    name = component or "workflow"
    if _looks_like_patch_apply_failure(evaluation):
        errors = _error_texts(evaluation)
        detail = errors[0] if errors else "git apply failed"
        return f"{name} diff was not applyable during validation/build; prefer exact old/new edit blocks copied from the source excerpt, or regenerate the unified diff with exact source context: {detail}"
    if evaluation.violations:
        return f"{name} patch violated static contract: {evaluation.violations[0]}"
    if evaluation.decision == "rejected_not_runtime_effective":
        return (
            f"{name} patch passed runnability but failed runtime provenance: rebuilt "
            "jackhmmer/hmmsearch executable hashes did not change."
        )
    if evaluation.errors:
        return f"{name} evaluation failed at {evaluation.stage}: {evaluation.errors[0]}"
    errors = result.get("errors") or []
    if errors:
        return f"{name} evaluation reported {result.get('status')}: {errors[0]}"
    if evaluation.decision in {"rejected_not_faster", "rejected_not_faster_paired"}:
        paired = (evaluation.metric_delta or {}).get("paired_confirmation") or {}
        speedup = (evaluation.metric_delta or {}).get("speedup_fraction")
        cost_delta = (evaluation.metric_delta or {}).get("cost_s_delta")
        paired_note = ""
        if paired:
            paired_note = f" paired_current_best_cost_s={paired.get('paired_cost_s')} paired_speedup_fraction={paired.get('speedup_fraction')}."
        return (
            f"{name} preserved TP/FP quality at {evaluation.stage} but was not faster enough: "
            f"cost_s={result.get('cost_s', 0.0)}, tp={result.get('tp', 0)}, fp={result.get('fp', 0)}, "
            f"speedup_fraction={speedup}, cost_s_delta={cost_delta}. "
            "Treat this as a negative speed lesson: do not repeat the same component/function patch pattern unless new fine-grained profile evidence and a different mechanism justify it."
            f"{paired_note}"
        )
    if evaluation.decision in {"rejected_quality_tp_regression", "rejected_quality_fp_regression"}:
        return (
            f"{name} failed the hard TP/FP quality gate at {evaluation.stage}: "
            f"tp={result.get('tp', 0)}, fp={result.get('fp', 0)}, cost_s={result.get('cost_s', 0.0)}."
        )
    return (
        f"{name} {evaluation.stage} result status={evaluation.status}, "
        f"cost_s={result.get('cost_s', 0.0)}, tp={result.get('tp', 0)}, fp={result.get('fp', 0)}, "
        f"ratio={result.get('tp_fp_ratio', 0.0)}."
    )


def _positive_patterns(evaluation: EvaluationRecord, result: dict[str, Any]) -> list[str]:
    if evaluation.decision in {"accepted_speedup", "accepted_speedup_paired", "accepted_smoke_speedup"}:
        timing = (result.get("meta") or {}).get("timing") or {}
        speedup = (evaluation.metric_delta or {}).get("speedup_fraction")
        detail = f"quality-preserving speedup cost_s={result.get('cost_s', 0.0)}"
        if speedup is not None:
            detail += f" speedup_fraction={speedup}"
        if timing.get("eval_repeats"):
            detail += f" repeats={timing.get('eval_repeats')}"
        return [detail]
    if evaluation.status == "ok" and not evaluation.violations:
        return [f"{evaluation.stage} completed with cost_s={result.get('cost_s', 0.0)} ratio={result.get('tp_fp_ratio', 0.0)}"]
    return []


def _negative_patterns(evaluation: EvaluationRecord, result: dict[str, Any]) -> list[str]:
    out: list[str] = []
    if _looks_like_patch_apply_failure(evaluation):
        out.append("git apply / patch context failure")
    if evaluation.violations:
        out.append("static contract violation")
    if evaluation.decision == "rejected_not_runtime_effective":
        out.append("runtime provenance failed")
    if evaluation.decision in {"rejected_not_faster", "rejected_not_faster_paired"}:
        out.append("quality preserved but speedup threshold not met")
        if evaluation.decision == "rejected_not_faster_paired":
            out.append("paired current-best confirmation rejected noisy speedup")
    if evaluation.decision == "rejected_quality_tp_regression":
        out.append("TP quality regression")
    if evaluation.decision == "rejected_quality_fp_regression":
        out.append("FP quality regression")
    if evaluation.status in {"compile_error", "run_error"}:
        out.append(evaluation.status)
    if result.get("status") in {"compile_error", "run_error", "skipped"}:
        out.append(str(result.get("status")))
    return out


def _profile_index_item(item: dict[str, Any]) -> dict[str, Any] | None:
    metric = item.get("metric_summary") or {}
    profile = metric.get("profile") or {}
    if not profile:
        return None
    return {
        "experience_id": item.get("experience_id"),
        "semantic_id": item.get("semantic_id"),
        "source_eval_ids": item.get("source_eval_ids") or [],
        "source_patchset_ids": item.get("source_patchset_ids") or [],
        "source_algorithm_plan_ids": item.get("source_algorithm_plan_ids") or [],
        "source_patch_plan_ids": item.get("source_patch_plan_ids") or [],
        "component_name": item.get("component_name"),
        "scope": item.get("scope"),
        "polarity": item.get("polarity"),
        "outcome": item.get("outcome"),
        "lesson_kind": item.get("lesson_kind"),
        "mechanism_projection": item.get("mechanism_projection") or {},
        "mechanism_fingerprint": item.get("mechanism_fingerprint"),
        "stage": metric.get("stage"),
        "decision": metric.get("decision"),
        "cost_s": metric.get("cost_s"),
        "tp": metric.get("tp"),
        "fp": metric.get("fp"),
        "profile": profile,
        "patch_signature": item.get("patch_signature") or {},
    }


def _stage_focus(stage_timing: dict[str, Any]) -> str:
    """Render "which pipeline stage dominates" for the planner.

    Returns ``""`` when no stage timing is available, so the digest never implies a
    stage split that was not measured. The instrumented profiling build is what
    turns the second and third branches from unreachable into reachable.
    """
    if not isinstance(stage_timing, dict) or not stage_timing:
        return ""
    fraction = stage_timing.get("stage_time_fraction")
    if isinstance(fraction, dict) and fraction:
        ordered = sorted(fraction.items(), key=lambda kv: float(kv[1]), reverse=True)
        return ",".join(f"{name}:{float(share):.3f}" for name, share in ordered[:4])
    stage_time_s = stage_timing.get("stage_time_s")
    if isinstance(stage_time_s, dict) and stage_time_s:
        ordered = sorted(stage_time_s.items(), key=lambda kv: float(kv[1]), reverse=True)
        return ",".join(f"{name}:{float(sec):.4f}s" for name, sec in ordered[:4])
    stats = stage_timing.get("jackhmmer_elapsed_stats")
    if isinstance(stats, dict) and stats.get("mean") is not None:
        return f"iteration_elapsed_mean:{float(stats['mean']):.4f}s(no stage split)"
    return ""


def _format_profile_item(item: dict[str, Any]) -> str:
    profile = item.get("profile") or {}
    evals = ",".join(item.get("source_eval_ids") or []) or "no-eval-id"
    top_queries = ",".join(str(q.get("query_id") or q.get("query_idx")) for q in (profile.get("top_slowest_queries") or [])[:3])
    line = (
        f"- eval={evals} stage={item.get('stage')} decision={item.get('decision')} "
        f"component={item.get('component_name') or 'stock'} dominant_command={profile.get('dominant_command')} "
        f"cost_s={item.get('cost_s')} tp={item.get('tp')} fp={item.get('fp')} "
        f"command_wall_fraction={json.dumps(profile.get('command_wall_fraction') or {}, ensure_ascii=False, sort_keys=True)} "
        f"top_queries={top_queries}"
    )
    focus = _stage_focus(profile.get("stage_timing") or {})
    if focus:
        line += f" stage_focus={focus}"
    return line


def _compact_items(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "experience_id": item.get("experience_id"),
            "semantic_id": item.get("semantic_id"),
            "source_eval_ids": item.get("source_eval_ids") or [],
            "source_patchset_ids": item.get("source_patchset_ids") or [],
            "source_algorithm_plan_ids": item.get("source_algorithm_plan_ids") or [],
            "source_patch_plan_ids": item.get("source_patch_plan_ids") or [],
            "component_name": item.get("component_name"),
            "scope": item.get("scope"),
            "polarity": item.get("polarity"),
            "failure_stage": item.get("failure_stage"),
            "failure_class": item.get("failure_class"),
            "outcome": item.get("outcome"),
            "lesson_kind": item.get("lesson_kind"),
            "metric_summary": item.get("metric_summary") or {},
            "patch_signature": item.get("patch_signature") or {},
            "lesson": item.get("lesson"),
            "retrieval_tags": item.get("retrieval_tags") or [],
            "mechanism_projection": item.get("mechanism_projection") or {},
            "mechanism_fingerprint": item.get("mechanism_fingerprint"),
        }
        for item in items
    ]


def _is_recent_negative(item: dict[str, Any]) -> bool:
    outcome = str(item.get("outcome") or "")
    if outcome in POSITIVE_OUTCOMES:
        return False
    if outcome == "neutral":
        return False
    return bool(outcome)


def _compact_negative_items(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    compact: list[dict[str, Any]] = []
    for item in items:
        metric = item.get("metric_summary") or {}
        patch = item.get("patch_signature") or {}
        delta = metric.get("metric_delta") or {}
        profile = metric.get("profile") or {}
        compact.append({
            "experience_id": item.get("experience_id"),
            "semantic_id": item.get("semantic_id"),
            "outcome": item.get("outcome"),
            "lesson_kind": item.get("lesson_kind"),
            "severity": item.get("severity"),
            "component_name": item.get("component_name"),
            "scope": item.get("scope"),
            "polarity": item.get("polarity"),
            "failure_stage": item.get("failure_stage"),
            "failure_class": item.get("failure_class"),
            "target_functions": patch.get("target_functions") or [],
            "patch_plan_id": patch.get("patch_plan_id"),
            "round_idx": patch.get("round_idx") or metric.get("round_idx"),
            "decision": metric.get("decision"),
            "tp": metric.get("tp"),
            "fp": metric.get("fp"),
            "cost_s": metric.get("cost_s"),
            "speedup_fraction": delta.get("speedup_fraction"),
            "cost_s_delta": delta.get("cost_s_delta"),
            "dominant_command": profile.get("dominant_command"),
            "symbol_hotspots": (profile.get("symbol_hotspots") or [])[:5],
            "negative_patterns": item.get("negative_patterns") or [],
            "lesson": item.get("lesson"),
            "source_eval_ids": item.get("source_eval_ids") or [],
            "source_patchset_ids": item.get("source_patchset_ids") or [],
            "source_algorithm_plan_ids": item.get("source_algorithm_plan_ids") or [],
            "source_patch_plan_ids": item.get("source_patch_plan_ids") or [],
            "mechanism_projection": item.get("mechanism_projection") or {},
            "mechanism_fingerprint": item.get("mechanism_fingerprint"),
        })
    return compact


def _negative_avoid_rules(items: list[dict[str, Any]]) -> list[str]:
    counts: Counter[tuple[str, tuple[str, ...], str]] = Counter()
    for item in items:
        patch = item.get("patch_signature") or {}
        component = str(item.get("component_name") or "workflow")
        functions = tuple(str(fn) for fn in (patch.get("target_functions") or []))
        outcome = str(item.get("outcome") or "")
        counts[(component, functions, outcome)] += 1
    rules: list[str] = []
    for (component, functions, outcome), count in counts.most_common(12):
        if count < 1:
            continue
        target = ",".join(functions) if functions else "workflow"
        if outcome == "rejected_not_faster":
            rules.append(f"{component}:{target} has recent quality-preserving slow reject count={count}; require a different mechanism and new profile evidence before repeating.")
        elif outcome.endswith("proposal_error") or outcome == "plan_validation_failure":
            rules.append(f"{component}:{target} has recent {outcome} count={count}; repair plan/schema/profile citation before spending full eval.")
        else:
            rules.append(f"{component}:{target} has recent negative outcome {outcome} count={count}; address this failure mode before repeating.")
    return rules


def _format_negative_item(item: dict[str, Any]) -> str:
    funcs = ",".join(item.get("target_functions") or []) or "none"
    evals = ",".join(item.get("source_eval_ids") or []) or "no-eval-id"
    return (
        f"- {item.get('outcome')} component={item.get('component_name') or 'workflow'} functions={funcs} "
        f"decision={item.get('decision')} tp={item.get('tp')} fp={item.get('fp')} cost_s={item.get('cost_s')} "
        f"speedup_fraction={item.get('speedup_fraction')} dominant_command={item.get('dominant_command')} "
        f"lesson={item.get('lesson')} eval={evals}"
    )


def _format_item(item: dict[str, Any]) -> str:
    metric = item.get("metric_summary") or {}
    patch = item.get("patch_signature") or {}
    evals = ",".join(item.get("source_eval_ids") or []) or "no-eval-id"
    patchsets = ",".join(item.get("source_patchset_ids") or []) or "no-patchset-id"
    tags = ",".join(item.get("retrieval_tags") or [])
    return (
        f"- {item.get('outcome')} [{item.get('lesson_kind', 'algorithm')}]: {item.get('lesson')} "
        f"(component={item.get('component_name') or 'workflow'}, stage={metric.get('stage')}, "
        f"tp={metric.get('tp')}, fp={metric.get('fp')}, cost_s={metric.get('cost_s')}, "
        f"diff={patch.get('diff_sha256')}, eval={evals}, patchset={patchsets}, tags={tags})"
    )


def _cost_key(item: dict[str, Any]) -> tuple[float, str]:
    metric = item.get("metric_summary") or {}
    return (float(metric.get("cost_s", 0.0)), str(item.get("created_at") or ""))
