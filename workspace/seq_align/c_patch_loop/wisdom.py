"""Durable, content-addressed wisdom for hierarchical context compression."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .experience import (
    algorithm_mechanism_projection,
    implementation_mechanism_projection,
    mechanism_fingerprint,
)
from .models import DurableWisdomRecord
from .store import LoopStore, canonical_json, record_hash, utc_now, write_json


_SEED = {
    "source": "verified_seed",
    "confidence": "confirmed",
    "wisdom_kind": "implementation_wisdom",
    "component_name": "sequence_io",
    "target_functions": ["addbuf"],
    "target_symbols": ["addbuf"],
    "lesson": "Cache repeatedly loaded scalar state in the addbuf hot loop while preserving every residue and line-state transition.",
    "source_eval_ids": [
        "cccb255b9c0d2ec126e3b913ecf22531de2270bff4ede48dda5c8bef5e420a0f",
        "87f40e4cf3282cf0fb4b88afbcf8118672dcad02035ae201ed6590ab985bfe67",
    ],
    "source_patchset_ids": [
        "3f0141f583e29f8531c2ba20eb6bad291f587b5d1edd8afcab7f7f8cbd2f22c2"
    ],
    "metric_evidence": {
        "median_speedup_fraction": 0.22439045235190677,
        "stock": {"tp": 14, "fp": 3},
        "patch": {"tp": 14, "fp": 3},
    },
    "retrieval_tags": ["sequence_io", "addbuf", "parser_hotspot"],
}


def load_durable_wisdom(wisdom_dir: str | Path) -> list[DurableWisdomRecord]:
    root = Path(wisdom_dir)
    if not root.exists():
        return []
    records: list[DurableWisdomRecord] = []
    seen: set[str] = set()
    for path in sorted(root.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        record = DurableWisdomRecord.model_validate(data)
        if record.wisdom_id and path.stem != record.wisdom_id:
            raise ValueError(f"embedded wisdom ID mismatch: {path}")
        records.append(record)
        seen.add(record.wisdom_id or canonical_json(record))
    # Early HCC runs used one append-only JSONL file.  Keep those immutable
    # artifacts readable while new records use one content-addressed JSON file.
    legacy_path = root / "wisdom.jsonl"
    if legacy_path.is_file():
        for line_no, line in enumerate(legacy_path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            try:
                record = DurableWisdomRecord.model_validate(json.loads(line))
            except (json.JSONDecodeError, TypeError, ValueError) as exc:
                raise ValueError(f"invalid legacy wisdom record at {legacy_path}:{line_no}") from exc
            key = record.wisdom_id or canonical_json(record)
            if key not in seen:
                records.append(record)
                seen.add(key)
    return records


def upsert_durable_wisdom(record: DurableWisdomRecord, wisdom_dir: str | Path) -> DurableWisdomRecord:
    if not record.wisdom_id:
        record.wisdom_id = record_hash(record, id_field="wisdom_id")
    write_json(Path(wisdom_dir) / f"{record.wisdom_id}.json", record)
    _refresh_wisdom_indexes(wisdom_dir)
    return record


def import_seed_wisdom(wisdom_dir: str | Path) -> list[DurableWisdomRecord]:
    record = DurableWisdomRecord(**_SEED, created_at="2026-08-01T00:00:00Z")
    record.wisdom_id = record_hash(record, id_field="wisdom_id")
    return [upsert_durable_wisdom(record, wisdom_dir)]


def retrieve_wisdom_for_context(
    profile_evidence: dict[str, Any],
    recent_history: list[dict[str, Any]],
    *,
    wisdom_dir: str | Path,
    limit: int = 12,
) -> list[DurableWisdomRecord]:
    """Rank wisdom deterministically by measured component/function overlap."""
    terms: set[str] = set()
    for key in ("contract_hotspots", "function_hotspots", "fine_grained_hotspots", "recommended_contracts"):
        for item in profile_evidence.get(key) or []:
            for field in ("component_name", "function", "symbol"):
                if item.get(field):
                    terms.add(str(item[field]).lower())
            for field in ("functions", "target_functions", "symbols"):
                terms.update(str(value).lower() for value in item.get(field) or [])
    terms.update(
        str(item.get("component_name")).lower()
        for item in recent_history
        if item.get("component_name")
    )

    def rank(record: DurableWisdomRecord) -> tuple[int, int, str]:
        data = record.model_dump(mode="json")
        haystack = canonical_json(data).lower()
        overlap = sum(1 for term in terms if term in haystack)
        confidence = {"confirmed": 2, "candidate": 1, "deprecated": 0}.get(record.confidence, 0)
        return (-overlap, -confidence, record.wisdom_id or "")

    records = [record for record in load_durable_wisdom(wisdom_dir) if record.confidence != "deprecated"]
    return sorted(records, key=rank)[:limit]


def promote_wisdom_from_run(
    store: LoopStore,
    *,
    wisdom_dir: str | Path,
    require_full: bool = True,
) -> list[DurableWisdomRecord]:
    """Promote only quality-preserving accepted speedups into durable memory."""
    patchsets = {item.get("patchset_id"): item for item in store.iter_patchsets()}
    patch_plans = {item.get("patch_plan_id"): item for item in store.iter_patch_plans()}
    algorithm_plans = {item.get("algorithm_plan_id"): item for item in store.iter_algorithm_plans()}
    promoted: list[DurableWisdomRecord] = []
    for evaluation in store.iter_evaluations():
        if require_full and evaluation.get("stage") != "full":
            continue
        if evaluation.get("decision") not in {"accepted_speedup", "accepted_speedup_paired"}:
            continue
        delta = evaluation.get("metric_delta") or {}
        if (delta.get("tp_delta") is not None and delta.get("tp_delta") < 0) or (
            delta.get("fp_delta") is not None and delta.get("fp_delta") > 0
        ):
            continue
        patchset = patchsets.get(evaluation.get("patchset_id")) or {}
        patches = patchset.get("patches") or []
        if not patches:
            continue
        patch = patches[-1]
        patch_plan_id = patchset.get("patch_plan_id")
        patch_plan = patch_plans.get(patch_plan_id) or {}
        algorithm_plan_id = patchset.get("algorithm_plan_id") or patch_plan.get("algorithm_plan_id")
        algorithm_plan = algorithm_plans.get(algorithm_plan_id) or {}
        confidence, confirmation = _promotion_confidence(evaluation.get("eval_result") or {})
        evidence = {
            "evaluation": evaluation.get("eval_result") or {},
            "metric_delta": delta,
            "promotion_confirmation": confirmation,
        }
        implementation_projection = implementation_mechanism_projection(patch_plan or patch)
        implementation = DurableWisdomRecord(
            wisdom_kind="implementation_wisdom",
            source="accepted_full_evaluation",
            confidence=confidence,
            component_name=patch.get("component_name"),
            target_functions=patch.get("c_target_functions") or [],
            target_symbols=patch_plan.get("target_symbols") or [],
            lesson=patch.get("rationale") or patch_plan.get("intended_code_change") or "Accepted quality-preserving implementation speedup.",
            source_eval_ids=[evaluation.get("eval_id")] if evaluation.get("eval_id") else [],
            source_patchset_ids=[patchset.get("patchset_id")] if patchset.get("patchset_id") else [],
            source_patch_plan_ids=[patch_plan_id] if patch_plan_id else [],
            metric_evidence=evidence,
            diff_sha256=patch.get("diff_sha256"),
            retrieval_tags=[value for value in [patch.get("component_name"), *(patch.get("c_target_functions") or [])] if value],
            mechanism_projection=implementation_projection,
            mechanism_fingerprint=mechanism_fingerprint(implementation_projection),
            created_at=utc_now(),
        )
        promoted.append(upsert_durable_wisdom(implementation, wisdom_dir))
        if algorithm_plan:
            algorithm_projection = algorithm_mechanism_projection(algorithm_plan)
            algorithm = DurableWisdomRecord(
                wisdom_kind="algorithm_wisdom",
                source="accepted_full_evaluation",
                confidence=confidence,
                algorithm_family=algorithm_plan.get("algorithm_family"),
                mechanism=algorithm_plan.get("speed_mechanism") or "",
                component_name=(algorithm_plan.get("target_contracts") or [None])[0],
                target_functions=algorithm_plan.get("target_functions") or [],
                lesson=algorithm_plan.get("proposed_algorithm_change") or "Accepted quality-preserving algorithm speedup.",
                source_eval_ids=[evaluation.get("eval_id")] if evaluation.get("eval_id") else [],
                source_patchset_ids=[patchset.get("patchset_id")] if patchset.get("patchset_id") else [],
                source_algorithm_plan_ids=[algorithm_plan_id] if algorithm_plan_id else [],
                metric_evidence=evidence,
                retrieval_tags=[value for value in [algorithm_plan.get("algorithm_family"), *(algorithm_plan.get("target_contracts") or [])] if value],
                mechanism_projection=algorithm_projection,
                mechanism_fingerprint=mechanism_fingerprint(algorithm_projection),
                created_at=utc_now(),
            )
            promoted.append(upsert_durable_wisdom(algorithm, wisdom_dir))
    return promoted


def _promotion_confidence(eval_result: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """Require repeated Full timing evidence before durable confirmation."""
    meta = eval_result.get("meta") or {}
    timing = meta.get("timing") or {}
    repeat_aggregate = (meta.get("profile") or {}).get("repeat_aggregate") or {}
    raw_repeats = (
        timing.get("eval_repeats")
        or timing.get("completed_repeats")
        or repeat_aggregate.get("count")
        or 1
    )
    try:
        repeats = max(1, int(raw_repeats))
    except (TypeError, ValueError):
        repeats = 1
    confirmed = repeats >= 2
    return (
        "confirmed" if confirmed else "candidate",
        {
            "reason": "repeated_full_evaluation" if confirmed else "single_full_eval_only",
            "eval_repeats": repeats,
            "required_eval_repeats": 2,
        },
    )


def _refresh_wisdom_indexes(wisdom_dir: str | Path) -> None:
    """Regenerate mutable lookup projections from immutable wisdom records."""
    root = Path(wisdom_dir)
    by_component: dict[str, list[dict[str, Any]]] = {}
    for record in load_durable_wisdom(root):
        component = record.component_name or "unscoped"
        by_component.setdefault(component, []).append(record.model_dump(mode="json"))
    for records in by_component.values():
        records.sort(key=lambda item: (str(item.get("wisdom_kind")), str(item.get("wisdom_id"))))
    path = root / "indexes" / "by_component.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(by_component, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)
