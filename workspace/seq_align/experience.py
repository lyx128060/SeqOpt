"""Auditable optimization memory with bounded model-facing retrieval.

Raw trajectories, patches, Slurm requests and evaluation shards remain the
source of truth. This module maintains append-only candidate events and
regenerable summaries so a fresh outer-loop Agent can use prior evidence
without loading the entire chat history.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .eval_feedback import compact_eval
from .eval_schema import EvalResult
from .inner_budget import candidate_signature
from .optimization_objective import objective_metadata
from .pending_mods import normalize_run_dir


MAX_CONTEXT_TOKENS = 256_000
DEFAULT_HISTORY_TOKENS = 64_000
LEDGER_FILENAME = "candidate_ledger.jsonl"
DIGEST_FILENAME = "experience_digest.json"


def estimate_tokens(text: str) -> int:
    """Conservative dependency-free estimate for mixed JSON/English/CJK text."""
    ascii_chars = sum(1 for char in text if ord(char) < 128)
    return max(1, (ascii_chars + 3) // 4 + (len(text) - ascii_chars))


def record_candidate_proposal(
    run_dir: str | Path,
    *,
    round_idx: int,
    mods: list[dict[str, Any]],
    component_name: str,
    hypothesis: str | None,
    expected_effects: dict[str, Any] | None,
    evidence: list[str] | None,
    pending_path: str | Path,
) -> dict[str, Any]:
    event = {
        "event_type": "proposal",
        "optimization_objective": objective_metadata(),
        "round_idx": round_idx,
        "candidate_signature": candidate_signature(mods),
        "component_name": component_name,
        "hypothesis": (hypothesis or "").strip(),
        "expected_effects": expected_effects or {},
        "evidence": evidence or [],
        "mods": mods,
        "artifacts": {"pending_mods": str(pending_path)},
    }
    return _append_event(run_dir, event)


def record_evaluation(
    run_dir: str | Path,
    *,
    round_idx: int,
    mods: list[dict[str, Any]],
    result: EvalResult,
    result_path: str | Path | None = None,
) -> dict[str, Any]:
    summary = compact_eval(result)
    event = {
        "event_type": "evaluation",
        "optimization_objective": objective_metadata(),
        "round_idx": round_idx,
        "candidate_signature": candidate_signature(mods),
        "component_name": _changed_component(mods, normalize_run_dir(run_dir)),
        "stage": result.stage,
        "result": summary,
        "artifacts": {
            "result": str(result_path) if result_path is not None else None,
            "slurm_job_dir": result.meta.get("job_dir"),
        },
    }
    stored = _append_event(run_dir, event)
    refresh_experience(run_dir)
    return stored


def record_analysis(
    run_dir: str | Path,
    *,
    round_idx: int,
    stage: str,
    analysis: str,
    evidence_refs: list[str],
    next_action: str,
    candidate_signature_value: str | None = None,
) -> dict[str, Any]:
    root = normalize_run_dir(run_dir)
    if candidate_signature_value is None:
        pending = _read_json(root / f"round_{round_idx:03d}" / "pending_mods.json", [])
        candidate_signature_value = candidate_signature(pending) if isinstance(pending, list) else None
    event = {
        "event_type": "analysis",
        "optimization_objective": objective_metadata(),
        "round_idx": round_idx,
        "candidate_signature": candidate_signature_value,
        "stage": stage,
        "analysis": analysis.strip(),
        "evidence_refs": evidence_refs,
        "next_action": next_action.strip(),
    }
    stored = _append_event(root, event)
    refresh_experience(root)
    return stored


def read_events(
    run_dir: str | Path,
    *,
    round_idx: int | None = None,
    component: str | None = None,
    query_id: str | None = None,
    signature: str | None = None,
    stage: str | None = None,
    limit: int = 50,
) -> list[dict[str, Any]]:
    events = _read_ledger(normalize_run_dir(run_dir))
    selected: list[dict[str, Any]] = []
    for event in reversed(events):
        if round_idx is not None and event.get("round_idx") != round_idx:
            continue
        if component and event.get("component_name") != component:
            continue
        if signature and not str(event.get("candidate_signature") or "").startswith(signature):
            continue
        if stage and event.get("stage") != stage:
            continue
        if query_id and not _event_mentions_query(event, query_id):
            continue
        selected.append(event)
        if len(selected) >= max(1, min(limit, 200)):
            break
    return list(reversed(selected))


def refresh_experience(run_dir: str | Path) -> Path:
    """Regenerate round summaries and the global compact digest."""
    root = normalize_run_dir(run_dir)
    root.mkdir(parents=True, exist_ok=True)
    events = _read_ledger(root)
    by_round: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for event in events:
        try:
            by_round[int(event.get("round_idx"))].append(event)
        except (TypeError, ValueError):
            continue

    round_paths = sorted(root.glob("round_[0-9][0-9][0-9]"))
    summaries: list[dict[str, Any]] = []
    for round_path in round_paths:
        round_idx = int(round_path.name.removeprefix("round_"))
        summary = _round_summary(root, round_path, round_idx, by_round.get(round_idx, []))
        _atomic_json(round_path / "round_summary.json", summary)
        summaries.append(summary)

    digest = _build_digest(root, summaries)
    path = root / DIGEST_FILENAME
    _atomic_json(path, digest)
    return path


def build_experience_context(
    run_dir: str | Path,
    *,
    round_idx: int,
    max_context_tokens: int = MAX_CONTEXT_TOKENS,
    history_token_budget: int | None = None,
) -> str:
    """Return a stable history pack that fits inside the configured budget."""
    root = normalize_run_dir(run_dir)
    refresh_experience(root)
    requested = history_token_budget or int(
        os.getenv("SEQ_ALIGN_HISTORY_MAX_TOKENS", str(DEFAULT_HISTORY_TOKENS))
    )
    budget = max(4_000, min(requested, max_context_tokens // 2))
    digest = _read_json(root / DIGEST_FILENAME, {})
    header = (
        "# Optimization experience (derived, auditable)\n"
        f"Current optimization objective: {json.dumps(objective_metadata(), ensure_ascii=False, sort_keys=True)}\n"
        f"Context ceiling: {max_context_tokens} tokens. Historical injection budget: {budget} tokens.\n"
        "Raw trajectories, exact patches, Slurm requests, and result shards are never deleted. "
        f"Candidate event ledger: {root / LEDGER_FILENAME}.\n"
        "Events without the current optimization_objective are legacy evidence only; do not reuse "
        "their acceptance rationale or quality-first goal. "
        "Use these summaries as evidence. Cite query IDs/artifact paths in analysis, and use "
        "read_optimization_history when older exact details are needed.\n"
    )
    digest_text = "\n## Global compressed experience\n" + json.dumps(
        digest, indent=2, ensure_ascii=False, sort_keys=True
    )
    content = _fit_text(header + digest_text, budget)
    if estimate_tokens(content) >= budget:
        return content

    round_summaries: list[tuple[int, dict[str, Any]]] = []
    for path in root.glob("round_[0-9][0-9][0-9]/round_summary.json"):
        try:
            idx = int(path.parent.name.removeprefix("round_"))
            round_summaries.append((idx, _read_json(path, {})))
        except ValueError:
            continue
    # Current round first, then the latest completed rounds. This guarantees
    # exact current code/evidence survives even when old history is large.
    ordered = sorted(
        round_summaries,
        key=lambda item: (item[0] == round_idx, item[0]),
        reverse=True,
    )
    for idx, summary in ordered:
        if idx != round_idx and idx < max(0, round_idx - 3):
            continue
        section = (
            f"\n\n## Detailed round {idx:03d}\n"
            + json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True)
        )
        if estimate_tokens(content + section) > budget:
            remaining = budget - estimate_tokens(content)
            if idx == round_idx and remaining > 1_000:
                content += _fit_text(section, remaining)
            break
        content += section
    return content


def _append_event(run_dir: str | Path, event: dict[str, Any]) -> dict[str, Any]:
    root = normalize_run_dir(run_dir)
    root.mkdir(parents=True, exist_ok=True)
    canonical = json.dumps(event, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    stored = {
        "schema_version": "seq_align_candidate_event.v1",
        "event_id": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
        "created_at": datetime.now(timezone.utc).isoformat(),
        **event,
    }
    ledger = root / LEDGER_FILENAME
    lock_path = root / f"{LEDGER_FILENAME}.lock"
    with lock_path.open("a+", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        existing_ids = {item.get("event_id") for item in _read_ledger(root)}
        if stored["event_id"] not in existing_ids:
            with ledger.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(stored, ensure_ascii=False, sort_keys=True) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
    return stored


def _read_ledger(root: Path) -> list[dict[str, Any]]:
    path = root / LEDGER_FILENAME
    if not path.is_file():
        return []
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            records.append(value)
    return records


def _round_summary(
    root: Path,
    round_path: Path,
    round_idx: int,
    events: list[dict[str, Any]],
) -> dict[str, Any]:
    pending = _read_json(round_path / "pending_mods.json", None)
    full = _read_eval(round_path / "full_eval.json")
    artifact_evals = [_slurm_artifact(job_dir) for job_dir in sorted((round_path / "slurm").glob("*_*"))]
    artifact_evals = [item for item in artifact_evals if item]
    trajectory_paths = sorted(str(path) for path in (root / "trajectories").glob(f"{round_path.name}/**/*") if path.is_file())
    lessons: list[str] = []
    event_objective_ids = sorted({
        str((event.get("optimization_objective") or {}).get("objective_id"))
        for event in events
        if (event.get("optimization_objective") or {}).get("objective_id")
    })
    if events and not event_objective_ids:
        event_objective_ids = ["legacy_unversioned"]
    for event in events:
        if event.get("event_type") == "analysis" and event.get("analysis"):
            lessons.append(str(event["analysis"]))
        elif event.get("event_type") == "evaluation":
            result = event.get("result") or {}
            decision = (result.get("acceptance") or {}).get("decision")
            lessons.append(
                f"{event.get('stage')} candidate {str(event.get('candidate_signature'))[:12]} "
                f"status={result.get('status')} ratio={result.get('tp_fp_ratio')} "
                f"cost_s={result.get('cost_s')} decision={decision}"
            )
    if not lessons and full:
        lessons.append(
            f"full artifact status={full.get('status')} ratio={full.get('tp_fp_ratio')} "
            f"cost_s={full.get('cost_s')} decision={(full.get('acceptance') or {}).get('decision')}"
        )
    return {
        "schema_version": "seq_align_round_summary.v1",
        "current_optimization_objective": objective_metadata(),
        "event_objective_ids": event_objective_ids,
        "round_idx": round_idx,
        "candidate_signature": candidate_signature(pending) if isinstance(pending, list) else None,
        "pending_mods": pending,
        "events": events,
        "artifact_evaluations": artifact_evals,
        "full_result": full,
        "lessons": lessons,
        "raw_artifacts": {
            "round_dir": str(round_path),
            "trajectory_files": trajectory_paths,
            "candidate_ledger": str(root / LEDGER_FILENAME),
        },
    }


def _slurm_artifact(job_dir: Path) -> dict[str, Any] | None:
    request = _read_json(job_dir / "request.json", None)
    if not isinstance(request, dict):
        return None
    submission = _read_json(job_dir / "submission.json", {})
    result_paths = sorted((job_dir / "results").glob("result_*.json"))
    results = [result for path in result_paths if (result := _read_eval_model(path)) is not None]
    query_rows: list[dict[str, Any]] = []
    for fallback, result in enumerate(results):
        profile = result.meta.get("profile") or {}
        profile_rows = profile.get("queries") or profile.get("top_slowest_queries") or []
        profile_row = profile_rows[0] if profile_rows and isinstance(profile_rows[0], dict) else {}
        query_rows.append({
            "query_index": result.meta.get("query_index", fallback),
            "query_id": result.meta.get("query_id") or profile_row.get("query_id"),
            "status": result.status,
            "tp": result.tp,
            "fp": result.fp,
            "tp_fp_ratio": result.tp_fp_ratio,
            "cost_s": result.cost_s,
            "iterations": result.iterations,
        })
    statuses = [result.status for result in results]
    aggregate_status = "running_or_incomplete"
    expected = int(submission.get("query_count") or len(request.get("query_paths") or []))
    if expected and len(results) == expected:
        aggregate_status = "timeout" if "timeout" in statuses else ("ok" if all(s == "ok" for s in statuses) else "run_error")
    return {
        "stage": request.get("stage"),
        "optimization_objective": request.get("optimization_objective"),
        "candidate_signature": candidate_signature(request.get("mods") or []),
        "mods": request.get("mods") or [],
        "job_id": submission.get("job_id"),
        "job_dir": str(job_dir),
        "status": aggregate_status,
        "completed_queries": len(results),
        "expected_queries": expected,
        "tp": sum(result.tp for result in results),
        "fp": sum(result.fp for result in results),
        "cost_s": round(sum(result.cost_s for result in results), 4),
        "query_results": query_rows,
    }


def _build_digest(root: Path, summaries: list[dict[str, Any]]) -> dict[str, Any]:
    component_outcomes: dict[str, Counter[str]] = defaultdict(Counter)
    query_stats: dict[str, dict[str, Any]] = {}
    accepted_lineage: list[dict[str, Any]] = []
    compact_rounds: list[dict[str, Any]] = []
    for summary in summaries:
        full = summary.get("full_result") or {}
        pending = summary.get("pending_mods") or []
        component = pending[-1].get("contract_name") if pending else "stock"
        decision = (full.get("acceptance") or {}).get("decision")
        status = full.get("status")
        if status:
            component_outcomes[str(component)][str(decision or status)] += 1
        if (full.get("acceptance") or {}).get("accepted") is True:
            accepted_lineage.append({
                "round_idx": summary.get("round_idx"),
                "component": component,
                "candidate_signature": summary.get("candidate_signature"),
                "tp": full.get("tp"),
                "fp": full.get("fp"),
                "tp_fp_ratio": full.get("tp_fp_ratio"),
                "cost_s": full.get("cost_s"),
                "decision": decision,
            })
        for source in [full, *(event.get("result") or {} for event in summary.get("events", []) if event.get("event_type") == "evaluation")]:
            for row in source.get("query_results") or []:
                query_id = str(row.get("query_id") or row.get("query_index"))
                stats = query_stats.setdefault(query_id, {"evaluations": 0, "timeouts": 0, "max_cost_ratio": None})
                stats["evaluations"] += 1
                stats["timeouts"] += int(row.get("status") == "timeout")
                ratio = row.get("cost_ratio")
                if ratio is not None:
                    stats["max_cost_ratio"] = max(float(ratio), stats["max_cost_ratio"] or float(ratio))
        compact_rounds.append({
            "round_idx": summary.get("round_idx"),
            "component": component,
            "candidate_signature": summary.get("candidate_signature"),
            "status": status,
            "tp": full.get("tp"),
            "fp": full.get("fp"),
            "tp_fp_ratio": full.get("tp_fp_ratio"),
            "cost_s": full.get("cost_s"),
            "decision": decision,
            "event_objective_ids": summary.get("event_objective_ids") or [],
            "lessons": (summary.get("lessons") or [])[-3:],
            "round_summary_path": str(root / f"round_{int(summary.get('round_idx', 0)):03d}" / "round_summary.json"),
        })
    return {
        "schema_version": "seq_align_experience_digest.v1",
        "optimization_objective": objective_metadata(),
        "max_context_tokens": MAX_CONTEXT_TOKENS,
        "history_default_tokens": DEFAULT_HISTORY_TOKENS,
        "accepted_lineage": accepted_lineage,
        "component_outcomes": {
            component: dict(sorted(counts.items()))
            for component, counts in sorted(component_outcomes.items())
        },
        "query_patterns": dict(sorted(query_stats.items())),
        "rounds": compact_rounds,
        "evidence": {
            "candidate_ledger": str(root / LEDGER_FILENAME),
            "raw_trajectories": str(root / "trajectories"),
        },
    }


def _read_eval(path: Path) -> dict[str, Any] | None:
    model = _read_eval_model(path)
    return compact_eval(model) if model is not None else None


def _read_eval_model(path: Path) -> EvalResult | None:
    if not path.is_file():
        return None
    try:
        return EvalResult.model_validate_json(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _changed_component(mods: list[dict[str, Any]], root: Path) -> str | None:
    accepted = _read_json(root / "accepted_mods.json", [])
    accepted_by_name = {item.get("contract_name"): item for item in accepted if isinstance(item, dict)}
    for mod in mods:
        if accepted_by_name.get(mod.get("contract_name")) != mod:
            return mod.get("contract_name")
    return mods[-1].get("contract_name") if mods else None


def _event_mentions_query(event: dict[str, Any], query_id: str) -> bool:
    for row in ((event.get("result") or {}).get("query_results") or []):
        if str(row.get("query_id")) == query_id:
            return True
    return query_id in (event.get("evidence_refs") or [])


def _read_json(path: Path, default: Any) -> Any:
    if not path.is_file():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return default


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def _fit_text(text: str, token_budget: int) -> str:
    if estimate_tokens(text) <= token_budget:
        return text
    marker = "\n... [historical context compressed; use read_optimization_history for raw evidence] ...\n"
    keep = max(500, int(len(text) * token_budget / estimate_tokens(text) * 0.9))
    while keep > 100:
        head = int(keep * 0.65)
        candidate = text[:head] + marker + text[-(keep - head):]
        if estimate_tokens(candidate) <= token_budget:
            return candidate
        keep = int(keep * 0.85)
    return marker[: max(1, token_budget)]
