"""Deterministic inner-loop budget for one outer optimization round."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

from .pending_mods import normalize_run_dir


DEFAULT_INNER_TRIALS = 10


def inner_trial_limit() -> int:
    raw = os.getenv(
        "SEQ_ALIGN_INNER_TRIALS_OVERRIDE",
        os.getenv("SEQ_ALIGN_INNER_TRIALS", str(DEFAULT_INNER_TRIALS)),
    )
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"inner trial limit must be an integer, got {raw!r}") from exc
    if value < 1:
        raise ValueError("inner trial limit must be >= 1")
    return value


def candidate_signature(mods: list[dict[str, Any]]) -> str:
    payload = json.dumps(mods, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def submitted_smoke_candidates(run_dir: str | Path, round_idx: int) -> set[str]:
    """Return distinct candidate signatures that reached Slurm smoke submission."""
    root = normalize_run_dir(run_dir) / f"round_{round_idx:03d}" / "slurm"
    signatures: set[str] = set()
    for job_dir in root.glob("smoke_*") if root.is_dir() else ():
        request_path = job_dir / "request.json"
        submission_path = job_dir / "submission.json"
        if not request_path.is_file() or not submission_path.is_file():
            continue
        try:
            request = json.loads(request_path.read_text(encoding="utf-8"))
            mods = request.get("mods")
            if request.get("stage") == "smoke" and isinstance(mods, list):
                signatures.add(candidate_signature(mods))
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            continue
    return signatures


def inner_budget_status(
    run_dir: str | Path,
    round_idx: int,
    candidate_mods: list[dict[str, Any]],
) -> dict[str, Any]:
    seen = submitted_smoke_candidates(run_dir, round_idx)
    signature = candidate_signature(candidate_mods)
    limit = inner_trial_limit()
    already_tested = signature in seen
    return {
        "limit": limit,
        "used": len(seen),
        "remaining": max(0, limit - len(seen)),
        "candidate_signature": signature,
        "candidate_already_tested": already_tested,
        "candidate_allowed": already_tested or len(seen) < limit,
    }
