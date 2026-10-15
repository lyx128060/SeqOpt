"""Shared optimization objective for prompts, evaluation, and audit records."""

from __future__ import annotations

import math
import os
from typing import Any


OBJECTIVE_ID = "quality_preserving_speed_v1"
DEFAULT_MIN_SPEEDUP_FRACTION = 0.03


def minimum_speedup_fraction() -> float:
    """Return the configured Full-evaluation speedup required for acceptance."""
    raw = os.getenv(
        "SEQ_ALIGN_MIN_SPEEDUP_FRACTION",
        str(DEFAULT_MIN_SPEEDUP_FRACTION),
    )
    value = float(raw)
    if not math.isfinite(value) or value <= 0.0 or value >= 1.0:
        raise ValueError("SEQ_ALIGN_MIN_SPEEDUP_FRACTION must be between 0 and 1")
    return value


def objective_metadata() -> dict[str, Any]:
    """Return the auditable objective contract used by the current process."""
    return {
        "objective_id": OBJECTIVE_ID,
        "primary_metric": "cost_s",
        "direction": "minimize",
        "hard_constraints": {
            "tp": "candidate >= accepted reference",
            "fp": "candidate <= accepted reference",
        },
        "minimum_speedup_fraction": minimum_speedup_fraction(),
        "ratio_role": "reported_only_not_an_acceptance_gate",
    }
