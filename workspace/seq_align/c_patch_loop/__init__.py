"""Deterministic C-patch optimization loop for seq_align.

The modules in this package keep C-patch search separate from the legacy broad
agent/tool loop: a controller builds a bounded context pack, asks a proposer for
one structured patch plan, asks for one C diff, then validates, evaluates, and records immutable evidence.
"""

from .models import (
    ContextPackRecord,
    EvaluationRecord,
    ExperienceRecord,
    PatchPlanProposal,
    PatchPlanRecord,
    PatchProposal,
    PatchsetRecord,
)
from .store import LoopStore

__all__ = [
    "ContextPackRecord",
    "EvaluationRecord",
    "ExperienceRecord",
    "LoopStore",
    "PatchPlanProposal",
    "PatchPlanRecord",
    "PatchProposal",
    "PatchsetRecord",
]
