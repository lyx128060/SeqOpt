"""Unified evaluation result schema for the seq_align optimization pipeline.

All evaluation entry points (component-level unit tests, the small smoke subset,
and the full SCOP benchmark) return / consume :class:`EvalResult`. This is the
single structured contract between the LLM agent and the algorithm-under-test,
so that the agent always reads the same fields regardless of evaluation depth.

Why this exists
---------------
Previously the pipeline had three inconsistent CSV schemas (``BaseTime`` vs
``BaseTime_s`` vs ``OptTime_s``) and a self-referential TP definition (the
query's own hits were used as ground truth). This module replaces that ad-hoc
mess with one pydantic model surfaced to the LLM via
``FunctionSpec.output_schema`` (see ``evomaster/agent/tools/base.py``).

Fields
------
stage : which evaluation ran ("unit_test" | "smoke" | "full")
status : "ok" | "compile_error" | "run_error" | "timeout" | "skipped"
tp / fp : same-SCOP-superfamily positives and different-SCOP-fold negatives.
    Same-fold/different-superfamily and unclassified targets are recorded in
    ``meta.scop_scoring`` but excluded from the precision denominator.
tp_fp_ratio : precision = tp/(tp+fp); 0.0 when denominator is 0
cost_s : wall-clock seconds for the jackhmmer+hmmsearch run itself
n_seqs : number of query sequences evaluated (smoke=10, full=full benchmark)
unit_tests : per-assertion results (only populated by the unit_test stage)
iterations : jackhmmer iteration count actually run / converged at
errors : human-readable error strings (compile errors, run failures, ...)
meta : free-form bag for stage-specific extras (e.g. patch id, filter values)
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

EvalStage = Literal["unit_test", "smoke", "full"]
EvalStatus = Literal["ok", "compile_error", "run_error", "timeout", "skipped"]


class UnitTestResult(BaseModel):
    """Result of a single component-level assertion.

    A unit test exercises one algorithmic component (e.g. an inclusion_strategy
    callable, a convergence_rule) in isolation against a known input/expected
    pair, without running the full jackhmmer binary. Cheap and high-frequency.
    """

    name: str = Field(description="Assertion / component name")
    passed: bool = Field(description="Whether the assertion held")
    detail: str | None = Field(default=None, description="Failure detail or assertion message")


class EvalResult(BaseModel):
    """Unified evaluation result returned by every eval tool and the orchestrator.

    The LLM-facing tool spec for run_short_eval / run_full_eval declares this
    model as its output_schema so the agent can rely on the same field shape at
    every evaluation depth.
    """

    stage: EvalStage = Field(description="Which evaluation depth produced this result")
    status: EvalStatus = Field(description="Outcome of the evaluation")
    tp: int = Field(default=0, description="True positives (same SCOP superfamily as query)")
    fp: int = Field(default=0, description="False positives (different SCOP fold from query)")
    tp_fp_ratio: float = Field(default=0.0, description="Precision tp/(tp+fp); 0.0 if none")
    cost_s: float = Field(default=0.0, description="Wall-clock seconds of the jackhmmer run")
    n_seqs: int = Field(default=0, description="Number of query sequences evaluated")
    iterations: int = Field(default=0, description="jackhmmer iterations run / converged at")
    unit_tests: list[UnitTestResult] = Field(
        default_factory=list, description="Per-assertion results (unit_test stage mainly)"
    )
    errors: list[str] = Field(default_factory=list, description="Human-readable error strings")
    meta: dict[str, Any] = Field(default_factory=dict, description="Stage-specific extras")

    def ratio(self) -> float:
        """Compute precision from tp/fp; safe for zero denominator."""
        denom = self.tp + self.fp
        self.tp_fp_ratio = (self.tp / denom) if denom > 0 else 0.0
        return self.tp_fp_ratio
