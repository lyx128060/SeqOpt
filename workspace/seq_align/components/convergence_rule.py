"""Backend-portable jackhmmer iteration policy.

The production evaluator streams Full UniRef through the HMMER CLI. HMMER can
represent its stock convergence rule or a fixed maximum iteration budget. More
elaborate Python checkpoint predicates cannot stop an already-running CLI
search, so they are rejected rather than producing misleading cost results.
"""

from __future__ import annotations

from .base import ComponentContract, ComponentKind, ComponentModification, ComponentSpec


CONTRACT = ComponentContract(
    name="convergence_rule",
    kind=ComponentKind.PYTHON,
    pyhmmer_hook="iteration_budget",
    description=(
        "Iteration policy with identical smoke/full semantics: stock "
        "no-new-target convergence or a fixed maximum iteration budget."
    ),
)


def materialize(mod: ComponentModification) -> ComponentSpec:
    payload = mod.python_payload or {"rule": "no_new_targets"}
    rule = payload.get("rule", "no_new_targets")
    if rule not in {"no_new_targets", "fixed"}:
        raise ValueError(
            f"Unsupported production convergence rule '{rule}'. "
            "CLI-equivalent rules: ['fixed', 'no_new_targets']"
        )
    allowed = {"rule", "n"} if rule == "fixed" else {"rule"}
    unknown = set(payload) - allowed
    if unknown:
        raise ValueError(f"Unsupported fields for convergence rule '{rule}': {sorted(unknown)}")
    normalized = {"rule": rule}
    if rule == "fixed":
        normalized["n"] = max(1, int(payload.get("n", 3)))
    return ComponentSpec(
        contract_name=CONTRACT.name,
        kind=ComponentKind.PYTHON,
        hook="iteration_budget",
        hook_factory=None,
        builder_kwargs={},
        pipeline_kwargs={},
        payload=normalized,
    )
