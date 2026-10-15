"""CLI-equivalent hit-inclusion policy for iterative jackhmmer searches.

Full UniRef evaluation must stream the database through the HMMER CLI. This
component therefore exposes only policies that both smoke and full evaluation
can represent exactly with jackhmmer flags. Python-only callbacks are excluded
from the production search space instead of being silently dropped at full
evaluation time.
"""

from __future__ import annotations

from .base import ComponentContract, ComponentKind, ComponentModification, ComponentSpec


CONTRACT = ComponentContract(
    name="inclusion_strategy",
    kind=ComponentKind.PYTHON,
    pyhmmer_hook="pipeline_kwargs",
    description=(
        "Hit-inclusion policy represented identically by pyhmmer kwargs and "
        "jackhmmer CLI flags: sequence/domain E-value or bit-score gates."
    ),
)


def materialize(mod: ComponentModification) -> ComponentSpec:
    """Materialize a backend-portable inclusion policy.

    Supported payloads::

        {"rule": "evalue", "incE": 1e-3}
        {"rule": "bitscore", "incT": 25.0}
        {"rule": "domain_evalue", "incdomE": 1e-3}
        {"rule": "domain_bitscore", "incdomT": 25.0}
        {"rule": "evalue_with_domain", "incE": 1e-3, "incdomE": 1e-3}
        {"rule": "bitscore_with_domain", "incT": 25.0, "incdomT": 25.0}

    E-value and bit-score thresholds are intentionally not mixed for the same
    gate because HMMER treats those flag pairs as alternatives.
    """
    payload = mod.python_payload or {"rule": "evalue", "incE": 1e-3}
    rule = payload.get("rule", "evalue")
    valid_rules = {
        "evalue",
        "bitscore",
        "domain_evalue",
        "domain_bitscore",
        "evalue_with_domain",
        "bitscore_with_domain",
    }
    if rule not in valid_rules:
        raise ValueError(
            f"Unsupported production inclusion rule '{rule}'. "
            f"CLI-equivalent rules: {sorted(valid_rules)}"
        )

    allowed_by_rule = {
        "evalue": {"rule", "incE"},
        "bitscore": {"rule", "incT"},
        "domain_evalue": {"rule", "incdomE"},
        "domain_bitscore": {"rule", "incdomT"},
        "evalue_with_domain": {"rule", "incE", "incdomE"},
        "bitscore_with_domain": {"rule", "incT", "incdomT"},
    }
    unknown = set(payload) - allowed_by_rule[rule]
    if unknown:
        raise ValueError(f"Unsupported fields for inclusion rule '{rule}': {sorted(unknown)}")

    kwargs: dict[str, float] = {}
    if rule in {"evalue", "evalue_with_domain"}:
        kwargs["incE"] = float(payload.get("incE", 1e-3))
    if rule in {"bitscore", "bitscore_with_domain"}:
        kwargs["incT"] = float(payload.get("incT", 25.0))
    if rule in {"domain_evalue", "evalue_with_domain"}:
        kwargs["incdomE"] = float(payload.get("incdomE", 1e-3))
    if rule in {"domain_bitscore", "bitscore_with_domain"}:
        kwargs["incdomT"] = float(payload.get("incdomT", 25.0))
    for key, value in kwargs.items():
        if value < 0:
            raise ValueError(f"{key} must be >= 0")

    return ComponentSpec(
        contract_name=CONTRACT.name,
        kind=ComponentKind.PYTHON,
        hook="pipeline_kwargs",
        hook_factory=None,
        builder_kwargs={},
        pipeline_kwargs=kwargs,
        payload=dict(payload),
    )
