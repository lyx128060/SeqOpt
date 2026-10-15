"""eff_seq_strategy component -- effective sequence number strategy via Builder.

How the effective sequence count is determined. pyhmmer 0.12 Builder takes a
*string-valued* ``effective_number=`` kwarg ('entropy'/'clust'/'none'), or a
fixed count via ``effective_number=<int>``. The HMMER CLI flags
--eent/--eclust/--enone/--eset map as below. Applied through plan7.Builder.
A parameterizable choice of eff-seq *algorithm*.
"""

from __future__ import annotations

from .base import ComponentContract, ComponentKind, ComponentModification, ComponentSpec

CONTRACT = ComponentContract(
    name="eff_seq_strategy",
    kind=ComponentKind.PYTHON,
    pyhmmer_hook="builder",
    description=(
        "Effective sequence number strategy (entropy/clust/none/fixed) via "
        "plan7.Builder's `effective_number` kwarg. Algorithm choice for HMM "
        "parameterization."
    ),
)

# CLI flag -> pyhmmer Builder effective_number value
_CLI_TO_PY = {
    "eent": "entropy",
    "eclust": "clust",
    "enone": "none",
}
_VALID = set(_CLI_TO_PY.values()) | set(_CLI_TO_PY.keys()) | {"eentexp", "eset"}


def materialize(mod: ComponentModification) -> ComponentSpec:
    """python_payload schema:
        {"effn": "eent", "ere": 0.018, "esigma": 45.0}
        {"effn": "eset", "eset_value": 100}   # fixed integer count
    """
    payload = mod.python_payload or {"effn": "eent"}
    allowed = {"effn", "eset", "eset_value", "ere", "esigma"}
    unknown = set(payload) - allowed
    if unknown:
        raise ValueError(f"Unsupported eff_seq_strategy fields: {sorted(unknown)}")
    e = payload.get("effn", "eent")
    if e not in _VALID:
        raise ValueError(f"Unknown eff-seq strategy '{e}'. Valid: {sorted(_VALID)}")
    if set(payload) & {"ere", "esigma"} and e not in {"eent", "entropy", "eentexp"}:
        raise ValueError("ere/esigma are valid only with entropy strategies")
    if e == "eset":
        n = int(payload.get("eset_value", payload.get("eset", 100)))
        if n <= 0:
            raise ValueError("eset_value must be > 0")
        return ComponentSpec(
            contract_name=CONTRACT.name, kind=ComponentKind.PYTHON,
            hook="builder", builder_kwargs={"effective_number": n},
            pipeline_kwargs={}, payload={"effn": "eset", "eset_value": n},
        )
    e_py = _CLI_TO_PY.get(e, "entropy" if e == "eentexp" else e)
    normalized = {"effn": e}
    for key in ("ere", "esigma"):
        if key in payload:
            value = float(payload[key])
            if value <= 0:
                raise ValueError(f"{key} must be > 0")
            normalized[key] = value
    return ComponentSpec(
        contract_name=CONTRACT.name, kind=ComponentKind.PYTHON,
        hook="builder", builder_kwargs={"effective_number": e_py},
        pipeline_kwargs={}, payload=normalized,
    )
