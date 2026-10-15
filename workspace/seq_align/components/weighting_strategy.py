"""weighting_strategy component -- relative sequence weighting via Builder.

How sequences in the MSA are relatively weighted before HMM construction.
pyhmmer 0.12 Builder takes a *string-valued* ``weighting=`` kwarg (the HMMER
CLI flags --wpb/--wgsc/--wblosum/--wnone map to the strings below), NOT the
boolean CLI flag names. This is a parameterizable choice of weighting
*algorithm* (not just a number).
"""

from __future__ import annotations

from .base import ComponentContract, ComponentKind, ComponentModification, ComponentSpec

CONTRACT = ComponentContract(
    name="weighting_strategy",
    kind=ComponentKind.PYTHON,
    pyhmmer_hook="builder",
    description=(
        "Relative sequence weighting algorithm (pb/gsc/blosum/none) via "
        "plan7.Builder's `weighting` string kwarg. Algorithm choice, not a float."
    ),
)

# CLI flag -> pyhmmer Builder weighting string
_CLI_TO_PY = {
    "wpb": "pb",
    "wgsc": "gsc",
    "wblosum": "blosum",
    "wnone": "none",
}
_VALID = set(_CLI_TO_PY.values()) | set(_CLI_TO_PY.keys())


def materialize(mod: ComponentModification) -> ComponentSpec:
    """python_payload schema:
        {"weighting": "wpb"}   # CLI-style alias accepted, mapped to "pb"
        {"weighting": "pb"}    # or the raw pyhmmer string
    """
    payload = mod.python_payload or {"weighting": "wpb"}
    unknown = set(payload) - {"weighting", "wid"}
    if unknown:
        raise ValueError(f"Unsupported weighting_strategy fields: {sorted(unknown)}")
    w = payload.get("weighting", "wpb")
    if w not in _VALID:
        raise ValueError(f"Unknown weighting '{w}'. Valid: {sorted(_VALID)}")
    w_py = _CLI_TO_PY.get(w, w)
    normalized = {"weighting": w_py}
    if "wid" in payload:
        wid = float(payload["wid"])
        if w_py != "blosum":
            raise ValueError("wid is valid only with blosum weighting")
        if not 0.0 <= wid <= 1.0:
            raise ValueError("wid must be in [0, 1]")
        normalized["wid"] = wid
    return ComponentSpec(
        contract_name=CONTRACT.name,
        kind=ComponentKind.PYTHON,
        hook="builder",
        builder_kwargs={"weighting": w_py},
        pipeline_kwargs={},
        payload=normalized,
    )
