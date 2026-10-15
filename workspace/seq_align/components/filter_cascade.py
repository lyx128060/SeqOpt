"""filter_cascade component -- MSV/Viterbi/Forward filter thresholds via Pipeline.

The acceleration pipeline thresholds F1 (MSV P-value), F2 (Viterbi), F3
(Forward) plus the bias-filter on/off, AND the reporting E-value/bit-score
thresholds (E/T/domE/domT). pyhmmer's plan7.Pipeline takes all of these as
kwargs. NOTE: per the HMMER docs, changing F1/F2/F3 only gates which targets
proceed to postprocessing; the reporting thresholds E/T/domE gate which
surviving hits are actually reported (and thus enter TP/FP scoring). None of
these alter bit scores / E-values / alignments (those come from the
Forward/Backward postprocessing, a C_PATCH component). So this component is a
clean parameterizable surface -- and E/domE are the knobs whose change is
directly visible in TP/FP.
"""

from __future__ import annotations

from .base import ComponentContract, ComponentKind, ComponentModification, ComponentSpec

CONTRACT = ComponentContract(
    name="filter_cascade",
    kind=ComponentKind.PYTHON,
    pyhmmer_hook="pipeline_kwargs",
    description=(
        "MSV/Viterbi/Forward filter P-value thresholds (F1/F2/F3), bias filter, "
        "and reporting E-value/bit-score thresholds (E/T/domE/domT), via "
        "pyhmmer.Pipeline kwargs. E/domE gate which hits are reported (visible "
        "in TP/FP)."
    ),
)


def materialize(mod: ComponentModification) -> ComponentSpec:
    """python_payload schema:
        {"F1": 0.02, "F2": 1e-3, "F3": 1e-5, "bias_filter": true}
        {"E": 1e-3, "domE": 1e-3}      # reporting E-value thresholds
        {"T": 25.0, "domT": 25.0}      # reporting bit-score thresholds
        {"max": true}   # disable all filters (F1=F2=F3=1.0, no bias)
    """
    payload = mod.python_payload or {}
    allowed = {"F1", "F2", "F3", "bias_filter", "max", "E", "T", "domE", "domT"}
    unknown = set(payload) - allowed
    if unknown:
        raise ValueError(f"Unsupported filter_cascade fields: {sorted(unknown)}")
    if "E" in payload and "T" in payload:
        raise ValueError("E and T are alternative reporting thresholds")
    if "domE" in payload and "domT" in payload:
        raise ValueError("domE and domT are alternative domain-reporting thresholds")
    if payload.get("max") and set(payload) & {"F1", "F2", "F3", "bias_filter"}:
        raise ValueError("max cannot be combined with F1/F2/F3/bias_filter")
    kwargs: dict = {}
    if payload.get("max"):
        # mirror HMMER --max: disable all filters
        kwargs["F1"] = 1.0
        kwargs["F2"] = 1.0
        kwargs["F3"] = 1.0
        kwargs["bias_filter"] = False
    else:
        for k in ("F1", "F2", "F3"):
            if k in payload:
                kwargs[k] = float(payload[k])
                if not 0.0 < kwargs[k] <= 1.0:
                    raise ValueError(f"{k} must be in (0, 1]")
        if "bias_filter" in payload:
            kwargs["bias_filter"] = bool(payload["bias_filter"])
    # reporting thresholds -- these are the visible TP/FP gates
    for k in ("E", "T", "domE", "domT"):
        if k in payload:
            kwargs[k] = float(payload[k])
            if kwargs[k] < 0:
                raise ValueError(f"{k} must be >= 0")
    return ComponentSpec(
        contract_name=CONTRACT.name,
        kind=ComponentKind.PYTHON,
        hook="pipeline_kwargs",
        pipeline_kwargs=kwargs,
        builder_kwargs={},
        payload=dict(payload),
    )
