"""scoring_system component -- substitution matrix + gap params via Builder.

Exposed by pyhmmer's ``builder=`` (plan7.Builder): substitution matrix
(BLOSUM62/PAM30/... or a custom matrix file), gap-open/extend probabilities.
This is the parameterizable scoring surface; the DP recurrences that *use* the
matrix are a separate C_PATCH component (dp_kernel).
"""

from __future__ import annotations

from .base import ComponentContract, ComponentKind, ComponentModification, ComponentSpec

CONTRACT = ComponentContract(
    name="scoring_system",
    kind=ComponentKind.PYTHON,
    pyhmmer_hook="builder",
    description=(
        "Substitution matrix and gap params, injected via plan7.Builder. "
        "Parameterizable scoring surface (matrix choice + gap probabilities)."
    ),
)

_VALID_MATRICES = {
    "BLOSUM45", "BLOSUM50", "BLOSUM62", "BLOSUM80", "BLOSUM90",
    "PAM30", "PAM70", "PAM250",
}


def materialize(mod: ComponentModification) -> ComponentSpec:
    """python_payload schema:
        {"matrix": "BLOSUM62", "popen": 0.02, "pextend": 0.4}
        {"score_matrix": "BLOSUM45"}    # raw pyhmmer kwarg name (alias for matrix)

    pyhmmer 0.12 Builder accepts ``score_matrix=<NAME>`` (a string, NOT a
    matrix object and NOT a kwarg named ``matrix``/``mxfile`` -- those raise
    TypeError). Valid names are the built-in substitution matrices loaded by
    HMMER (BLOSUM45/50/62/80/90, PAM30/70/250, ...). Gap params are
    ``popen``/``pextend`` (probabilities).
    """
    payload = mod.python_payload or {"matrix": "BLOSUM62"}
    allowed = {"matrix", "score_matrix", "popen", "pextend"}
    unknown = set(payload) - allowed
    if unknown:
        raise ValueError(f"Unsupported scoring_system fields: {sorted(unknown)}")
    if "matrix" in payload and "score_matrix" in payload:
        raise ValueError("matrix and score_matrix are aliases; provide only one")
    kwargs: dict = {}
    matrix = payload.get("matrix") or payload.get("score_matrix")
    if matrix:
        if matrix not in _VALID_MATRICES:
            raise ValueError(
                f"Unknown substitution matrix '{matrix}'. "
                f"Valid: {sorted(_VALID_MATRICES)}."
            )
        kwargs["score_matrix"] = matrix
    if "popen" in payload:
        kwargs["popen"] = float(payload["popen"])
        if not 0.0 < kwargs["popen"] < 1.0:
            raise ValueError("popen must be in (0, 1)")
    if "pextend" in payload:
        kwargs["pextend"] = float(payload["pextend"])
        if not 0.0 < kwargs["pextend"] < 1.0:
            raise ValueError("pextend must be in (0, 1)")
    return ComponentSpec(
        contract_name=CONTRACT.name,
        kind=ComponentKind.PYTHON,
        hook="builder",
        builder_kwargs=kwargs,
        pipeline_kwargs={},
        payload=dict(payload),
    )
