"""Component contracts: the stable interface each algorithm slot binds to.

The contract is what the Assembler validates and what the agent's patches land
on. Crucially, the agent never patches raw HMMER source files directly -- it
patches a *named component* whose contract constrains where C edits may land
and what portable policy a Python component may provide. This is what makes the
"Python orchestration layer structurally constrains the C optimization space"
design real rather than aspirational: the set of legal edit locations is
explicit and finite.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Callable

from pydantic import BaseModel, Field


class ComponentKind(str, Enum):
    """Where a component's modification lives.

    PYTHON : a validated, serializable policy mapped to equivalent HMMER CLI
             flags. No C edit or recompile. Covers thresholds, inclusion,
             scoring, weighting, and iteration budget.
    C_PATCH: applied as a git diff against a specific function/file in the
             hmmer-3.4 source tree, requiring recompile in the Docker sandbox.
             The algorithmic kernels (DP recurrences, domain definition, null2).
    """

    PYTHON = "python"
    C_PATCH = "c_patch"


class ComponentContract(BaseModel):
    """Stable contract for one algorithm slot.

    name : slot identifier (matches a key in the Assembler's component map)
    kind : PYTHON or C_PATCH -- determines how modifications are applied
    pyhmmer_hook : legacy field describing the policy integration point
    c_target_file : for C_PATCH components, the hmmer source file the patch
        must touch (e.g. "impl_sse/null2.c"). Patches touching other files
        are rejected by the Assembler -- this is the structural constraint.
    c_target_functions : for C_PATCH components, the function(s) a patch may
        modify within c_target_file. Narrowing to named functions keeps the
        LLM's C edits well-bounded.
    description : human-readable purpose, surfaced to the agent
    """

    name: str = Field(description="Slot identifier")
    kind: ComponentKind = Field(description="PYTHON or C_PATCH")
    pyhmmer_hook: str | None = Field(
        default=None,
        description="For PYTHON: portable policy integration point",
    )
    c_target_file: str | None = Field(
        default=None,
        description="For C_PATCH: hmmer source file the patch must touch"
    )
    c_target_functions: list[str] = Field(
        default_factory=list,
        description="For C_PATCH: functions within c_target_file a patch may modify"
    )
    c_entry_functions: list[str] = Field(
        default_factory=list,
        description=(
            "For C_PATCH: the subset of c_target_functions that must be present in the "
            "linked binaries. Empty means undeclared, and verification then checks nothing."
        ),
    )
    c_entry_binaries: list[str] = Field(
        default_factory=lambda: ["jackhmmer", "hmmsearch"],
        description="For C_PATCH: which linked programs must carry c_entry_functions",
    )
    c_allow_translation_unit: bool = Field(
        default=False,
        description=(
            "Allow edits anywhere in the existing target translation unit. Used only "
            "for pre-scaffolded algorithm islands whose exported ABI is frozen separately."
        ),
    )
    description: str = Field(default="", description="Human-readable purpose")


class ComponentModification(BaseModel):
    """One modification the agent proposes against a component slot.

    For PYTHON components, ``python_payload`` carries a serializable spec that
    the component module knows how to materialize into a pyhmmer hook (e.g. an
    inclusion threshold + rule name, or a weighting algorithm enum). The agent
    writes algorithm *logic* here, not bare floats -- e.g. an inclusion spec
    may say "include hits whose E-value < 1e-3 AND bit-score > 25" rather than
    just a single --incE number.

    For C_PATCH components, ``c_patch`` carries a unified diff string. The
    Assembler verifies it only touches ``contract.c_target_functions`` in
    ``contract.c_target_file``, then applies + recompiles in the sandbox.

    At most one of python_payload / c_patch is set, matching contract.kind.
    """

    contract_name: str = Field(description="Which component slot this modifies")
    python_payload: dict[str, Any] | None = Field(
        default=None,
        description="For PYTHON components: serializable spec materialized into a pyhmmer hook",
    )
    c_patch: str | None = Field(
        default=None,
        description="For C_PATCH components: unified diff against the contract's C file",
    )


# A materialized Python hook: the component module turns a ComponentModification
# into one of these, which the orchestrator wires into the jackhmmer call.
PyhmmerHook = Callable[..., Any]


class ComponentSpec(BaseModel):
    """A fully-resolved, validated component instance ready to assemble.

    Produced by a component module's ``materialize(modification)`` from a
    ComponentModification + its contract. The Assembler collects a dict of
    these and feeds them to the orchestrator (run_jack.py).
    """

    contract_name: str = Field(description="Slot this resolves")
    kind: ComponentKind = Field(description="PYTHON or C_PATCH")
    hook: str | None = Field(default=None, description="For PYTHON: pyhmmer hook name")
    hook_factory: Any | None = Field(
        default=None,
        description="For PYTHON: callable producing the hook (not serialized; runtime only)",
    )
    pipeline_kwargs: dict[str, Any] = Field(
        default_factory=dict,
        description="For PYTHON pipeline_kwargs kind: kwargs merged into pyhmmer.Pipeline"
    )
    builder_kwargs: dict[str, Any] = Field(
        default_factory=dict,
        description="For PYTHON builder kind: kwargs merged into plan7.Builder"
    )
    payload: dict[str, Any] = Field(
        default_factory=dict,
        description="Original normalized component payload, used by alternate backends"
    )
    c_patch: str | None = Field(default=None, description="For C_PATCH: the verified diff")
    c_target_file: str | None = Field(default=None, description="For C_PATCH: target file")
    c_target_functions: list[str] = Field(default_factory=list)
    c_allow_translation_unit: bool = False
