"""Component slots for the Jackhmmer algorithm-optimization pipeline.

The optimization surface is decomposed into named *components*, each with a
fixed input/output contract. This is the core of the "decouple parameter
tuning" design: instead of the agent emitting four floats (F1-F4) that were
never even wired to the binary, it emits *modifications to components*, each of
which is either:

- a **portable policy specification** (for parameterizable surface: inclusion strategy,
  scoring system, filter cascade, weighting, effective-sequence-number,
  convergence rule) mapped to equivalent HMMER CLI flags, with NO C edit; or
- a **C source patch** (for profiled algorithmic/runtime paths including
  optimized SSE kernels, domain definition, and FASTA parsing) applied by the
  Assembler + Docker sandbox (apply
  patch -> recompile -> test -> git revert).

The production evaluator uses one CLI backend for smoke and full runs. This
keeps policy and C-patch candidates comparable while streaming Full UniRef90.

Modules:
- base.py            : ComponentContract base + ComponentKind enum
- inclusion_strategy : backend-portable inclusion policy
- convergence_rule   : stock convergence or fixed iteration budget
- scoring_system     : substitution matrix + gap params via Builder
- filter_cascade     : F1/F2/F3 + bias filter via Pipeline kwargs
- weighting_strategy : relative weighting (--wpb/--wgsc/...) via Builder
- eff_seq_strategy   : effective sequence number strategy via Builder
- c_kernels          : contracts for dp_kernel / domain_def / null2 (C patches)
"""

from .base import ComponentContract, ComponentKind, ComponentSpec, ComponentModification

__all__ = [
    "ComponentContract",
    "ComponentKind",
    "ComponentSpec",
    "ComponentModification",
]
