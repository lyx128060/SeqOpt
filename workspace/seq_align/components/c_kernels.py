"""c_kernels -- contracts for the algorithmic C kernels.

These components are the parts of HMMER that Python *cannot* constrain at
runtime (no plugin ABI): the dedicated comparison/iteration algorithm islands,
DP recurrences, dominant filter hot paths, domain-definition posterior
heuristics, and null2 composition-bias correction.
The agent edits these as unified diffs against specific, named functions in the
hmmer-3.4 source tree; the Assembler verifies each diff only touches its
contract's allowed functions before applying + recompiling in the Docker sandbox.

The contract's c_target_file / c_target_functions are the structural constraint
that realizes "the Python orchestration layer constrains the C optimization
space": the set of legal C edit locations is explicit and finite, so an LLM
patch touching an unrelated file is rejected outright.
"""

from __future__ import annotations

from .base import ComponentContract, ComponentKind


# Dedicated comparison island. The public p7_Pipeline ABI and the exact stock
# fallback live in frozen driver files; implementations are confined here and
# receive one opaque state object per P7_PIPELINE/worker.
pipeline_flow = ComponentContract(
    name="pipeline_flow",
    kind=ComponentKind.C_PATCH,
    c_target_file="p7_evo_align.c",
    c_target_functions=[
        "p7_evo_align_Create",
        "p7_evo_align_Reuse",
        "p7_evo_align_Destroy",
        "p7_evo_align_Run",
    ],
    c_entry_functions=["p7_evo_align_Run"],
    c_allow_translation_unit=True,
    description=(
        "Agent-editable sequence/profile comparison island in tools/"
        "hmmer-3.4/src/src/p7_evo_align.c. The p7_Pipeline input/output ABI "
        "and p7_evo_align_stock_Run fallback are frozen; implementations may "
        "use only worker-local opaque state and must preserve TP/FP behavior."
    ),
)


# Dedicated jackhmmer cross-round island. Database I/O, work distribution,
# output, and checkpointing remain in the frozen serial/thread/MPI drivers.
jackhmmer_loop = ComponentContract(
    name="jackhmmer_loop",
    kind=ComponentKind.C_PATCH,
    c_target_file="p7_evo_iter.c",
    c_target_functions=[
        "p7_evo_iter_Create",
        "p7_evo_iter_Destroy",
        "p7_evo_iter_BeginRound",
        "p7_evo_iter_EndRound",
    ],
    c_entry_functions=["p7_evo_iter_BeginRound"],
    c_entry_binaries=["jackhmmer"],
    c_allow_translation_unit=True,
    description=(
        "Agent-editable jackhmmer iteration island in tools/hmmer-3.4/src/"
        "src/p7_evo_iter.c. It controls round profile construction and the "
        "hits-to-MSA/convergence policy through a frozen query-local API; "
        "Slurm, database I/O, worker scheduling, and presentation stay frozen."
    ),
)

# DP recurrences: target the optimized SSE path used by the normal CLI build.
# Generic reference implementations are readable, but they can compile without
# affecting jackhmmer/hmmsearch runtime on this cluster.
dp_kernel = ComponentContract(
    name="dp_kernel",
    kind=ComponentKind.C_PATCH,
    c_target_file="impl_sse/fwdback.c",
    c_target_functions=[
        "forward_engine",     # optimized Forward recurrence engine
        "backward_engine",    # optimized Backward recurrence engine
    ],
    c_entry_functions=["forward_engine", "backward_engine"],
    description=(
        "Optimized SSE Forward/Backward DP recurrence engines in tools/"
        "hmmer-3.4/src/src/impl_sse/fwdback.c. Patches land in the runtime "
        "path used by rebuilt jackhmmer/hmmsearch on this cluster."
    ),
)

# Versioned AArch64/NEON runtime contract. Keep the unversioned SSE contract
# above immutable so historical content-addressed patchsets continue to replay
# against the source file they were authored for.
dp_kernel_neon = ComponentContract(
    name="dp_kernel_neon",
    kind=ComponentKind.C_PATCH,
    c_target_file="impl_neon/fwdback.c",
    c_target_functions=[
        "forward_engine",     # optimized Forward recurrence engine
        "backward_engine",    # optimized Backward recurrence engine
    ],
    c_entry_functions=["forward_engine", "backward_engine"],
    description=(
        "Versioned AArch64 NEON Forward/Backward DP recurrence engines in "
        "tools/hmmer-3.4/src/src/impl_neon/fwdback.c. Patches land in the "
        "runtime path used by rebuilt jackhmmer/hmmsearch on this cluster; "
        "the legacy dp_kernel contract remains bound to its historical SSE path."
    ),
)

_SSV_BAND_FUNCTIONS = [f"calc_band_{idx}" for idx in range(1, 19)]

# SSV/MSV-style vector filter: gprof on full UniRef90 repeatedly mapped
# dominant jackhmmer time to calc_band_* in this optimized SSE implementation.
ssv_filter = ComponentContract(
    name="ssv_filter",
    kind=ComponentKind.C_PATCH,
    c_target_file="impl_sse/ssvfilter.c",
    c_target_functions=[
        *_SSV_BAND_FUNCTIONS,
        "get_xE",
        "p7_SSVFilter",
    ],
    # The pipeline enters this island through p7_SSVFilter.
    c_entry_functions=["p7_SSVFilter"],
    description=(
        "Optimized SSE SSV filter / vector band scoring in tools/hmmer-3.4/"
        "src/src/impl_sse/ssvfilter.c. gprof on full UniRef90 identified "
        "calc_band_* as dominant jackhmmer hotspots; patches here must preserve "
        "filter score semantics and TP/FP behavior."
    ),
)

# Versioned AArch64/NEON SSV contract. The function names intentionally mirror
# the SSE implementation, but the component name and target path are distinct
# so an old SSE diff can never be validated as a NEON patch.
ssv_filter_neon = ComponentContract(
    name="ssv_filter_neon",
    kind=ComponentKind.C_PATCH,
    c_target_file="impl_neon/ssvfilter.c",
    c_target_functions=[
        *_SSV_BAND_FUNCTIONS,
        "get_xE",
        "p7_SSVFilter",
    ],
    c_entry_functions=["p7_SSVFilter"],
    description=(
        "Versioned AArch64 NEON SSV filter / vector band scoring in tools/"
        "hmmer-3.4/src/src/impl_neon/ssvfilter.c. Patches must preserve filter "
        "score semantics, overflow/J-state fallback behavior, MAX_BANDS guards, "
        "and TP/FP behavior; the legacy ssv_filter contract remains SSE-bound."
    ),
)

# FASTA parsing/data conversion is now an active optimization surface. Its
# existing contract name and source boundary stay unchanged, so historical
# sequence_io trajectories remain directly replayable.
sequence_io = ComponentContract(
    name="sequence_io",
    kind=ComponentKind.C_PATCH,
    c_target_file="easel/esl_sqio_ascii.c",
    c_target_functions=[
        "seebuf",
        "addbuf",
        "header_fasta",
    ],
    c_entry_functions=["addbuf"],
    description=(
        "FASTA sequence parser hot path in tools/hmmer-3.4/src/easel/"
        "esl_sqio_ascii.c. gprof on full UniRef90 identified addbuf, seebuf, "
        "and header_fasta as dominant jackhmmer hotspots while reading the "
        "target database; patches here must preserve accepted FASTA parsing "
        "and residue/name semantics."
    ),
)

# Domain definition: posterior heuristics that call/score domains.
domain_def = ComponentContract(
    name="domain_def",
    kind=ComponentKind.C_PATCH,
    c_target_file="p7_domaindef.c",
    c_target_functions=[
        "p7_domaindef_ByPosteriorHeuristics",
        "p7_domaindef_Create",
        "p7_domaindef_Reuse",
    ],
    c_entry_functions=["p7_domaindef_ByPosteriorHeuristics"],
    description=(
        "Domain-definition posterior heuristics in tools/hmmer-3.4/src/"
        "p7_domaindef.c. Patches altering how domains are called/scored land here."
    ),
)

# null2 / composition-bias correction: target the optimized SSE implementation
# called by p7_domaindef.c in the normal CLI runtime.
null2_correction = ComponentContract(
    name="null2_correction",
    kind=ComponentKind.C_PATCH,
    c_target_file="impl_sse/null2.c",
    c_target_functions=[
        "p7_Null2_ByExpectation",
        "p7_Null2_ByTrace",
    ],
    c_entry_functions=["p7_Null2_ByExpectation", "p7_Null2_ByTrace"],
    description=(
        "Runtime Null2 / composition-bias correction in tools/hmmer-3.4/"
        "src/src/impl_sse/null2.c. Patches altering bias correction land in "
        "the optimized p7_Null2_* functions called by domain definition."
    ),
)

# Versioned AArch64/NEON null2 implementation used by the current runtime.
# Retain null2_correction above unchanged for historical SSE patch replay.
null2_correction_neon = ComponentContract(
    name="null2_correction_neon",
    kind=ComponentKind.C_PATCH,
    c_target_file="impl_neon/null2.c",
    c_target_functions=[
        "p7_Null2_ByExpectation",
        "p7_Null2_ByTrace",
    ],
    c_entry_functions=["p7_Null2_ByExpectation", "p7_Null2_ByTrace"],
    description=(
        "Versioned AArch64 NEON Null2 / composition-bias correction in tools/"
        "hmmer-3.4/src/src/impl_neon/null2.c. Patches alter the optimized "
        "p7_Null2_* functions called by domain definition while the legacy "
        "null2_correction contract remains bound to its historical SSE path."
    ),
)

ACTIVE_C_KERNEL_CONTRACTS: dict[str, ComponentContract] = {
    pipeline_flow.name: pipeline_flow,
    jackhmmer_loop.name: jackhmmer_loop,
    dp_kernel_neon.name: dp_kernel_neon,
    ssv_filter_neon.name: ssv_filter_neon,
    null2_correction_neon.name: null2_correction_neon,
    sequence_io.name: sequence_io,
}


LEGACY_C_KERNEL_CONTRACTS: dict[str, ComponentContract] = {
    dp_kernel.name: dp_kernel,
    ssv_filter.name: ssv_filter,
    domain_def.name: domain_def,
    null2_correction.name: null2_correction,
}

# Compatibility registry for replaying/auditing old content-addressed runs.
# New outer/inner prompts use ACTIVE_C_KERNEL_CONTRACTS exclusively.
ALL_C_KERNEL_CONTRACTS: dict[str, ComponentContract] = {
    **ACTIVE_C_KERNEL_CONTRACTS,
    **LEGACY_C_KERNEL_CONTRACTS,
}


READABLE_C_KERNEL_CONTRACTS: dict[str, ComponentContract] = {
    **ALL_C_KERNEL_CONTRACTS,
    **LEGACY_C_KERNEL_CONTRACTS,
}
