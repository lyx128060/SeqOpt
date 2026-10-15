# Profiling-only build (true per-stage timing) — Stage 2

**Purpose**: obtain the **true timing of each stage** of the HMMER speedup pipeline. HMMER reports only the total time per round and the **pass counts** of each filter level, never per-stage times; breaking them out requires inserting timers at the stage boundaries.

**This is not** the benchmark binary. The acceptance definition of `cost_s` must remain unchanged, so the profiling build is a **separate copy** with separate compile macros and results written to a separate file.

---

## Status (read this first)

| Part | Status |
|---|---|
| Parsing and wiring of the `p7_evo_prof.txt` data contract | ✅ **Implemented and tested** (`hmmer_stage_timing.parse_instrumented_stage_times` / `load_instrumented_stage_times`, 25 tests) |
| `p7_evo_prof.h` / `p7_evo_prof.c` | ✅ Written, reviewed as correct |
| **Patch injected into the contract functions + build wiring** | ⚠️ **Not implemented, not verified** — requires a real HMMER build to iterate |
| Trigger strategy (only on baseline and accepted) | ⚠️ See below; one design question must be settled first |

**I have not verified that a single line of the C compiles or runs**: no usable HMMER build environment is available on this side. The C files above are "written to spec and reviewable", not "verified".

---

## Injection points (HMMER 3.4)

Timing is inserted inside the **contract functions** — precisely the functions the agent is permitted to modify (the `components/c_kernels.py` whitelist) — so the profiling build and the candidate under test are semantically aligned.

| Contract slot | Target file | Target function | Corresponding pipeline stage |
|---|---|---|---|
| `ssv_filter` | `impl_neon/ssvfilter.c` | `p7_SSVFilter` | MSV/SSV filter (paper HMMERHEAD stages 1–2: word seeds + ungapped extension) |
| `dp_kernel` | `impl_neon/fwdback.c` | `forward_engine`, `backward_engine` | Viterbi/Forward filter and full scoring (paper stages 3–4) |
| `null2_correction` | `impl_neon/null2.c` | `p7_Null2_ByExpectation`, `p7_Null2_ByTrace` | composition bias correction |
| `domain_def` | — | no contract function yet | domain definition (no corresponding modifiable function in HMMER; **leave it empty for now**, do not modify non-contract files just to insert timing) |
| `pipeline_flow` | `src/src/p7_evo_align.c` | `p7_evo_align_Run` | per-round iteration driver |

For each function: `P7_EVO_PROF_ENTER(t0);` at entry and `P7_EVO_PROF_EXIT(SLOT, t0);` **before every `return`**. Note that missing a `return` silently undercounts time — this is the most common source of error in this kind of instrumentation.

---

## Build recipe

1. Copy the candidate's materialized source (reuse the existing `run_hmmer_cli._copy_source`).
2. Put `p7_evo_prof.{h,c}` into `src/src/`.
3. Instrument as in the table above (**do not** go through `assemble()` / `verify_c_patch` — that is the validator for candidate patches and will correctly reject these out-of-scope changes; profiling instrumentation must be applied outside of it).
4. Add `-DP7_EVO_PROF` at configure time and add `p7_evo_prof.c` to the build.
5. Call `p7_evo_prof_install_atexit()` once (for example at the start of `main()` in `jackhmmer.c`).
6. At runtime set `P7_EVO_PROF_OUT=<qdir>/p7_evo_prof.txt`.
7. No Python-side change is needed: `_query_profile` already loads this file automatically and attaches `time_s` to the corresponding contract.

One thing to note: `P7_EVO_PROF_ENTER` in `p7_evo_prof.h` relies on **C99 mid-block declarations**; HMMER 3.4 already relies on C99, but supporting `-std=c89` as well would require rewriting it.

---

## Trigger-strategy question that must be settled first

You chose "run profiling only on baseline and accepted". **This strategy cannot be decided before evaluation**: whether a candidate is accepted is known only after the evaluation has run and passed the acceptance gate.

And in the existing code, `external_profiler` is a **static setting** of the runner that takes effect for every evaluation (`runner.py` line 859), with no notion of "baseline / accepted".

Two feasible attachment points:

- **A. Run after commit**: trigger profiling when `runner.commit_evaluation()` is called (the controller only calls it on acceptance); for the baseline, trigger it at `evaluate_stock_baseline()`. The semantics fit best; the cost is that an accepted candidate must be evaluated with profiling one more time.
- **B. Switch to "collect on every evaluation, keep only what is needed"**: the profiling build still runs, but only the baseline and the finally accepted candidate write `stage_timing` into the artifacts. This avoids the repeated evaluation, but every inner trial pays the profiling overhead (10× cost), which contradicts the original intent of keeping cost controllable.

**This choice changes cost by an order of magnitude, and I have not decided it for you.** A fits the strategy you chose better, but it requires adding a "profile after commit" process step.

---

## Honest boundaries

- Per-stage timing **inevitably carries instrumentation overhead**, especially `p7_SSVFilter`, which is called per target sequence. These numbers are for **attribution** (which segment is hottest), not for acceptance; do not subtract them from or compare them against `cost_s`.
- In the contract, `domain_def` has no instrumentable function (HMMER's domain definition is outside the agent-modifiable scope). Do not modify non-contract files just to complete the table — that would break the premise "profiling build == candidate under test". The current implementation leaves `domain_def`'s `time_s` empty, which is the correct behavior.
