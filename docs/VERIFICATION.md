# Verification Mechanisms (A Focus of This Project)

This document explains how the question "is a speedup real?" is answered in this project, and **which checks do not constitute scientific validation**.
Three kinds of checking must be strictly distinguished; any statement that conflates them is wrong.

---

## 0. Why verification is the core problem here

The object under measurement is the wall-clock time of `jackhmmer` on UniRef90. The expected speedup is a **single-digit percentage**, while repeated measurements of the same query on the same cluster already fluctuate appreciably. Therefore:

> Unless we first establish how large pure noise can be, we cannot distinguish a real 1% speedup from a lucky 1% fluctuation.

The verification system of this project is built around this point.

---

## 1. First kind: program-level checks — **not** scientific validation

These checks answer "can this patch be applied, can it compile, is the modified location legal"; they **do not answer at all** whether it is faster or whether quality is unchanged.

| Mechanism | Implementation location | Rejection codes (observed counts) |
|---|---|---|
| Component contract whitelist | `components/c_kernels.py` | `rejected_patch_apply` (115), `rejected_plan_diff_binding` (24) |
| Patch applicability | `harness_plugins/check_patch_apply.py` | Same as above |
| Compile validation | `c_patch_loop/runner.py` | `rejected_compile_error` (65) |
| Static structural validation | `assembler.py` | `rejected_static` (3) |
| Parameter-tuning interception | `components/*` | `rejected_parameter_tuning` (2) |

> ⚠️ **Do not call this kind of check "independent validation".** It only guarantees that a patch is "formally legal".
> A patch that compiles can be slower, and can silently change the results.

The "parameter-tuning interception" is worth mentioning: `docs/AGENT_PROMPT.md` explicitly forbids trading speed for "changing thresholds / suppressing hits / reducing recall", and intercepts such wording and changes at the static stage.

---

## 2. Second kind: measurement-level verification — the actual "subsequent verification"

### 2.1 Comparability fingerprint (`c_patch_loop/comparability.py`)

Every acceptance decision is in essence **a difference or ratio of two wall-clock numbers**. Two numbers are comparable only when "everything that affects the measurement is identical".

The module extracts these fields into a fingerprint and **enforces equality**:

```
Required: backend, allocated_cpus, query_threads, stage, query_set_sha256,
      max_iter, uniref_db_path/sha256, astral_db_path/sha256, scop_txt_path/sha256
Compared but may be empty: partition, queries_in_job, hmmer_source_sha256,
      eval_repeats, paired_current_best_repeats, bundle_id, build_context
```

When they are unequal, it **raises `IncomparableMeasurement` and reports the differences field by field**, instead of returning a number that looks like a speedup ratio.

The module's docstring records the three **real** errors that prompted its creation:

1. Comparing costs across the local and Slurm backends;
2. Comparing costs across the 12h and 24h configurations;
3. A baseline that was overwritten by a profiling pass and still used as the reference.

Each of these once **silently produced a decision based on incomparable numbers**.

### 2.2 Paired interleaved re-measurement (paired interleaved confirmation)

**Problem**: the candidate is measured "now", while the reference is the cost "archived last time"; run-to-run drift gets counted as a speedup of the patch.

**Method**: within the same job, **interleave** R measurements each of the candidate and the current best, take the **median** of the R ratios as the decision quantity, and mark the decision as `accepted_speedup_paired`. Related rejection codes:

```
rejected_not_faster_paired
rejected_paired_confirmation_error
rejected_paired_incomplete_evidence
rejected_paired_quality_regression
```

**A real defect that was fixed** (see `tests/test_seq_align_paired_confirmation_wiring.py`): the staged loop used to hard-reset `paired_current_best_repeats` to 0, so the mechanism existed but **never took effect** (dead code); and the commit gate recomputed the speedup from the archived cost, so that even passing the parameter did not affect the decision. The test pins down both halves.

### 2.3 Null-effect control experiment (noise floor calibration)

**Design**: have **the same binary** play against itself (arm A vs arm B), interleaved within the same job. Any non-zero A/B ratio is then **necessarily pure measurement noise**, and its distribution is the floor that every future threshold must clear.

Results and definitions of terms: see [MEASUREMENT_CALIBRATION.md](MEASUREMENT_CALIBRATION.md) and `results/noise_floor_calibration.json`.

### 2.4 Decoy probe

**Design**: deliberately inject a patch **known only to add work** (run `get_xE()` one extra time, then spin for k times that duration), and see whether the acceptance gate judges it as a speedup. This is a "negative control".

**Result**: it **was judged to be a speedup** (`accepted_speedup_paired`, median +4.04%). Full evidence and explanation: see [LIMITATIONS.md](LIMITATIONS.md) and `results/decoy_probe_finding.json`.

This is the **most valuable verification output** of this project: it shows that "having a verification mechanism" and "having a verification mechanism that works" are two different things, and it gives the specific conditions under which it fails.

### 2.5 Stage-attribution gate

`hmmer_stage_timing.py` plus a C probe (`harness_plugins/profiling/`) do per-stage timing. If the **ordering of stage durations** differs between the new and the old profiling, the comparison script exits with a non-zero status, preventing subsequent jobs from continuing under a "hotspot hypothesis that no longer holds" — that is, it reports rather than silently retunes.

---

## 3. Third kind: the part requiring human judgement — explicitly isolated

`EXPERIENCE_DESIGN.md` keeps "measured facts" and "model hypotheses" separate:

- The summary (`compressed_summary.md`) **states only measured facts and citations**, each tagged with `eval_id` / `patchset_id` / `diff_sha256` for audit;
- Mechanistic / causal explanations must be registered separately through `record_optimization_analysis`.

In this way later agents do not inherit a previous round's model **explanation** as **evidence**.
"Why this patch is faster" is a scientific explanation and must be judged by humans; the code is only responsible for laying out the evidence.

---

## 4. Overview of the verification chain

```
Patch JSON
   │
   ├─[program level] contract whitelist → applicability → compilation   ← proves only "it runs"
   │
   ├─[measurement level] comparability fingerprint (rejects incomparable)
   │            └→ paired interleaved re-measurement (cancels drift)
   │                 └→ noise floor calibration (whether thresholds are meaningful)
   │                      └→ decoy probe (whether the gate is really effective)
   │
   └─[human level] measured facts vs model hypotheses separation (experience design)
```

---

## 5. Conclusion: what this system proves and what it does not

**Proved**
- Incomparable cost numbers can be rejected **mechanically**, rather than relying on people remembering to check;
- Paired interleaved re-measurement **can** produce a decision different from "measured now vs archived", and has indeed blocked several candidates (`rejected_not_faster_paired`);
- Measurement noise **can be quantified**, and the values measured are reproducible digit for digit from the raw records.

**Not proved**
- ❌ This method yields a stable N% speedup (only 7 samples have paired measurements);
- ❌ Quality preservation has been **independently** verified (TP/FP come from the same scoring pipeline, i.e. self-attestation from the same source);
- ❌ The acceptance gate is reliable in all configurations (one counterexample is documented).

See [LIMITATIONS.md](LIMITATIONS.md).
