# Evaluation Protocol

This document defines the subject of evaluation, the inputs and outputs, the metrics, the data
splits, and the summary format. It is kept separate from the verification mechanism: **this
document describes how measurement is done; [VERIFICATION.md](VERIFICATION.md) describes whether
the measured numbers count.**

---

## 1. Subject of evaluation

| Item | Content |
|---|---|
| Program under test | HMMER 3.4 `jackhmmer` (optionally `hmmsearch`), primarily the AArch64/NEON path |
| Variable under test | **C source patches** proposed by the agent, restricted to the component contract |
| Reference | Top of the current accepted lineage (stock, or the patch accepted in the previous round) |

---

## 2. Fixed evaluation pipeline (not modifiable)

```
For each query:
  ① jackhmmer -N<max_iter>  search UniRef90, take the highest-iteration checkpoint
  ② hmmsearch <checkpoint>  search the ASTRAL 40% subset
  ③ parse hmmsearch.tbl
  ④ score against SCOPe lineage → TP / FP
Timing: combined wall clock of both stages = cost_s
```

`docs/AGENT_PROMPT.md` lists this pipeline as a hard constraint:
**the agent must not modify it, nor bypass it by running HMMER directly.**
For the reason, see [ARCHITECTURE.md §6](ARCHITECTURE.md): different pipelines make costs
incomparable.

---

## 3. Inputs and outputs

### Inputs

| Input | Form | Notes |
|---|---|---|
| Query set | FASTA + `query_manifest.json` | Sampled from ASTRAL with a fixed seed; the manifest records query IDs and database SHA-256 |
| Search database | `UNIREF_DB` | UniRef90 for the full stage |
| Scoring database | `ASTRAL_DB` + `SCOP_TXT` | Hit determination and lineage ground truth |
| Patch | `PatchProposal` JSON | `unified_diff` (preferred) or exact edit blocks |
| Resources | Slurm configuration | partition / CPU / memory / wall clock / concurrency |

### Outputs

Each evaluation produces one `EvaluationRecord` (JSON, written to `RUN_DIR/c_patch_loop/evaluations/`):

```jsonc
{
  "eval_id": "<content hash>",
  "stage": "smoke" | "full",
  "status": "ok" | "compile_error" | "run_error" | "timeout",
  "decision": "accepted_speedup_paired" | "rejected_not_faster_paired" | ...,
  "eval_result": {
    "tp": 424, "fp": 9, "tp_fp_ratio": 0.9792,
    "cost_s": 124.3849, "n_seqs": 50, "iterations": 1
  },
  "metric_delta": { /* difference from the reference, thresholds, paired confirmation details */ },
  "dataset_manifest": { /* paths and SHA-256 of the three databases */ },
  "source_hashes":  { /* HMMER source / patchset hashes */ },
  "eval_config":    { /* configuration actually used, including backend and core count */ },
  "violations": []
}
```

---

## 4. Metrics and how they are computed

| Metric | Definition | Source |
|---|---|---|
| **TP** | number of domains that are hits and share the query's SCOP lineage | `scorer.py` |
| **FP** | number of hits whose lineage does not match | `scorer.py` |
| **TP_FP_Ratio** | `TP / (TP + FP)` | derived |
| **cost_s** | combined wall-clock seconds of both stages (jackhmmer + hmmsearch) | `run_hmmer_cli.py` `time.perf_counter` |
| **iterations** | actual number of jackhmmer iterations | checkpoint parsing |

> ⚠️ **Note on the metric definition**: the value of `TP_FP_Ratio` depends on the scoring
> procedure, and TP/FP come from the **same** SCOPe pipeline. This is **self-confirmation from
> the same source** and does not constitute independent validation of quality.
> See [LIMITATIONS.md §2.2](LIMITATIONS.md).

---

## 5. Data splits

| Split | Purpose | Size (as recorded) |
|---|---|---|
| **smoke** | Fast screen: eliminate patches that clearly cannot be used or applied | Small query set (10 built-in examples) |
| **full** | The **only** basis for acceptance | Fixed query set, at most 10 concurrent, `queries_per_job=5` |

**Hard rule**: when the full query set is missing the run **fails outright**, and smoke results
**must not** be passed off as full.
(`README_SeqOpt_Pipeline.md`, original wording: "When the full query set is missing the run
fails directly; smoke results are never passed off as Full.")

**Design intent of the split**: smoke is only a filter and **produces no** speedup numbers usable
for conclusions. Any statement of the form "speedup of X %" that comes from smoke is invalid.

---

## 6. Acceptance gate

```
① Quality gate (hard): TP >= reference.TP * accept_min_tp_fraction
                       and FP <= reference.FP
② Speed gate:          cost_s < reference.cost_s * (1 - min_speedup_fraction)
③ Structural constraint: exactly one component changed
④ Comparability:       the comparability fingerprints of candidate and reference must match
                       (otherwise the comparison is refused and no verdict is issued)
⑤ Paired confirmation: with paired_current_best_repeats >= 2, decide by interleaved re-measurement
```

Objective ID: `speed_under_tp_fp_non_regression`.

**A candidate is committed only if ①–⑤ all pass**; `min_speedup_fraction` must be meaningful
relative to the noise of the current configuration
(see [MEASUREMENT_CALIBRATION.md](MEASUREMENT_CALIBRATION.md)).

---

## 7. Anomalies, failure samples, and edge cases

| Case | Rejection code | Handling |
|---|---|---|
| Patch cannot be applied | `rejected_patch_apply` | Returned to the agent for rewriting (observed 115 times) |
| Compilation failure | `rejected_compile_error` | Same as above (observed 65 times) |
| Patch inconsistent with the plan binding | `rejected_plan_diff_binding` | Same as above (observed 24 times) |
| Parameter tuning only | `rejected_parameter_tuning` | Blocked statically (observed 2 times) |
| Not faster | `rejected_not_faster` / `rejected_not_faster_paired` | Recorded as trajectory evidence, not inherited |
| Incomplete paired evidence | `rejected_paired_incomplete_evidence` | Not committed |
| Quality regression | `rejected_quality_tp_regression` | Not committed (observed 2 times) |
| Single-query timeout | `rejected_timeout` | Write the query result, reject and seal the candidate, advance the outer layer (observed 16 times) |
| Infrastructure error | `run_error` | **Retryable**, not the candidate's fault (observed 16 times) |

**How the timeout cap is set**: the wall-clock cap for each query is frozen at **1.2× the current
accepted Full time of that same query** (`per_query_cost_ratio_cap`, independently calibrated to
the 99.9th percentile = 1.2401).

**Key distinction**: `rejected_timeout` (the candidate is too slow) and `run_error` (an
infrastructure problem) must be kept apart — the former is a property of the candidate, the
latter is not.

---

## 8. Result storage and summary

```
RUN_DIR/
├── round_NNN/full_eval.json          # per-round evaluation snapshot
├── c_patch_loop/
│   ├── evaluations/<eval_id>.json    # immutable evaluation record
│   ├── patchsets/<id>.json           # exact diff + contract + hashes
│   ├── patch_plans/<id>.json         # validated algorithm plan
│   ├── context_packs/<id>.json       # what the model was allowed to see at the time
│   ├── experience/experience.jsonl   # one experience per result
│   ├── slot_events/                  # slot state machine events
│   └── indexes/                      # regenerable projections (speed-best / lineage / leaderboard)
└── data_test/query_manifest.json     # query set fingerprint
```

The summary layer is the **`experiment ledger`** (`ledger.py`): it compresses the records above
into one append-only table, each row carrying six classes of provenance hash (query set / UniRef /
ASTRAL / SCOPe / HMMER source / patchset). Total ledger size seen while assembling this document:
**1367 records**, of which `full` 648, `smoke` 426, and `static_verify` 293.

---

## 9. Known evaluation design problems

1. **Thresholds are inconsistent across historical runs**, so verdicts cannot be compared across runs;
2. **A debiasing probe was misjudged as a speedup**, showing that the gate is unreliable under that configuration;
3. **Speedup gains are not quantified** (only 7 records carry paired measurements).

See [LIMITATIONS.md](LIMITATIONS.md). These are **unfixed**, because changing the acceptance gate
would change scientific conclusions and must be done separately after confirmation by the authors.
