# results/

This directory holds only results that are **allowed to be public and worth documenting**, and every number is traceable to its source.

| File | Content | Source |
|---|---|---|
| `noise_floor_calibration.json` | Statistical summary of the zero-effect noise floor calibration | **Recomputed** from `zero_effect_raw.jsonl` (per-query cost of 12 passes) and `zero_effect_report.json` |
| `decoy_probe_finding.json` | Full evidence that a patch which only adds work was misjudged as a speedup | Extracted and recomputed from that evaluation record `evaluations/<eval_id>.json` |

---

## How these numbers were produced

Both files were **recomputed from the raw experimental records during the curation process**, rather than copied from existing reports.
The recomputed results agree with the original records (minor decimal differences are noted below).

**What was not done**: no end-to-end full evaluation was rerun.
The numbers here are therefore a **recomputation of existing experimental records**, not experiments newly produced at this time.
The original experiments were completed on the group's Slurm cluster.

---

## `noise_floor_calibration.json` summary

Zero-effect control: **the same jackhmmer binary A/B'd against itself**, interleaved within the same job.
Any non-zero ratio must be pure measurement noise.

- 12 passes (6 replicates × 2 arms), 25 queries/pass, 8 CPU/query, concurrency 5, `max_iter=1`
- pass cost: min 12826.4 s / median 13176.8 s / max 13801.3 s, **σ = 332.9 s (CV 2.53%)**
- actually measured A/B paired ratios (6): −3.40%, −0.89%, +2.52%, +0.49%, **+4.00%**, −2.81%
- σ of the **gate statistic** (median of R paired ratios):

  | R | 1 | 2 | 3 | 4* | 5* | 6* |
  |---|---:|---:|---:|---:|---:|---:|
  | σ | 2.68% | 1.70% | 1.77% | 1.18% | 0.69% | 0.00% |

  \* R≥4 are extrapolated values, and R=6 has degenerated (only 1 distinct subset). The original report carried this annotation; it is reproduced here.

**Definition warning**: `pair_speedup.n = 132` in the original report is the number of paired ratios obtained by **cross-pass resampling**
(σ=3.54%, 0.035582 in this recomputation), describing a gate that **does not cancel drift**;
the actually measured A/B pairs number only **6**. Both definitions are retained in that JSON and labeled separately.
**Do not describe the 132 as 132 independent experiments.**

---

## `decoy_probe_finding.json` summary

A **workload-only** patch (one extra `get_xE()` call plus a spin of k=1.0 × that duration)
was judged **`accepted_speedup_paired`, median paired speedup +4.04%**.

- That pairing **was indeed interleaved** (not a block-order artifact)
- 5 paired ratios: +23.1% / +6.8% / −13.8% / −10.8% / +4.0% (σ = **14.9%**)
- candidate arm costs: 118.10 / 118.16 / 124.38 / 125.63 / **154.88** s
- control arm costs: **113.43** / 126.77 / 129.62 / 136.06 / **153.74** s

**This is a "known limitation", not a "result"**; see
[docs/LIMITATIONS.md](../docs/LIMITATIONS.md).
Its `what_it_does_not_show` field explicitly lists the conclusions that **must not** be drawn from it.

---

## Recomputation consistency

| Quantity | Recomputed value in this directory | Value stored in the original record |
|---|---|---|
| pass cost σ | 332.923570 | 332.92357 ✅ |
| pooled paired-ratio σ | 0.035582 | 0.035447 |
| paired ratio min / max | −0.076008 / +0.070638 | consistent ✅ |

The only difference is in the 4th decimal of the pooled σ, most likely because the resampling random seed differed,
**and it does not affect any conclusion**.

---

## Artifacts not included, and why

| Artifact | Size | Reason not included |
|---|---|---|
| full `runs/` directory | ~100 GB | size; also contains site information such as cluster absolute paths and job numbers |
| `runs/ledger/all_runs_ledger.csv` | 1.9 MB | 1367 rows containing full cluster paths and job numbers; must be sanitized before it can be published |
| `dry_run_out/` | 438 MB | size and site information |
| build outputs and candidate source trees | several GB | can be rebuilt by script |

To cite finer-grained evidence (for example per-query cost),
**recompute and sanitize** it in the same way before adding it to this directory,
and note the source file and definition of that number in the table.
