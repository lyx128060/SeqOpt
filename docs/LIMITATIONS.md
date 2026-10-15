# Known Limitations

This document lists only limitations **for which there is evidence**. Each entry gives the location of the evidence and what it does **not** support.
Items that cannot be confirmed are marked "to be confirmed"; no speculation is made.

---

## 1. A confirmed measurement problem: the decoy probe was misjudged as a speedup

### 1.1 Facts

Source: `results/decoy_probe_finding.json`

| Item | Value |
|---|---|
| Probe file / function | `impl_neon/ssvfilter.c` / `p7_SSVFilter` |
| What the probe does | Runs `get_xE()` one extra time, then spins for k=1.0 × that duration; return values and all score vectors unchanged |
| Why it is a valid negative control | It **strictly increases** the work in the timed region, so under any correct measurement it can only be slower; the patch's own rationale also states that it "only changes stage durations" |
| Decision | **`accepted_speedup_paired`**, median paired speedup **+4.04%** |
| Was it interleaved | **Yes** (candidate/control really alternated, not an artifact of block ordering) |
| The 5 paired ratios | +23.1% / +6.8% / −13.8% / −10.8% / +4.0%, σ = **14.9%** |
| Candidate arm costs | 118.10 / 118.16 / 124.38 / 125.63 / **154.88** s |
| Control arm costs | **113.43** / 126.77 / 129.62 / 136.06 / **153.74** s |
| `min_speedup_fraction` for that run | **0.0** (any positive speedup passes the gate) |

### 1.2 Conclusions it **supports**

1. Under **that configuration**, the paired confirmation mechanism **failed** to block a patch that is deterministically slower;
2. The within-arm fluctuation of that measurement (interval span about 25–31%) is **far larger** than
   the noise floor derived from the 25-query calibration experiment (2.68% at R=1) — that is, **the calibration cannot be extrapolated directly to that configuration**;
3. `min_speedup_fraction = 0.0` lets "any positive median" pass the gate,
   and in that case the decision is almost entirely determined by noise.

### 1.3 Conclusions it does **not** support (must not be written this way)

- ❌ "The paired confirmation mechanism is ineffective" — the mechanism **did run**, and did produce the
  paired decision it was designed to produce; the problem is that the noise and the threshold did not match in that configuration.
- ❌ "This method improves accuracy by X%" — this repository has **no** accuracy measurement at all.
- ❌ "The incidence rate of such misjudgements is Y%" — there is only 1 record, so no frequency can be inferred.
- ❌ "That patch really was 4% faster" — on the contrary, it can only be slower.

---

## 2. Items not yet quantified

### 2.1 End-to-end speedup benefit is **not** quantified

The experiment ledger contains only **7** records with paired speedup measurements, of which 2 are
`accepted_speedup_paired` (+2.99%, +4.04%), and the latter has been shown by §1 to be a misjudgement.

**Conclusion**: the sample size is insufficient to support any statement that "this method reliably yields an N% speedup".
Any specific percentage should be regarded as unproven.

### 2.2 Quality preservation has **not** been independently verified

TP / FP are produced by the **same** SCOPe lineage scoring pipeline. Confirming the output of a pipeline with that same pipeline
is **self-attestation from the same source**, not independent scientific validation.

**Conclusion**: one may say "under the established scoring convention, TP did not decrease and FP did not increase",
but **not** "quality preservation has been independently verified".

### 2.3 Inconsistent thresholds across historical decisions

Different runs used different `accept_min_tp_fraction` and `min_speedup_fraction`
(0.0 for the probe run). Therefore **decisions from historical runs cannot be compared directly with one another**.

`comparability.py` was written precisely to solve this class of problem, but the historical records predate the module and
**cannot be retroactively corrected**.

---

## 3. Engineering limitations

| Limitation | Impact | Mitigation |
|---|---|---|
| HMMER source tree not bundled with the repository | 37 tests **skip** by default | `scripts/setup_hmmer.sh` fetches and patches it in one step; the reason for skipping is printed |
| Full evaluation depends on the group's Slurm + 60.9 GB UniRef90 + a private LLM endpoint | Hard to reproduce end to end externally | README §10.3 states this explicitly; the offline part can be accepted independently |
| `data_loader.py` and `dataloader.py` overlap in responsibility | Easy to confuse | To be confirmed: only one is the current path; they were not merged in this cleanup (it would touch scientific logic) |
| `evaluate.py` and `scorer.py` overlap in responsibility | Same as above | Same as above |
| `query_manifest.json` inside `data_test/` still records absolute paths from the original machine | The file is metadata only and does not affect computation | Not read by any test assertion; it is regenerated when the query set is rebuilt |

---

## 4. What this cleanup did **not** do (honest boundaries)

1. It did **not** re-run the end-to-end full evaluation. The numbers in §1 and in `results/` are
   **recomputations of existing experiment records**, not experiments newly produced here.
2. It did **not** change any scientific logic (thresholds, scoring, pipeline). The engineering cleanup covered only
   directory placement, dependency declarations, test skip conditions, and documentation.
3. It did **not** fix the measurement problem in §1. Fixing it requires redoing the calibration and re-validating,
   which is a scientific decision; it is listed as a direction for improvement rather than implemented unilaterally.
4. It did **not** retroactively add provenance fields to historical runs.

---

## 5. Directions for improvement (in priority order)

1. **Redo the noise floor calibration on the same configuration as the decision** (50-query Full, same backend, same core count,
   same database), and write the **within-arm cost span** into the evaluation record,
   so that "measurements with excessive fluctuation" can be flagged at decision time rather than discovered only afterwards.
2. Raise `paired_current_best_repeats` and `min_speedup_fraction` above the measured noise
   (forbid degenerate values such as 0.0), then **regression-test with the same decoy probe**,
   confirming that it blocks the probe this time.
3. Re-check the quality conclusion through an **independent path** (a different lineage ground truth or an independent tool),
   so that "quality has not degraded" no longer relies on a single scoring pipeline attesting to itself.
4. Enlarge the sample of paired speedup measurements (currently 7); only then can a conclusion about the method's effectiveness be drawn.
5. Merge the duplicate modules in §3, after first confirming which one is the current production path.

---

## 6. To be confirmed

- Why the thresholds were inconsistent across historical runs: configuration drift, or gates deliberately set differently for different experiments?
  This can be confirmed only by the author; **no speculation** is made.
- The three identities `daqige` / `TerminalBench` / `lyx128060` have been confirmed by the author to belong to the same person,
  so this document treats them as a single author.
