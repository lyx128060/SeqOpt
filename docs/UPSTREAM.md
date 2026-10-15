# Relationship to the Upstream Project

This repository is an **improved version of an existing project**, not a from-scratch development.
This document states its origins, the contribution boundary, and the license obligations truthfully.

---

## 1. Lineage

```
sjtu-sai-agents/EvoMaster                     ← upstream framework (Apache-2.0)
│   arXiv:2604.17406
│   "A Foundational Auto-Research Agent Framework for Agentic Science at Scale"
│
└── lyx128060/EvoMaster-Jackhmmer             ← fork (development repository for this project)
        │  initial upload commit 76ed75e (2026-07-18)
        │
        ├── origin/main   @ e4f175e            ← pushed
        └── opt           @ 8bda906            ← ahead by 9 commits (contains all of this repository's content)
```

**Evidence**: `git remote -v`, `git merge-base origin/main HEAD = e4f175e`,
and `README_Original_EvoMaster.md` pointing to `github.com/sjtu-sai-agents/EvoMaster`.

---

## 2. Contribution boundary

| Part | Attribution | Included in this repository |
|---|---|---|
| `evomaster/` agent framework proper | Upstream EvoMaster team | ❌ not included, used as an external dependency |
| `playground/`, `extensions/` extensions | Upstream EvoMaster team | ❌ not included |
| SeqOpt core library `workspace/seq_align/` | **Author of this project** | ✅ included |
| C patch optimization loop `c_patch_loop/` | **Author of this project** | ✅ included |
| Evaluation and verification machinery | **Author of this project** | ✅ included |
| HMMER 3.4 proper | Sean R. Eddy et al. (BSD-3-Clause) | ❌ not included, fetched by script |
| HMMER optimization-island interface patch | **Author of this project** | ✅ included as a patch |

### Why upstream framework code is **not** included

1. Upstream already has its own repository; copying it again serves no purpose and would create
   the burden of propagating the Apache-2.0 NOTICE;
2. The development work tree is about 100 GB, 99.9% of which is framework code and experiment artifacts identical to upstream;
3. The application context calls for **a clear presentation of my own contributions**, not another release of the upstream framework.

**This repository is therefore positioned as a "personal contribution showcase version", not a redistribution of the upstream framework.**

---

## 3. List of my changes (relative to upstream)

Based on git author records (the three identities `lyx128060` / `daqige` / `TerminalBench`
are confirmed by the author to belong to **the same person**):

| Stage | Period | Main content |
|---|---|---|
| Low-memory evaluation pipeline | 2026-07-18 → 07-27 | pyhmmer dual backends, per-query process isolation, UniRef90 peak held below 64 GB, SCOP mapping fixes, removal of hard-coded paths |
| C patch optimization loop | 2026-07-28 → 08-31 | the full `c_patch_loop/` suite, HMMER optimization-island interface, staged optimization workflow, profiling-guided context |
| Measurement verification machinery | same period | comparability fingerprints, paired interleaved re-measurement, zero-effect calibration, debiasing probe, experiment ledger |
| Engineering clean-up | this release | extraction into a self-contained repository, generation of a reproducible HMMER patch, precise test skip conditions, documentation |

The roughly **290** upstream EvoMaster commits from 2026-01 → 2026-05 are **not counted** toward my contributions.

---

## 4. License obligations

| Component | License | Obligation for this repository | How it is met |
|---|---|---|---|
| Upstream EvoMaster | Apache-2.0 | Retain notices, mark modifications, propagate NOTICE | `LICENSE`, `NOTICE`, §3 of this document |
| This repository's own code | **Apache-2.0** | — | `LICENSE` |
| HMMER 3.4 | **BSD-3-Clause** | Retain the copyright and disclaimer; do not use the authors' names for endorsement | `patches/HMMER-3.4-LICENSE` distributed with the patch; `NOTICE` §2 |
| libdivsufsort (inside HMMER) | see the HMMER distribution | Retain notices | obtained with the original HMMER tarball |

> ⚠️ **The HMMER BSD-3 notice must be retained with the patch**.
> This is a hard obligation and is not waived by "reorganizing".

---

## 5. Dataset attribution

| Data | Attribution | Handling in this repository |
|---|---|---|
| UniRef90 | UniProt Consortium | Not included, download instructions only |
| ASTRAL 40% subset | SCOPe | Not included; only a 112 KB smoke query subset is included |
| SCOPe lineage file | SCOPe | Not included |

---

## 6. Note on development tools

Parts of the code were completed with the assistance of an AI coding assistant (some commits carry `Co-Authored-By: Claude`).
The code and all scientific conclusions were reviewed by the author, who is responsible for their correctness.
The repository's design of separating "measured facts" from "model assumptions" (see
[VERIFICATION.md §3](VERIFICATION.md)) stems in part from this.
