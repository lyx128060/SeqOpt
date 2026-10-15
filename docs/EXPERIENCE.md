# C-patch loop experience management

The C-patch optimizer treats history as evidence, not as free-form chat memory. The model should see a small, stable context pack; the run directory keeps the full audit trail.

## Layers

1. **Immutable raw evidence**
   - `context_packs/*.json`: what the model was allowed to see.
   - `patch_plans/*.json`: validated algorithm plan before diff generation.
   - `patchsets/*.json`: exact C diff, target contract, source hashes, provider metadata hashes.
   - `evaluations/*.json`: static/build/smoke/full result, dataset manifest, build/runtime provenance, metric deltas.
   - `experience/experience.jsonl`: one evidence-linked lesson per evaluated outcome.

2. **Derived indexes**
   - `indexes/current_speed_best[_stage].json`: speed-first incumbent under TP/FP gates.
   - `indexes/accepted_lineage[_stage].json`: accepted baseline/speedup lineage.
   - `indexes/leaderboard[_stage].json`: speed-sorted accepted candidates.
   - `indexes/experience_component_stats.json`: component outcome counters and recent experience IDs.
   - `indexes/experience_workflow_failures.json`: repair/build/apply/provenance lessons.
   - `indexes/experience_accepted_speed_lineage.json`: accepted quality-preserving speed patterns.

3. **Model-facing compression**
   - `experience/compressed_summary.md` is regenerated from raw JSONL.
   - It carries source `eval_id` / `patchset_id` / `diff_sha256` so summaries stay auditable.
   - It separates algorithm lessons from workflow repair lessons.

## Taxonomy

- Algorithm lessons: accepted speedups, rejected-not-faster candidates, TP/FP quality regressions, neutral metric outcomes.
- Workflow lessons: patch apply failures, static contract violations, compile failures, runtime failures, runtime provenance failures.

Workflow lessons should guide immediate repair of the diff/build/provenance mechanics. They should not be interpreted as evidence that an algorithmic idea is bad.

## Retrieval policy for context packs

Each new round should include:

1. Stock quality reference and current speed best from indexes.
2. Recent two or three attempts from raw evaluations/patchsets.
3. Compressed component-level lessons relevant to the allowed C contracts.
4. Workflow repair lessons only as constraints for producing applyable/runtime-effective diffs.

Do not include the full history in the prompt. If a decision needs deeper audit, inspect the referenced raw `eval_id` / `patchset_id` records outside the model proposal step.

## Acceptance objective

The historical store is aligned with the current speed-first objective:

- Hard gate: `TP >= quality_reference.TP * accept_min_tp_fraction`.
- Hard gate: `FP <= quality_reference.FP`.
- Speed gate: `candidate.cost_s < current_speed_best.cost_s * (1 - min_speedup_fraction)`.

Only candidates that pass the quality gates and speed gate enter accepted speed lineage. Other runnable candidates are still recorded so the compressor can learn which directions were quality-preserving but too slow.
