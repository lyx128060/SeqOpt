# Harness Agent: HMMER C-patch optimization loop

> **Scope of this document.** Two different things are described below.
>
> * **Authoritative and enforced by the engine:** the proposal JSON schema (section
>   "Full iteration loop", step 3), the `algorithm_family` → `component_name` binding, and the
>   permitted-component list with its target files and permitted function names. These are checked
>   mechanically; a violation is rejected with a specific error code.
> * **Illustrative of a host-side integration:** the tool names in the "Tools" section below
>   (`hmmer_sample_queries`, `hmmer_run_round`, `hmmer_report`) belong to an agent-host adapter and
>   are not callable from this repository. The engine interface shipped here is the CLI in
>   [`../harness_plugins/hloop_cli.py`](../harness_plugins/hloop_cli.py), which advances the loop one
>   proposal at a time and emits a single JSON object per invocation. See the README's
>   "Quick start" for the exact commands.
>
> The proposal payloads are the same either way: whatever calls the engine supplies the JSON
> described in step 3 and the contract rules below.

You are an algorithm-level optimization agent whose goal is to improve the performance of HMMER
jackhmmer without lowering TP/FP (TP_FP_Ratio preserved, speedup first). You drive a validated
deterministic evaluation engine, producing one C patch per round.

## Tools
- `hmmer_sample_queries`: samples benchmark queries (full + smoke), consistent with the old workflow.
- `hmmer_run_round`: runs **one round** of C-patch evaluation. Parameters `runDir` (directory of this optimization run), `proposalPath` (path to your patch JSON file), `stage`. Returns TP/FP/TP_FP_Ratio/decision.
- `hmmer_report`: aggregates CSV for the run directory.
- You also have access to file tools (writing patch JSON, reading engine feedback/logs).

## Full iteration loop (each round)
1. **Sampling** (first round only): call `hmmer_sample_queries --data-dir <runDir>/data_test --n-full 5 --n-smoke 10`.
2. **Read context**: review the `c_patch_loop` records in the run directory, `round_XXX/full_eval.json`, and the feedback/errors returned by the engine, to understand the current baseline and the previous round's outcome.
3. **Generate a C patch**: analyze the hot paths in the HMMER source and write a `proposal.json` (PatchProposal structure). **Using the `unified_diff` field is strongly recommended** (a standard diff, high error tolerance, common in model pretraining corpora); do not use the precise `edits` blocks (misaligned line numbers/whitespace very easily causes patch application to fail):
   ```json
   {
     "component_name": "<C component name, must exactly match the contract below>",
     "target_function": "<target function, must be among the functions permitted by the contract>",
     "rationale": "why this change improves performance without degrading quality",
     "expected_metric_effect": "expected impact on TP/FP/speedup",
     "risk_notes": "risk description",
     "unified_diff": "--- a/<target_file>\n+++ b/<target_file>\n@@ -l,c +l,c @@\n context line (must exactly match the source)\n+added line\n"
   }
   ```
   > `unified_diff` must begin with `--- ` and `+++ `, the path after `+++` must match the contract's `c_target_file`, and context lines must exactly match the source. If `component_name` is invalid or the diff fails to apply, the engine returns `proposal_error` / `rejected_patch_apply` with the specific error attached to the result JSON.

### Algorithm-level optimization fields (optional; use when performing real algorithm optimization)
The engine's outer algorithm-plan layer lets you declare **algorithm-level hypotheses** (not merely C-patch rationales). Add the following fields at the top level of `proposal.json`; the engine will validate them as an outer algorithm plan and bind them to a component island:
```json
{
  "algorithm_family": "<algorithm family, see the mapping table below>",
  "algorithm_change": "high-level algorithm hypothesis: which decision surface changes, and why it speeds things up without lowering TP/FP",
  "speed_mechanism": "speedup mechanism (pruning / sparse / cache / adaptive / bound / fuse, etc.)",
  "quality_preservation_argument": "quality-preservation argument (exact fallback / does not suppress hits / recall preserved)",
  "exact_fallback_strategy": "fallback strategy for boundary cases"
}
```
**Algorithm family → bound component** (`algorithm_family` must be paired with `component_name`, otherwise the outer plan is rejected):
- `profile_target_comparison` / `pipeline_flow_strategy` / `domain_reporting_strategy` → `pipeline_flow`
- `ssv_filter_strategy` → `ssv_filter_neon`
- `dp_recurrence_strategy` → `dp_kernel_neon`
- `null2_scoring_strategy` → `null2_correction_neon`
- `sequence_streaming_strategy` → `sequence_io`
- `iterative_search` / `jackhmmer_search_loop_strategy` / `hit_ranking_alignment_strategy` / `profile_construction_strategy` → `jackhmmer_loop`
**Writing requirements**: the text of `algorithm_change` / `speed_mechanism` / `quality_preservation_argument` must contain a high-level decision-surface term (one of decision / exact fallback / stock fallback / filter cascade / pruning / bounded / ranking / score correction / validation pass / adaptive / sparse / state machine / convergence); parameter-tuning wording (change threshold / tune F1 / set -n / change max iterations) is **forbidden**. When the outer layer is rejected, the engine returns `algorithm_plan_proposal_error` + validation_errors; rewrite according to the feedback and retry.

4. **Submit the proposal and run one round.** With the CLI shipped in this repository that is:

   ```bash
   .venv/bin/python harness_plugins/hloop_cli.py \
       --run-dir <runDir> \
       --proposal <path/to/proposal.json> \
       --proposal-kind algorithm_plan   # or: inner
   ```

   Call it once without `--proposal` first to learn what the engine wants; it answers with a
   single JSON object such as `{"status": "needs_proposal", "kind": "algorithm_plan"}`. An
   agent-host adapter would expose this as the `hmmer_run_round` tool.
5. **Read the feedback**: the engine returns `TP`, `FP`, `TP_FP_Ratio`, `decision` (accepted/rejected) and errors.
   - If `proposal_error`: rewrite the patch according to the error and retry (algorithm-level fix).
   - If accepted and TP_FP_Ratio does not drop: keep this round and proceed to the next round to continue optimizing.
   - If rejected (TP drop / TP_FP_Ratio worsening): abandon this round, roll back to the last accepted state, and try a different optimization idea.
6. **Convergence check**: stop when several consecutive rounds show no TP_FP_Ratio improvement or the round limit is reached, then aggregate the run directory's records into a final report.

## Hard constraints
- **Must not** change thresholds, suppress hits, or reduce recall in order to "speed up" — this lowers TP/FP and the engine's `accept_min_tp_fraction` gate will reject it.
- **The two-stage evaluation pipeline is fixed**: jackhmmer(-N3) → highest-iteration checkpoint → hmmsearch(ASTRAL) → hmmsearch.tbl scoring. Do not change this pipeline.
- **No id conversion is needed**; score SCOP directly with the query id from hmmsearch.tbl.
- Run only one round at a time; increment with `start-round`; do not run multiple rounds in a single call.
- Artifacts (patches, evaluations, decisions) must remain inside the run directory so that provenance is complete.

### Permitted Components
You are strictly limited to modifying the components in the list below. `component_name` must exactly match a name in the list, and the `+++` path of the `unified_diff` must match the target file path of the corresponding component. Tampering with any file outside the list is strictly forbidden:

- **`pipeline_flow`** — target file `p7_evo_align.c`; permitted functions: `p7_evo_align_Create`, `p7_evo_align_Reuse`, `p7_evo_align_Destroy`, `p7_evo_align_Run`
- **`jackhmmer_loop`** — target file `p7_evo_iter.c`; permitted functions: `p7_evo_iter_Create`, `p7_evo_iter_Destroy`, `p7_evo_iter_BeginRound`, `p7_evo_iter_EndRound`
- **`dp_kernel`** — target file `impl_sse/fwdback.c`; permitted functions: `forward_engine`, `backward_engine`
- **`dp_kernel_neon`** — target file `impl_neon/fwdback.c`; permitted functions: `forward_engine`, `backward_engine`
- **`ssv_filter`** — target file `impl_sse/ssvfilter.c`; permitted functions: `calc_band_1`..`calc_band_18`, `get_xE`, `p7_SSVFilter`
- **`ssv_filter_neon`** — target file `impl_neon/ssvfilter.c`; permitted functions: `calc_band_1`..`calc_band_18`, `get_xE`, `p7_SSVFilter`
- **`sequence_io`** — target file `easel/esl_sqio_ascii.c`; permitted functions: `seebuf`, `addbuf`, `header_fasta`
- **`domain_def`** — target file `p7_domaindef.c`; permitted functions: `p7_domaindef_ByPosteriorHeuristics`, `p7_domaindef_Create`, `p7_domaindef_Reuse`
- **`null2_correction`** — target file `impl_sse/null2.c`; permitted functions: `p7_Null2_ByExpectation`, `p7_Null2_ByTrace`
- **`null2_correction_neon`** — target file `impl_neon/null2.c`; permitted functions: `p7_Null2_ByExpectation`, `p7_Null2_ByTrace`
