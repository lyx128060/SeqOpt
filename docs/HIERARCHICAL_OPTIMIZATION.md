# Hierarchical Full Optimization

The production loop runs both LLM roles and all controller logic on the login
host. Slurm is used only to build a candidate and evaluate the fixed Full
UniRef90 query set.

```bash
.venv/bin/python harness_plugins/hloop_cli.py \
  --run-dir runs/seq_align_hierarchical \
  --outer-rounds 10 \
  --inner-trials 10
```

The engine reads its configuration from the environment. Start from the
credential-free `configs/config.example.yaml` (copy it to
`configs/config.local.yaml`, which is gitignored) and export the LLM and
benchmark data variables listed in the README before running. Use
`SEQ_ALIGN_CONFIG` only when selecting an uncommitted site-specific
configuration.

Each outer round freezes one high-level algorithm plan in exactly one semantic
family and its active source contract:

- `profile_target_comparison` binds to `pipeline_flow` (`p7_evo_align.c`).
- `iterative_search` binds to `jackhmmer_loop` (`p7_evo_iter.c`).
- `dp_recurrence_strategy` binds to `dp_kernel_neon` (`impl_neon/fwdback.c`).
- `ssv_filter_strategy` binds to `ssv_filter_neon` (`impl_neon/ssvfilter.c`).
- `null2_scoring_strategy` binds to `null2_correction_neon` (`impl_neon/null2.c`).
- `sequence_streaming_strategy` binds to `sequence_io` (`easel/esl_sqio_ascii.c`).

The production cluster is AArch64, so new proposals expose the NEON contracts.
The older SSE contracts remain registered only to replay and audit historical
content-addressed patchsets; new plans cannot select them.

The outer proposal contains no source binding or patch. The local controller
chooses the corresponding island and freezes its public interface. Ten inner
trials then produce complete C patches sequentially. A quality-valid, strictly
faster trial becomes the next trial's provisional parent; failed or regressing
trials remain trajectory evidence but are never inherited.

Every inner trial runs the ten fixed queries as a Slurm array. A candidate is
built once with 4 CPU, 8 GiB, and 30 minutes, then all query tasks reuse the
content-addressed binary bundle. Each query requests 1 CPU, 16 GiB, and 12
hours, with at most ten concurrent tasks. Its runtime limit is fixed at 1.2
times the same query's outer-parent runtime for the complete inner round.

After all ten inner trials, the provisional winner is committed only when TP
does not decrease, FP does not increase, and aggregate runtime improves by at
least 3% against the frozen outer parent.

Canonical records live under `RUN_DIR/c_patch_loop/`:

- `algorithm_plans/`, `implementation_trials/`, `patch_plans/`, `patchsets/`
- `evaluations/`, `slot_events/`, `invocations/`
- `experience/` for raw trajectory and compressed older experience
- `indexes/hierarchical_run_summary.json` for the current projection

Re-running the identical command and run directory resumes the same invocation
and reattaches to matching submitted Slurm jobs instead of proposing or
submitting them again.
