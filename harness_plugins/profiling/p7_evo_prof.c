/* Accumulator and reporter for the SeqAlign profiling build.
 *
 * Writes the p7_evo_prof.txt contract consumed by
 * workspace/seq_align/hmmer_stage_timing.py (INSTRUMENTED_FILE / parse_instrumented_stage_times).
 *
 * Deliberately writes to a FILE named by $P7_EVO_PROF_OUT rather than to stdout or
 * stderr: the eval path parses HMMER's stdout and .tbl outputs, and polluting
 * either would corrupt real results. If the env var is unset nothing is written.
 *
 * Compile only into the profiling build (-DP7_EVO_PROF); see README.md.
 */

#ifdef P7_EVO_PROF

#include "p7_evo_prof.h"

double p7_evo_prof_seconds[P7_EVO_PROF_NSTAGES] = {0.0, 0.0, 0.0, 0.0, 0.0};
long   p7_evo_prof_calls[P7_EVO_PROF_NSTAGES]   = {0, 0, 0, 0, 0};

/* Names must match the contract keys in hmmer_stage_timing.PIPELINE_STAGES. */
static const char *p7_evo_prof_names[P7_EVO_PROF_NSTAGES] = {
  "ssv_filter",
  "dp_kernel",
  "null2_correction",
  "domain_def",
  "pipeline_flow"
};

void
p7_evo_prof_add(int slot, double seconds)
{
  if (slot < 0 || slot >= P7_EVO_PROF_NSTAGES) return;
  if (seconds < 0.0) return;   /* clock went backwards; ignore rather than corrupt */
  p7_evo_prof_seconds[slot] += seconds;
  p7_evo_prof_calls[slot]   += 1;
}

void
p7_evo_prof_report(const char *path)
{
  FILE *fp;
  int   i;

  if (path == NULL || path[0] == '\0') return;
  fp = fopen(path, "w");
  if (fp == NULL) return;

  fputs("# p7_evo_prof.v1\n", fp);
  fputs("# timer clock_gettime(CLOCK_MONOTONIC); unit seconds; time inside the stage\n", fp);
  fputs("# note profiling build (-DP7_EVO_PROF); timers add overhead; do not compare to cost_s\n", fp);
  for (i = 0; i < P7_EVO_PROF_NSTAGES; i++)
    fprintf(fp, "%s %.6f %ld\n", p7_evo_prof_names[i], p7_evo_prof_seconds[i], p7_evo_prof_calls[i]);

  fclose(fp);
}

void
p7_evo_prof_install_atexit(void)
{
  atexit(p7_evo_prof_report_from_env);
}

void
p7_evo_prof_report_from_env(void)
{
  p7_evo_prof_report(getenv("P7_EVO_PROF_OUT"));
}

/* Self-registering on load. This matters: requiring an explicit call in
 * jackhmmer's main() would mean injecting a line into a non-contract file, and
 * that line would break any build where the shim is not compiled in. Registering
 * here keeps instrumentation confined to the contract functions it wraps. */
#if defined(__GNUC__) || defined(__clang__)
__attribute__((constructor))
static void p7_evo_prof_auto_install(void)
{
  p7_evo_prof_install_atexit();
}
#endif

#endif /* P7_EVO_PROF */
