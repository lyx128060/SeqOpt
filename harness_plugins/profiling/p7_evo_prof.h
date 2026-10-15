/* Per-stage timing for the SeqAlign profiling build.
 *
 * Compiled in only when P7_EVO_PROF is defined. Without it every macro is a
 * no-op, so this header can sit in a source tree harmlessly and the benchmark
 * binary is bit-for-bit unaffected -- which matters, because cost_s is the
 * acceptance gate and must stay comparable across candidate builds.
 *
 * Why in-band timers at all: HMMER reports per-iteration elapsed time and
 * per-filter *pass counts*, but never per-stage *times*. Splitting the time by
 * stage is the only way to tell an agent whether it should be attacking the
 * filter cascade or the DP kernel.
 *
 * Timer overhead: two clock_gettime calls per stage entry. Inside p7_SSVFilter
 * that is once per target sequence, so the cost is real but small relative to the
 * filter itself. Report it honestly: these numbers are for attribution, not for
 * acceptance. Never compare them against cost_s.
 */
#ifndef p7_evo_prof_h
#define p7_evo_prof_h

#ifdef P7_EVO_PROF

/* struct timespec / clock_gettime / CLOCK_MONOTONIC are POSIX, and a strict
 * -std=c99 (which defines __STRICT_ANSI__) hides them. Request the POSIX.1-1993
 * baseline BEFORE any header is pulled in -- this header must therefore be
 * included first in an instrumented translation unit. */
#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 199309L
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* Fail loudly rather than silently measuring nothing: if CLOCK_MONOTONIC is
 * still unavailable, some earlier header already fixed the feature set, and a
 * "timing run" would report zeros. A build error is the honest outcome. */
#if !defined(CLOCK_MONOTONIC)
#error "P7_EVO_PROF requires POSIX clock_gettime/CLOCK_MONOTONIC. Build with -std=gnu99 (or later), or define _POSIX_C_SOURCE=199309L before any other include, or include p7_evo_prof.h first."
#endif

/* Slot order must match PIPELINE_STAGES in
 * workspace/seq_align/hmmer_stage_timing.py. The names written to the report
 * file are the component contract names from components/c_kernels.py, because
 * those are the only names an agent is allowed to target. */
enum p7_evo_prof_slot {
  P7_EVO_PROF_SSV_FILTER      = 0,
  P7_EVO_PROF_DP_KERNEL       = 1,
  P7_EVO_PROF_NULL2           = 2,
  P7_EVO_PROF_DOMAIN_DEF      = 3,
  P7_EVO_PROF_PIPELINE_FLOW   = 4,
  P7_EVO_PROF_NSTAGES         = 5
};

extern double p7_evo_prof_seconds[P7_EVO_PROF_NSTAGES];
extern long   p7_evo_prof_calls[P7_EVO_PROF_NSTAGES];

void   p7_evo_prof_add(int slot, double seconds);
void   p7_evo_prof_report(const char *path);
void   p7_evo_prof_report_from_env(void);
void   p7_evo_prof_install_atexit(void);

static inline double p7_evo_prof_now(void)
{
  struct timespec ts;
#if defined(CLOCK_MONOTONIC)
  clock_gettime(CLOCK_MONOTONIC, &ts);
#else
  ts.tv_sec  = time(NULL);
  ts.tv_nsec = 0;
#endif
  return (double) ts.tv_sec + (double) ts.tv_nsec * 1e-9;
}

/* Declares the start timestamp in the enclosing block. Requires C99 (mid-block
 * declarations), which HMMER 3.4 already relies on. */
#define P7_EVO_PROF_ENTER(slot_var) \
  const double slot_var = p7_evo_prof_now()

/* Call immediately before every `return` in the instrumented function. */
#define P7_EVO_PROF_EXIT(slot_id, slot_var) \
  p7_evo_prof_add((slot_id), p7_evo_prof_now() - (slot_var))

#else /* !P7_EVO_PROF */

#define P7_EVO_PROF_ENTER(slot_var)         ((void) 0)
#define P7_EVO_PROF_EXIT(slot_id, slot_var) ((void) 0)

#endif /* P7_EVO_PROF */
#endif /* p7_evo_prof_h */
