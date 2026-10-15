"""Per-stage timing evidence for the HMMER acceleration pipeline.

Why this module exists
----------------------
``cost_s`` is a single wall-clock number covering jackhmmer + hmmsearch. That is
enough to decide *whether* a candidate is faster, but it tells an optimizing agent
nothing about *where* the time goes. This module produces that missing evidence.

Two independent sources, deliberately kept separate
---------------------------------------------------
``output_parse`` (non-invasive)
    HMMER already reports, **per iteration**, an elapsed time, a throughput and
    the per-filter pass counts. ``p7_pli_Statistics()`` is called from inside
    jackhmmer's iteration loop and ends with
    ``esl_stopwatch_Display(ofp, w, "# CPU time: ")`` plus ``# Mc/sec:``. So
    "which iteration dominates" and "which filter stage culls the work" need no C
    change, no extra run, and no timing overhead.

``instrumented_build`` (opt-in)
    HMMER reports per-stage *counts*, never per-stage *times*. Splitting the time
    by stage requires a dedicated profiling build with timers at the stage
    boundaries. Those numbers are only present when such a build ran.

**Never fabricate stage times from the parse layer.** When only the parse layer is
available, a stage entry carries its counts and no ``time_s`` key at all. A wrong
number is worse than a missing one: the agent will act on it.

Timer semantics (a real trap)
----------------------------
``esl_stopwatch``'s printed ``# CPU time:`` is ``user+sys``, while jackhmmer runs
``--cpu N`` worker threads, so CPU time can exceed wall time and neither equals
``cost_s``. The line also carries ``Elapsed:`` which *is* wall clock. Both are
reported, explicitly labelled, so the two are never conflated.

Called from the worker threads, ``user``/``sys`` include child CPU
(``tms_cutime``/``tms_cstime``); treat them as machine-wide work, not per-thread.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "hmmer_stage_timing.v1"

# --------------------------------------------------------------------------
# Contract <-> pipeline stage mapping.
#
# JSON keys are the *component contract* names from
# ``workspace/seq_align/components/c_kernels.py``, because those are the only
# things an agent is allowed to modify (``assembler.verify_c_patch``). The real
# HMMER stage name is carried alongside as a description so the two vocabularies
# never drift apart silently.
# --------------------------------------------------------------------------
PIPELINE_STAGES: tuple[dict[str, Any], ...] = (
    {
        "contract": "ssv_filter",
        "pipeline_stage": "MSV filter",
        # HMMERHEAD's first two levels (word generation + ungapped extension) are
        # fused into the modern MSV/SSV filter; this is the paper's stage 1-2.
        "hmmer_counters": ("n_past_msv",),
        "note": "HMMERHEAD stages 1-2 (word seed + ungapped extension) fused into the MSV/SSV filter.",
    },
    {
        "contract": "dp_kernel",
        "pipeline_stage": "Viterbi and Forward filters",
        # HMMERHEAD stages 3-4: gapped Viterbi, then the full scoring algorithm.
        "hmmer_counters": ("n_past_vit", "n_past_fwd"),
        "note": "HMMERHEAD stages 3-4: gapped Viterbi filter, then full Forward/Viterbi scoring.",
    },
    {
        "contract": "null2_correction",
        "pipeline_stage": "biased composition score correction",
        "hmmer_counters": ("n_past_bias",),
        "note": "Composition-bias filter / null2 correction.",
    },
    {
        "contract": "domain_def",
        "pipeline_stage": "domain definition and per-domain reporting",
        "hmmer_counters": (),
        "note": "Postprocessing: domain boundaries, per-domain scores.",
    },
    {
        "contract": "pipeline_flow",
        "pipeline_stage": "per-iteration jackhmmer driver",
        "hmmer_counters": (),
        "note": "Iteration driver: model rebuild, merge, convergence test.",
    },
)

CONTRACT_ORDER: tuple[str, ...] = tuple(stage["contract"] for stage in PIPELINE_STAGES)

STAGE_BY_CONTRACT: dict[str, dict[str, Any]] = {stage["contract"]: stage for stage in PIPELINE_STAGES}

# HMMER's counter name -> the contract that owns that stage.
_COUNTER_TO_CONTRACT: dict[str, str] = {
    counter: stage["contract"]
    for stage in PIPELINE_STAGES
    for counter in stage["hmmer_counters"]
}

# --------------------------------------------------------------------------
# Regexes. Each mirrors an exact printf format string in HMMER 3.4 so a format
# change surfaces as a missing field rather than a silently wrong number.
# --------------------------------------------------------------------------
_ROUND_RE = re.compile(r"^@@\s*Round:\s*(\d+)")
_MSA_RE = re.compile(
    r"^@@\s*Included in MSA:\s*(\d+)\s+subsequences\s+\(query \+ (\d+) subseqs from (\d+) targets\)"
)
_MODEL_SIZE_RE = re.compile(r"^@@\s*Model size:\s*(\d+)\s+positions")
_NEW_TARGETS_RE = re.compile(r"^@@\s*New targets included:\s*(\d+)")
_NEW_ALIGN_RE = re.compile(
    r"^@@\s*New alignment includes:\s*(\d+)\s+subseqs\s+\(was (\d+)\)"
)
_CONVERGED_RE = re.compile(r"^@@\s*CONVERGED\s+\(in (\d+) rounds\)")
_QUERY_MODELS_RE = re.compile(r"^Query model\(s\):\s*(\d+)\s+\((\d+)\s+nodes\)")
_TARGET_SEQS_RE = re.compile(r"^Target sequences:\s*(\d+)\s+\((\d+)\s+residues searched\)")
# "Passed MSV filter:  <count>  (<frac>); expected <exp> (<expfrac>)"
_PASSED_RE = re.compile(
    r"^Passed (?P<stage>[A-Za-z]+) filter:\s*(?P<count>\d+)\s+\((?P<frac>[^)]*)\);"
    r"\s*expected\s+(?P<exp>[^ ]+)\s+\((?P<expfrac>[^)]*)\)"
)
_SEARCH_SPACE_RE = re.compile(r"^Initial search space \(Z\):\s*([\d.]+)")
_DOM_SPACE_RE = re.compile(r"^Domain search space\s+\(domZ\):\s*([\d.]+)")
# "# CPU time: 12.34u 0.56s 00:00:12.90  Elapsed: 00:00:12.90"
# The "0.56s" segment is absent when easel was built without HAVE_TIMES.
_CPU_TIME_RE = re.compile(
    r"^#\s*CPU time:\s*(?P<user>[\d.]+)u\s+(?:(?P<sys>[\d.]+)s\s+)?"
    r"(?P<cpu_hms>[\d:.]+)\s+Elapsed:\s*(?P<elapsed_hms>[\d:.]+)"
)
_MC_PER_SEC_RE = re.compile(r"^#\s*Mc/sec:\s*([\d.]+)")
_PIPELINE_HEADER_RE = re.compile(r"^Internal pipeline statistics summary:")

# easel format_time_string(..., do_frac=TRUE) -> "HH:MM:SS.cc"
_HMS_RE = re.compile(r"^(?:(\d+):)?(\d+):(\d+(?:\.\d+)?)$")


def _parse_hms(token: str) -> float | None:
    """Parse easel's ``HH:MM:SS.cc`` into seconds."""
    match = _HMS_RE.match(token.strip())
    if match is None:
        return None
    hours = int(match.group(1) or 0)
    minutes = int(match.group(2))
    seconds = float(match.group(3))
    return round(hours * 3600 + minutes * 60 + seconds, 4)


def _empty_record() -> dict[str, Any]:
    return {
        "round": None,
        "model_positions": None,
        "msa_subsequences": None,
        "msa_from_targets": None,
        "counts": {},
        "expected": {},
        "search_space": {},
        "timing": {},
        "convergence": {},
        "has_pipeline_block": False,
    }


def _finalise(record: dict[str, Any]) -> dict[str, Any]:
    """Add derived fields; drop empty sub-dicts so the JSON stays readable."""
    counts = record.get("counts") or {}
    if counts:
        # Filter-cascade attrition: the paper's core mechanism. ``nseqs`` is the
        # denominator that entered stage 1.
        nseqs = (record.get("search_space") or {}).get("target_sequences")
        cascade: dict[str, Any] = {}
        previous: int | None = None
        for counter in ("n_past_msv", "n_past_bias", "n_past_vit", "n_past_fwd"):
            value = counts.get(counter)
            if value is None:
                continue
            entry: dict[str, Any] = {"count": value}
            if nseqs:
                entry["fraction_of_targets"] = round(value / nseqs, 8)
            if previous is not None and previous > 0:
                entry["kept_from_previous_stage"] = round(value / previous, 8)
            cascade[counter] = entry
            previous = value
        if cascade:
            record["filter_cascade"] = cascade
    for key in ("counts", "expected", "search_space", "timing", "convergence"):
        if isinstance(record.get(key), dict) and not record[key]:
            record.pop(key)
    for key in ("model_positions", "msa_subsequences", "msa_from_targets"):
        if record.get(key) is None:
            record.pop(key)
    return record


def parse_jackhmmer_output(text: str) -> list[dict[str, Any]]:
    """Parse per-iteration evidence from a ``jackhmmer -o`` file.

    Records are keyed off ``@@ Round:`` markers, because the pipeline statistics
    block that follows belongs to that iteration.
    """
    records: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    pending_round: int | None = None

    for raw_line in text.splitlines():
        line = raw_line.rstrip()

        match = _ROUND_RE.match(line)
        if match:
            # A new round marker must start a NEW record immediately. Creating the
            # record lazily on the first following field would let round N+1
            # overwrite round N's fields, since `current` is still the old record.
            pending_round = int(match.group(1))
            current = _empty_record()
            current["round"] = pending_round
            records.append(current)
            continue

        # A stats block without a preceding Round marker (jackhmmer prints one in
        # its single-query path too) still belongs to the latest iteration.
        if _PIPELINE_HEADER_RE.match(line):
            if current is None or current.get("has_pipeline_block"):
                current = _empty_record()
                current["round"] = pending_round
                records.append(current)
            current["has_pipeline_block"] = True
            continue

        if current is None:
            continue

        match = _MSA_RE.match(line)
        if match:
            current["msa_subsequences"] = int(match.group(1))
            current["msa_from_targets"] = int(match.group(3))
            continue

        match = _MODEL_SIZE_RE.match(line)
        if match:
            current["model_positions"] = int(match.group(1))
            continue

        match = _NEW_TARGETS_RE.match(line)
        if match:
            current.setdefault("convergence", {})["new_targets_included"] = int(match.group(1))
            continue

        match = _NEW_ALIGN_RE.match(line)
        if match:
            current.setdefault("convergence", {})["alignment_subsequences"] = int(match.group(1))
            current["convergence"]["alignment_subsequences_before"] = int(match.group(2))
            continue

        match = _CONVERGED_RE.match(line)
        if match:
            current.setdefault("convergence", {})["converged_at_round"] = int(match.group(1))
            continue

        match = _QUERY_MODELS_RE.match(line)
        if match:
            spaces = current.setdefault("search_space", {})
            spaces["query_models"] = int(match.group(1))
            spaces["model_nodes"] = int(match.group(2))
            continue

        match = _TARGET_SEQS_RE.match(line)
        if match:
            spaces = current.setdefault("search_space", {})
            spaces["target_sequences"] = int(match.group(1))
            spaces["residues_searched"] = int(match.group(2))
            continue

        match = _PASSED_RE.match(line)
        if match:
            stage = match.group("stage")
            counter = {
                "MSV": "n_past_msv",
                "bias": "n_past_bias",
                "Vit": "n_past_vit",
                "Fwd": "n_past_fwd",
            }.get(stage)
            if counter:
                current.setdefault("counts", {})[counter] = int(match.group("count"))
                try:
                    current.setdefault("expected", {})[counter] = float(match.group("exp"))
                except ValueError:
                    pass
            continue

        match = _SEARCH_SPACE_RE.match(line)
        if match:
            current.setdefault("search_space", {})["Z"] = float(match.group(1))
            continue

        match = _DOM_SPACE_RE.match(line)
        if match:
            current.setdefault("search_space", {})["domZ"] = float(match.group(1))
            continue

        match = _CPU_TIME_RE.match(line)
        if match:
            timing = current.setdefault("timing", {})
            user = float(match.group("user"))
            sys = float(match.group("sys")) if match.group("sys") else None
            timing["cpu_user_s"] = round(user, 4)
            if sys is not None:
                timing["cpu_sys_s"] = round(sys, 4)
                timing["cpu_total_s"] = round(user + sys, 4)
            timing["cpu_reported_s"] = _parse_hms(match.group("cpu_hms"))
            # ``Elapsed`` is wall clock -- the one comparable to cost_s.
            timing["elapsed_s"] = _parse_hms(match.group("elapsed_hms"))
            timing["timer"] = "esl_stopwatch"
            timing["cpu_is_not_wall"] = True
            continue

        match = _MC_PER_SEC_RE.match(line)
        if match:
            current.setdefault("timing", {})["mc_per_sec"] = float(match.group(1))
            continue

    return [_finalise(record) for record in records]


def parse_hmmsearch_output(text: str) -> dict[str, Any] | None:
    """Parse the single stats block ``hmmsearch`` prints at the end."""
    records = _parse_statistics_only(text)
    return records[-1] if records else None


def _parse_statistics_only(text: str) -> list[dict[str, Any]]:
    """Parse stats blocks that carry no ``@@ Round:`` markers."""
    records: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        if _PIPELINE_HEADER_RE.match(line):
            current = _empty_record()
            current["has_pipeline_block"] = True
            records.append(current)
            continue
        if current is None:
            continue
        for pattern, handler in (
            (_QUERY_MODELS_RE, lambda m, r: r.setdefault("search_space", {}).update(
                {"query_models": int(m.group(1)), "model_nodes": int(m.group(2))})),
            (_TARGET_SEQS_RE, lambda m, r: r.setdefault("search_space", {}).update(
                {"target_sequences": int(m.group(1)), "residues_searched": int(m.group(2))})),
            (_SEARCH_SPACE_RE, lambda m, r: r.setdefault("search_space", {}).update(
                {"Z": float(m.group(1))})),
            (_DOM_SPACE_RE, lambda m, r: r.setdefault("search_space", {}).update(
                {"domZ": float(m.group(1))})),
        ):
            match = pattern.match(line)
            if match:
                handler(match, current)
                break
        else:
            match = _PASSED_RE.match(line)
            if match:
                counter = {
                    "MSV": "n_past_msv",
                    "bias": "n_past_bias",
                    "Vit": "n_past_vit",
                    "Fwd": "n_past_fwd",
                }.get(match.group("stage"))
                if counter:
                    current.setdefault("counts", {})[counter] = int(match.group("count"))
                continue
            match = _CPU_TIME_RE.match(line)
            if match:
                timing = current.setdefault("timing", {})
                user = float(match.group("user"))
                sys = float(match.group("sys")) if match.group("sys") else None
                timing["cpu_user_s"] = round(user, 4)
                if sys is not None:
                    timing["cpu_sys_s"] = round(sys, 4)
                    timing["cpu_total_s"] = round(user + sys, 4)
                timing["cpu_reported_s"] = _parse_hms(match.group("cpu_hms"))
                timing["elapsed_s"] = _parse_hms(match.group("elapsed_hms"))
                timing["timer"] = "esl_stopwatch"
                timing["cpu_is_not_wall"] = True
                continue
            match = _MC_PER_SEC_RE.match(line)
            if match:
                current.setdefault("timing", {})["mc_per_sec"] = float(match.group(1))
                continue
    return [_finalise(record) for record in records]


def build_stage_timing(
    *,
    jackhmmer_text: str | None = None,
    hmmsearch_text: str | None = None,
    stage_times_s: dict[str, float] | None = None,
    stage_calls: dict[str, int] | None = None,
) -> dict[str, Any]:
    """Assemble the composite report the optimizing agent consumes.

    ``stage_times_s`` is filled by the instrumented profiling build. Absent it,
    stage entries report counts only and carry no ``time_s`` key -- see the module
    docstring on why a missing number beats a fabricated one.

    ``stage_calls`` distinguishes "instrumented, and it genuinely cost ~0" from
    "never instrumented at all". Without it a stage that was never entered would
    report ``time_s: 0.0``, which reads to an agent as "this stage is free" rather
    than "this stage was not measured". When call counts are supplied, a stage with
    zero calls gets no ``time_s`` key.
    """
    iterations = parse_jackhmmer_output(jackhmmer_text) if jackhmmer_text else []
    final_search = parse_hmmsearch_output(hmmsearch_text) if hmmsearch_text else None

    measured = bool(stage_times_s)
    stages: dict[str, dict[str, Any]] = {}
    for spec in PIPELINE_STAGES:
        contract = spec["contract"]
        entry: dict[str, Any] = {"pipeline_stage": spec["pipeline_stage"]}

        calls = stage_calls.get(contract) if stage_calls else None
        has_time = measured and stage_times_s is not None and contract in stage_times_s
        # Zero calls means the stage was never entered, so any 0.0 is an artefact
        # of the report listing every slot, not a measurement.
        if has_time and (calls is None or calls > 0):
            # 6 decimals = microsecond resolution, matching what the C shim writes
            # (%.6f) and what the parser reads. Rounding coarser would collapse a
            # genuinely measured fast stage (null2 is tens of microseconds) into
            # 0.0, which reads as "not measured" -- the exact confusion to avoid.
            entry["time_s"] = round(float(stage_times_s[contract]), 6)
            entry["time_source"] = "instrumented_build"
        else:
            entry["time_source"] = None
        if calls is not None:
            entry["calls"] = int(calls)

        counters = {
            counter: sum(
                int(rec.get("counts", {}).get(counter, 0)) for rec in iterations
            )
            for counter in spec["hmmer_counters"]
        }
        if counters:
            entry["counts_total"] = counters
        entry["note"] = spec["note"]
        stages[contract] = entry

    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "time_source": "instrumented_build" if measured else "output_parse",
        "stage_time_available": any(
            entry.get("time_s") is not None for entry in stages.values()
        ),
        "timer_semantics": (
            "elapsed_s is wall clock (esl_stopwatch w->elapsed) and is the figure "
            "comparable to cost_s; cpu_*_s is user+sys across worker threads and may "
            "exceed elapsed_s. Neither sums to cost_s, which also covers hmmsearch. "
            "stage time_s comes from in-band timers in a separate profiling build and "
            "carries instrumentation overhead: use it for attribution, not acceptance."
        ),
        "contract_stages": stages,
        "jackhmmer_iterations": iterations,
    }
    if final_search is not None:
        report["hmmsearch"] = final_search
    if iterations:
        elapsed = [
            rec["timing"]["elapsed_s"]
            for rec in iterations
            if isinstance(rec.get("timing"), dict) and rec["timing"].get("elapsed_s") is not None
        ]
        if elapsed:
            slowest = max(range(len(elapsed)), key=elapsed.__getitem__)
            report["jackhmmer_summary"] = {
                "iterations_reported": len(iterations),
                "elapsed_total_s": round(sum(elapsed), 4),
                "slowest_iteration_index": slowest,
                "slowest_iteration_round": iterations[slowest].get("round"),
                "slowest_iteration_elapsed_s": elapsed[slowest],
            }
    if not measured:
        report["stage_time_unavailable_reason"] = (
            "HMMER reports per-stage pass counts but not per-stage times. Run the "
            "instrumented profiling build to populate stage time_s."
        )
    return report


# --------------------------------------------------------------------------
# Bounded reader.
#
# ``jackhmmer.out`` interleaves full per-iteration hit tables with the statistics
# blocks, and the hit tables dominate the size. Retaining only the evidence lines
# keeps memory flat and lets one pass over a multi-megabyte file feed the parser.
# --------------------------------------------------------------------------
EVIDENCE_PREFIXES: tuple[str, ...] = (
    "@@",
    "Internal pipeline statistics summary:",
    "Query model(s):",
    "Target sequences:",
    "Passed ",
    "Initial search space (Z):",
    "Domain search space  (domZ):",
    "# CPU time:",
    "# Mc/sec:",
)


def read_pipeline_evidence(
    path: str | Path | None,
    *,
    max_bytes: int = 64 * 1024 * 1024,
) -> tuple[str, dict[str, Any]]:
    """Return ``(evidence_text, stats)`` for a HMMER output file.

    Never raises: a missing or unreadable file yields empty evidence plus a reason
    in ``stats``, because profiling evidence must not be able to fail an eval.
    """
    stats: dict[str, Any] = {"path": str(path) if path is not None else None}
    if path is None:
        stats["reason"] = "no path"
        return "", stats

    candidate = Path(path)
    if not candidate.is_file():
        stats["reason"] = "file not found"
        return "", stats

    kept: list[str] = []
    bytes_read = 0
    truncated = False
    try:
        with open(candidate, "r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                bytes_read += len(line)
                if bytes_read > max_bytes:
                    truncated = True
                    break
                stripped = line.rstrip()
                if stripped.startswith(EVIDENCE_PREFIXES):
                    kept.append(stripped)
    except OSError as exc:
        stats["reason"] = f"read failed: {exc}"
        return "", stats

    stats["bytes_read"] = bytes_read
    stats["lines_kept"] = len(kept)
    stats["truncated"] = truncated
    if truncated:
        stats["reason"] = f"stopped after max_bytes={max_bytes}; later iterations omitted"
    return "\n".join(kept), stats


# --------------------------------------------------------------------------
# Instrumented-build output contract.
#
# The profiling build's C shim writes this file. Keeping the format trivial and
# line-oriented means the Python side never has to guess, and a malformed line
# degrades to a missing stage rather than a wrong number.
#
#   # p7_evo_prof.v1
#   # timer clock_gettime(CLOCK_MONOTONIC); unit seconds; time spent inside the stage
#   # note <free text>
#   ssv_filter 121.500000 10000
#   dp_kernel 160.250000 20480
#
# Fields: <contract> <seconds> <calls>. Contract names must match the component
# contracts in components/c_kernels.py, because those are what an agent may edit.
# --------------------------------------------------------------------------
INSTRUMENTED_SCHEMA = "p7_evo_prof.v1"
INSTRUMENTED_FILE = "p7_evo_prof.txt"
INSTRUMENTED_ENV_VAR = "P7_EVO_PROF_OUT"

_INSTRUMENTED_LINE_RE = re.compile(r"^(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s+(?P<seconds>[\d.]+)\s+(?P<calls>\d+)\s*$")


def parse_instrumented_stage_times(text: str) -> tuple[dict[str, float], dict[str, Any]]:
    """Parse the profiling build's output into ``{contract: seconds}``.

    Unknown contract names and malformed lines are collected in the diagnostics
    rather than silently dropped, so a naming drift is visible instead of looking
    like "the stage used no time".
    """
    stage_times: dict[str, float] = {}
    calls: dict[str, int] = {}
    unknown: list[str] = []
    malformed: list[str] = []
    notes: list[str] = []
    schema: str | None = None

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("#"):
            body = line.lstrip("#").strip()
            if body.startswith("p7_evo_prof"):
                schema = body.split()[0]
            elif body.lower().startswith("note"):
                notes.append(body[4:].strip())
            continue
        match = _INSTRUMENTED_LINE_RE.match(line)
        if match is None:
            malformed.append(line)
            continue
        name = match.group("name")
        if name not in CONTRACT_ORDER:
            unknown.append(name)
            continue
        stage_times[name] = round(float(match.group("seconds")), 6)
        calls[name] = int(match.group("calls"))

    diagnostics: dict[str, Any] = {
        "schema": schema,
        "calls": calls,
        "unknown_contracts": unknown,
        "malformed_lines": malformed,
    }
    if notes:
        diagnostics["notes"] = notes
    return stage_times, diagnostics


def load_instrumented_stage_times(
    qdir: str | Path | None,
) -> tuple[dict[str, float] | None, dict[str, Any]]:
    """Load stage times if the profiling build wrote them, else ``(None, reason)``."""
    if qdir is None:
        return None, {"reason": "no run directory"}
    path = Path(qdir) / INSTRUMENTED_FILE
    if not path.is_file():
        return None, {"reason": f"{INSTRUMENTED_FILE} not present", "path": str(path)}
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return None, {"reason": f"read failed: {exc}", "path": str(path)}
    stage_times, diagnostics = parse_instrumented_stage_times(text)
    diagnostics["path"] = str(path)
    if not stage_times:
        return None, {**diagnostics, "reason": "no recognised stage lines"}
    return stage_times, diagnostics
