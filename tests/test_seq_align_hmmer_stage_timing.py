"""Tests for per-stage HMMER timing extraction.

The fixtures below are transcribed from the printf format strings in HMMER 3.4
(``src/src/p7_pipeline.c`` ``p7_pli_Statistics``, ``src/src/jackhmmer.c`` iteration
loop, ``easel/esl_stopwatch.c`` ``esl_stopwatch_Display``). If a format changes,
these tests fail loudly rather than letting a parser silently return nothing.
"""

from __future__ import annotations

import json

from seq_align.hmmer_stage_timing import (
    CONTRACT_ORDER,
    INSTRUMENTED_FILE,
    build_stage_timing,
    load_instrumented_stage_times,
    parse_hmmsearch_output,
    parse_instrumented_stage_times,
    parse_jackhmmer_output,
    read_pipeline_evidence,
)

JACKHMMER_OUT = """\
# jackhmmer :: iteratively search a protein sequence against a protein database
# HMMER 3.4 (Aug 2023); http://hmmer.org/
# Copyright (C) 2023 Howard Hughes Medical Institute.
# Freely distributed under the BSD open source license.
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# query sequence file:             query.fa
# target sequence database:        uniref90.fasta
# maximum iterations set to:       3

Query:       sp|P12345|TEST_HUMAN  [L=118]

@@
@@ Round:                  1
@@ Included in MSA:        5 subsequences (query + 4 subseqs from 3 targets)
@@ Model size:             124 positions
@@

Internal pipeline statistics summary:
-------------------------------------
Query model(s):                            1  (124 nodes)
Target sequences:                      10000  (2500000 residues searched)
Passed MSV filter:                       742   (0.0742); expected 800.0 (0.08)
Passed bias filter:                      701   (0.0701); expected 760.0 (0.076)
Passed Vit filter:                        83   (0.0083); expected 90.0 (0.009)
Passed Fwd filter:                        12   (0.0012); expected 14.0 (0.0014)
Initial search space (Z):              10000  [actual number of targets]
Domain search space  (domZ):              11  [number of targets reported over threshold]
# CPU time: 41.20u 1.35s 00:00:42.55  Elapsed: 00:02:12.30
# Mc/sec: 234.42

@@ New targets included:   7
@@ New alignment includes: 8 subseqs (was 5), including original query

@@
@@ Round:                  2
@@ Included in MSA:        9 subsequences (query + 8 subseqs from 7 targets)
@@ Model size:             131 positions
@@

Internal pipeline statistics summary:
-------------------------------------
Query model(s):                            1  (131 nodes)
Target sequences:                      10000  (2500000 residues searched)
Passed MSV filter:                       803   (0.0803); expected 820.0 (0.082)
Passed bias filter:                      760   (0.0760); expected 780.0 (0.078)
Passed Vit filter:                        91   (0.0091); expected 95.0 (0.0095)
Passed Fwd filter:                        14   (0.0014); expected 15.0 (0.0015)
Initial search space (Z):              10000  [actual number of targets]
Domain search space  (domZ):              13  [number of targets reported over threshold]
# CPU time: 88.90u 2.10s 00:01:31.00  Elapsed: 00:04:45.10
# Mc/sec: 110.20

@@ New targets included:   1
@@ New alignment includes: 10 subseqs (was 9), including original query

@@
@@ Round:                  3
@@ Included in MSA:        10 subsequences (query + 9 subseqs from 8 targets)
@@ Model size:             133 positions
@@

Internal pipeline statistics summary:
-------------------------------------
Query model(s):                            1  (133 nodes)
Target sequences:                      10000  (2500000 residues searched)
Passed MSV filter:                       799   (0.0799); expected 815.0 (0.0815)
Passed bias filter:                      755   (0.0755); expected 775.0 (0.0775)
Passed Vit filter:                        90   (0.0090); expected 94.0 (0.0094)
Passed Fwd filter:                        13   (0.0013); expected 14.0 (0.0014)
Initial search space (Z):              10000  [actual number of targets]
Domain search space  (domZ):              12  [number of targets reported over threshold]
# CPU time: 30.10u 0.80s 00:00:30.90  Elapsed: 00:01:02.70
# Mc/sec: 421.30

@@ New targets included:   0
@@ New alignment includes: 10 subseqs (was 10), including original query
@@
@@ CONVERGED (in 3 rounds).
"""

HMMSEARCH_OUT = """\
# hmmsearch :: search profile(s) against a sequence database
# HMMER 3.4 (Aug 2023); http://hmmer.org/

Internal pipeline statistics summary:
-------------------------------------
Query model(s):                            1  (133 nodes)
Target sequences:                      16986  (3200000 residues searched)
Passed MSV filter:                       512   (0.0301); expected 550.0 (0.0324)
Passed bias filter:                      498   (0.0293); expected 530.0 (0.0312)
Passed Vit filter:                        64   (0.0038); expected 70.0 (0.0041)
Passed Fwd filter:                        11   (0.0006); expected 12.0 (0.0007)
Initial search space (Z):              16986  [actual number of targets]
Domain search space  (domZ):              10  [number of targets reported over threshold]
# CPU time: 9.80u 0.30s 00:00:10.10  Elapsed: 00:00:08.40
# Mc/sec: 1506.19
"""


def test_parses_every_iteration_with_wall_and_cpu_time():
    iterations = parse_jackhmmer_output(JACKHMMER_OUT)

    assert [rec["round"] for rec in iterations] == [1, 2, 3]

    first = iterations[0]
    assert first["model_positions"] == 124
    assert first["msa_subsequences"] == 5
    assert first["msa_from_targets"] == 3
    timing = first["timing"]
    # Elapsed is wall clock; CPU is user+sys and is explicitly marked as such.
    assert timing["elapsed_s"] == 132.30
    assert timing["cpu_user_s"] == 41.20
    assert timing["cpu_sys_s"] == 1.35
    assert timing["cpu_total_s"] == 42.55
    assert timing["cpu_is_not_wall"] is True
    assert timing["mc_per_sec"] == 234.42


def test_parses_pipeline_counts_and_search_space():
    first = parse_jackhmmer_output(JACKHMMER_OUT)[0]

    assert first["counts"] == {
        "n_past_msv": 742,
        "n_past_bias": 701,
        "n_past_vit": 83,
        "n_past_fwd": 12,
    }
    assert first["expected"]["n_past_msv"] == 800.0
    assert first["search_space"]["target_sequences"] == 10000
    assert first["search_space"]["residues_searched"] == 2500000
    assert first["search_space"]["Z"] == 10000.0
    assert first["search_space"]["domZ"] == 11.0
    assert first["search_space"]["model_nodes"] == 124


def test_filter_cascade_attrition_is_derived():
    """The paper's mechanism: each stage culls most of what reached it."""
    cascade = parse_jackhmmer_output(JACKHMMER_OUT)[0]["filter_cascade"]

    assert cascade["n_past_msv"]["count"] == 742
    assert cascade["n_past_msv"]["fraction_of_targets"] == 0.0742
    # 701 of the 742 that passed MSV also passed the bias filter.
    assert cascade["n_past_bias"]["kept_from_previous_stage"] == round(701 / 742, 8)
    # The Viterbi filter is the sharpest cut here: 83/701.
    assert cascade["n_past_vit"]["kept_from_previous_stage"] == round(83 / 701, 8)
    assert cascade["n_past_fwd"]["kept_from_previous_stage"] == round(12 / 83, 8)


def test_convergence_lines_are_captured():
    iterations = parse_jackhmmer_output(JACKHMMER_OUT)

    assert iterations[0]["convergence"]["new_targets_included"] == 7
    assert iterations[0]["convergence"]["alignment_subsequences"] == 8
    assert iterations[0]["convergence"]["alignment_subsequences_before"] == 5
    assert iterations[2]["convergence"]["new_targets_included"] == 0
    assert iterations[2]["convergence"]["converged_at_round"] == 3


def test_hmmsearch_stats_block_is_parsed():
    block = parse_hmmsearch_output(HMMSEARCH_OUT)

    assert block is not None
    assert block["counts"]["n_past_msv"] == 512
    assert block["search_space"]["target_sequences"] == 16986
    assert block["timing"]["elapsed_s"] == 8.40


def test_no_stage_time_is_fabricated_without_the_instrumented_build():
    """A missing number must never be replaced by a guessed one."""
    report = build_stage_timing(
        jackhmmer_text=JACKHMMER_OUT, hmmsearch_text=HMMSEARCH_OUT
    )

    assert report["stage_time_available"] is False
    assert report["time_source"] == "output_parse"
    for contract, entry in report["contract_stages"].items():
        assert "time_s" not in entry, f"{contract} must not invent a stage time"
        assert entry["time_source"] is None
    assert "stage_time_unavailable_reason" in report


def test_stage_keys_are_component_contracts_with_descriptions():
    """JSON keys are contract names; the real HMMER stage is a description."""
    report = build_stage_timing(jackhmmer_text=JACKHMMER_OUT)
    stages = report["contract_stages"]

    assert tuple(stages) == CONTRACT_ORDER
    assert set(stages) == {
        "ssv_filter",
        "dp_kernel",
        "null2_correction",
        "domain_def",
        "pipeline_flow",
    }
    # The shape the migration plan calls for:
    #   {"dp_kernel": {"pipeline_stage": "Viterbi and Forward filters", ...}}
    assert stages["dp_kernel"]["pipeline_stage"] == "Viterbi and Forward filters"
    assert stages["ssv_filter"]["pipeline_stage"] == "MSV filter"


def test_stage_counts_are_summed_across_iterations():
    report = build_stage_timing(jackhmmer_text=JACKHMMER_OUT)

    # 742 + 803 + 799
    assert report["contract_stages"]["ssv_filter"]["counts_total"]["n_past_msv"] == 2344
    # dp_kernel owns both the Viterbi and Forward counters.
    dp = report["contract_stages"]["dp_kernel"]["counts_total"]
    assert dp["n_past_vit"] == 83 + 91 + 90
    assert dp["n_past_fwd"] == 12 + 14 + 13


def test_instrumented_stage_times_are_attached_and_labelled():
    report = build_stage_timing(
        jackhmmer_text=JACKHMMER_OUT,
        stage_times_s={
            "ssv_filter": 121.5,
            "dp_kernel": 160.25,
            "null2_correction": 4.5,
            "domain_def": 6.0,
            "pipeline_flow": 1.25,
        },
    )

    assert report["stage_time_available"] is True
    assert report["time_source"] == "instrumented_build"
    stages = report["contract_stages"]
    assert stages["dp_kernel"]["time_s"] == 160.25
    assert stages["dp_kernel"]["time_source"] == "instrumented_build"
    assert stages["ssv_filter"]["time_s"] == 121.5
    assert "stage_time_unavailable_reason" not in report


def test_identifies_the_slowest_iteration():
    """Iteration 2 is the slowest (285.10s) -- the agent should know that."""
    report = build_stage_timing(jackhmmer_text=JACKHMMER_OUT)
    summary = report["jackhmmer_summary"]

    assert summary["iterations_reported"] == 3
    assert summary["elapsed_total_s"] == round(132.30 + 285.10 + 62.70, 4)
    assert summary["slowest_iteration_round"] == 2
    assert summary["slowest_iteration_elapsed_s"] == 285.10


def test_report_is_json_serialisable():
    report = build_stage_timing(
        jackhmmer_text=JACKHMMER_OUT, hmmsearch_text=HMMSEARCH_OUT
    )
    # Must survive a round trip: it is written into EvalResult meta.
    assert json.loads(json.dumps(report))["schema_version"] == "hmmer_stage_timing.v1"


def test_missing_output_is_not_an_error():
    report = build_stage_timing()
    assert report["jackhmmer_iterations"] == []
    assert report["stage_time_available"] is False


def test_output_without_times_flag_still_parses_cpu_segment():
    """easel built without HAVE_TIMES omits the trailing '<sys>s' segment."""
    text = JACKHMMER_OUT.replace(
        "# CPU time: 41.20u 1.35s 00:00:42.55  Elapsed: 00:02:12.30",
        "# CPU time: 41.20u 00:00:42.55  Elapsed: 00:02:12.30",
    )
    timing = parse_jackhmmer_output(text)[0]["timing"]

    assert timing["cpu_user_s"] == 41.20
    assert "cpu_sys_s" not in timing
    assert "cpu_total_s" not in timing
    assert timing["elapsed_s"] == 132.30


def test_bounded_reader_keeps_only_evidence_lines(tmp_path):
    """jackhmmer.out is dominated by hit tables; those must not be retained."""
    noisy = (
        "Query:       sp|P12345|TEST_HUMAN  [L=118]\n"
        + "\n".join(f"     {i}   1.2e-10   120.5   3.1   9/120   hit_{i}" for i in range(5000))
        + "\n"
        + JACKHMMER_OUT
    )
    path = tmp_path / "jackhmmer.out"
    path.write_text(noisy, encoding="utf-8")

    text, stats = read_pipeline_evidence(path)

    assert stats["truncated"] is False
    assert "hit_4999" not in text
    assert "Internal pipeline statistics summary:" in text
    assert len(text.splitlines()) < 120  # evidence only, not 5000+ hit rows
    assert [rec["round"] for rec in parse_jackhmmer_output(text)] == [1, 2, 3]


def test_bounded_reader_reports_missing_file_without_raising(tmp_path):
    text, stats = read_pipeline_evidence(tmp_path / "nope.out")

    assert text == ""
    assert stats["reason"] == "file not found"


def test_query_profile_embeds_stage_timing_and_leaves_cost_s_alone(tmp_path):
    """The eval path must gain evidence without changing the measured cost."""
    from seq_align import run_hmmer_cli

    qdir = tmp_path / "query_0001"
    qdir.mkdir()
    (qdir / "jackhmmer.out").write_text(JACKHMMER_OUT, encoding="utf-8")
    (qdir / "hmmsearch.out").write_text(HMMSEARCH_OUT, encoding="utf-8")
    (qdir / "chk-3.hmm").write_text("HMMER3/f\n", encoding="utf-8")

    profile = run_hmmer_cli._query_profile(
        query_idx=1,
        query_id="TEST",
        query_path="query.fa",
        qdir=qdir,
        status="ok",
        cost_s=99.1234,
        iterations=3,
        nhits=12,
        commands={"jackhmmer": {"wall_s": 480.1}, "hmmsearch": {"wall_s": 8.4}},
    )

    # Schema stays v1: this is an additive field, and consumers read by key.
    assert profile["schema_version"] == "hmmer_cli_query_profile.v1"
    # cost_s is passed through untouched -- the whole acceptance gate rests on it.
    assert profile["cost_s"] == 99.1234

    stage_timing = profile["stage_timing"]
    assert stage_timing["stage_time_available"] is False
    assert stage_timing["jackhmmer_summary"]["slowest_iteration_round"] == 2
    assert "time_s" not in stage_timing["contract_stages"]["dp_kernel"]
    assert stage_timing["evidence_reads"]["jackhmmer_out"]["lines_kept"] > 0


def test_query_profile_on_missing_outputs_still_produces_a_profile(tmp_path):
    """A failed query has no HMMER output; profiling must not break the profile."""
    from seq_align import run_hmmer_cli

    qdir = tmp_path / "query_0002"
    qdir.mkdir()

    profile = run_hmmer_cli._query_profile(
        query_idx=2,
        query_id="TEST",
        query_path="query.fa",
        qdir=qdir,
        status="run_error",
        cost_s=1.0,
        iterations=0,
        nhits=None,
        commands={},
    )

    assert profile["cost_s"] == 1.0
    assert profile["stage_timing"]["jackhmmer_iterations"] == []
    assert profile["stage_timing"]["evidence_reads"]["jackhmmer_out"]["reason"] == "file not found"


def test_aggregate_rolls_stage_times_up_across_queries():
    """Session-level 'which stage dominates' must be available to the agent."""
    from seq_align import run_hmmer_cli

    def profile(query_idx: int, slowest_round: int, stage_times: dict | None) -> dict:
        report = build_stage_timing(
            jackhmmer_text=JACKHMMER_OUT,
            stage_times_s=stage_times,
        )
        report["jackhmmer_summary"]["slowest_iteration_round"] = slowest_round
        return {
            "query_idx": query_idx,
            "query_id": f"q{query_idx}",
            "status": "ok",
            "cost_s": 100.0,
            "iterations": 3,
            "nhits": 10,
            "commands": {"jackhmmer": {"wall_s": 90.0, "status": "ok"}},
            "stage_timing": report,
        }

    profiles = [
        profile(1, 1, {"ssv_filter": 10.0, "dp_kernel": 30.0, "null2_correction": 1.0}),
        profile(2, 2, {"ssv_filter": 20.0, "dp_kernel": 40.0, "null2_correction": 2.0}),
    ]

    aggregate = run_hmmer_cli._aggregate_profiles(
        profiles,
        stage="full",
        status="ok",
        cpus=1,
        max_iter=3,
        query_count=2,
        cost_s=200.0,
    )

    rollup = aggregate["stage_timing"]
    assert rollup["stage_time_available"] is True
    assert rollup["time_source"] == "instrumented_build"
    assert rollup["stage_time_s"] == {
        "dp_kernel": 70.0,
        "null2_correction": 3.0,
        "ssv_filter": 30.0,
    }
    # dp_kernel dominates: 70 / 103 of measured stage time.
    assert rollup["stage_time_fraction"]["dp_kernel"] == round(70.0 / 103.0, 6)
    assert rollup["measured_stage_total_s"] == 103.0
    # The untouched measured cost is still reported separately.
    assert aggregate["command_wall_s"] == {"jackhmmer": 180.0}


def test_aggregate_without_instrumentation_reports_parse_source_only():
    from seq_align import run_hmmer_cli

    profiles = [
        {
            "query_idx": 1,
            "query_id": "q1",
            "status": "ok",
            "cost_s": 50.0,
            "iterations": 3,
            "nhits": 5,
            "commands": {"jackhmmer": {"wall_s": 45.0, "status": "ok"}},
            "stage_timing": build_stage_timing(jackhmmer_text=JACKHMMER_OUT),
        }
    ]

    aggregate = run_hmmer_cli._aggregate_profiles(
        profiles, stage="smoke", status="ok", cpus=1, max_iter=3, query_count=1, cost_s=50.0
    )

    rollup = aggregate["stage_timing"]
    assert rollup["stage_time_available"] is False
    assert rollup["stage_time_s"] == {}
    assert rollup["time_source"] == "output_parse"
    assert rollup["slowest_iteration_query_id"] == "q1"
    # Per-iteration wall clock is still available without any C instrumentation.
    assert rollup["jackhmmer_elapsed_stats"]["count"] == 1


def test_compact_query_profile_shrinks_iteration_detail():
    from seq_align import run_hmmer_cli

    full = {
        "query_idx": 1,
        "query_id": "q1",
        "status": "ok",
        "cost_s": 10.0,
        "iterations": 3,
        "nhits": 5,
        "commands": {"jackhmmer": {"wall_s": 9.0}},
        "stage_timing": build_stage_timing(jackhmmer_text=JACKHMMER_OUT),
    }

    compact = run_hmmer_cli._compact_query_profile(full)

    # Headline facts are kept...
    assert compact["stage_timing"]["iterations_reported"] == 3
    assert compact["stage_timing"]["slowest_iteration_round"] == 2
    # ...but per-iteration detail is dropped so stored profiles stay small.
    assert "jackhmmer_iterations" not in compact["stage_timing"]
    assert "contract_stages" not in compact["stage_timing"]


def test_compact_query_profile_keeps_the_instrumented_profile_diagnostic():
    """2026-10-08 regression: the *reason* profiling failed was thrown away.

    Stored records said only "no stage times", so a profiling build that never
    emitted p7_evo_prof.txt was indistinguishable from one that emitted nothing
    parseable. The diagnostic has to survive compaction.
    """
    from seq_align import run_hmmer_cli

    diagnostic = {
        "reason": f"{INSTRUMENTED_FILE} not present",
        "path": "/some/qdir/p7_evo_prof.txt",
        "bytes_read": 0,
    }
    stage_timing = build_stage_timing(jackhmmer_text=JACKHMMER_OUT)
    stage_timing["evidence_reads"] = {"instrumented_profile": diagnostic}
    full = {
        "query_idx": 7,
        "query_id": "q7",
        "status": "ok",
        "cost_s": 3.0,
        "iterations": 1,
        "nhits": 2,
        "commands": {"jackhmmer": {"wall_s": 3.0}},
        "stage_timing": stage_timing,
    }

    compact = run_hmmer_cli._compact_query_profile(full)

    assert compact["stage_timing"]["evidence_reads"]["instrumented_profile"] == diagnostic


def test_aggregate_carries_the_instrumented_profile_reason():
    """A failed profiling pass must persist its cause into the aggregate profile."""
    from seq_align import run_hmmer_cli

    diagnostic = {"reason": f"{INSTRUMENTED_FILE} not present", "path": "/q/p7_evo_prof.txt"}
    profiles = []
    for idx, qid in ((1, "q1"), (2, "q2")):
        stage_timing = build_stage_timing(jackhmmer_text=JACKHMMER_OUT)
        stage_timing["evidence_reads"] = {"instrumented_profile": diagnostic}
        profiles.append(
            {
                "query_idx": idx,
                "query_id": qid,
                "status": "ok",
                "cost_s": 10.0,
                "iterations": 1,
                "nhits": 3,
                "commands": {"jackhmmer": {"wall_s": 10.0, "status": "ok"}},
                "stage_timing": stage_timing,
            }
        )

    aggregate = run_hmmer_cli._aggregate_profiles(
        profiles, stage="full", status="ok", cpus=8, max_iter=1, query_count=2, cost_s=20.0
    )

    rollup = aggregate["stage_timing"]
    assert rollup["time_source"] == "output_parse"
    # The cause is now visible in the aggregate itself, not only per query.
    assert rollup["instrumented_profile"]["reason"] == f"{INSTRUMENTED_FILE} not present"
    assert [item["query_id"] for item in rollup["per_query_instrumented_profile"]] == ["q1", "q2"]
    # And the stored (compacted) per-query profiles keep it too.
    assert aggregate["queries"][0]["stage_timing"]["evidence_reads"]["instrumented_profile"]["reason"]


# ---------------------------------------------------------------------------
# Instrumented profiling build: output contract
# ---------------------------------------------------------------------------

INSTRUMENTED_OK = """\
# p7_evo_prof.v1
# timer clock_gettime(CLOCK_MONOTONIC); unit seconds; time inside the stage
# note built with -DP7_EVO_PROF from the candidate source
ssv_filter 121.500000 10000
dp_kernel 160.250000 20480
null2_correction 4.500000 128
domain_def 6.000000 64
pipeline_flow 1.250000 3
"""


def test_parses_instrumented_stage_times():
    stage_times, diagnostics = parse_instrumented_stage_times(INSTRUMENTED_OK)

    assert stage_times == {
        "ssv_filter": 121.5,
        "dp_kernel": 160.25,
        "null2_correction": 4.5,
        "domain_def": 6.0,
        "pipeline_flow": 1.25,
    }
    assert diagnostics["schema"] == "p7_evo_prof.v1"
    assert diagnostics["calls"]["dp_kernel"] == 20480
    assert diagnostics["unknown_contracts"] == []
    assert diagnostics["malformed_lines"] == []


def test_instrumented_parser_reports_drift_instead_of_dropping_it():
    """A renamed contract must be visible, not look like 'the stage used no time'."""
    text = INSTRUMENTED_OK + "some_renamed_kernel 5.0 10\nnot a valid line\n"

    stage_times, diagnostics = parse_instrumented_stage_times(text)

    assert "some_renamed_kernel" not in stage_times
    assert diagnostics["unknown_contracts"] == ["some_renamed_kernel"]
    assert diagnostics["malformed_lines"] == ["not a valid line"]


def test_load_instrumented_stage_times_missing_file_is_not_an_error(tmp_path):
    stage_times, diagnostics = load_instrumented_stage_times(tmp_path)

    assert stage_times is None
    assert "not present" in diagnostics["reason"]


def test_load_instrumented_stage_times_reads_the_contract_file(tmp_path):
    (tmp_path / INSTRUMENTED_FILE).write_text(INSTRUMENTED_OK, encoding="utf-8")

    stage_times, diagnostics = load_instrumented_stage_times(tmp_path)

    assert stage_times is not None
    assert stage_times["ssv_filter"] == 121.5
    assert diagnostics["path"].endswith(INSTRUMENTED_FILE)


def test_uninstrumented_stage_gets_no_time_rather_than_zero():
    """A slot the profiling build never entered must not read as 'free'.

    The C reporter writes one line per slot, including slots that no injected
    timer ever reached. Those arrive as 0.0, which an agent would read as "this
    stage costs nothing" instead of "this stage was not measured".
    """
    stage_times = {
        "ssv_filter": 1.5,
        "dp_kernel": 2.0,
        "null2_correction": 0.0,
        "domain_def": 0.0,
        "pipeline_flow": 0.0,
    }
    calls = {
        "ssv_filter": 100,
        "dp_kernel": 50,
        "null2_correction": 0,
        "domain_def": 0,
        "pipeline_flow": 0,
    }

    report = build_stage_timing(stage_times_s=stage_times, stage_calls=calls)
    stages = report["contract_stages"]

    assert stages["ssv_filter"]["time_s"] == 1.5
    assert stages["ssv_filter"]["calls"] == 100
    for never_entered in ("null2_correction", "domain_def", "pipeline_flow"):
        assert "time_s" not in stages[never_entered], never_entered
        assert stages[never_entered]["calls"] == 0
        assert stages[never_entered]["time_source"] is None
    # Overall availability is still true: some stages were genuinely measured.
    assert report["stage_time_available"] is True


def test_both_layers_compose_in_one_query_profile(tmp_path):
    """Parse layer gives iterations/counts; instrumented layer fills stage times."""
    from seq_align import run_hmmer_cli

    qdir = tmp_path / "query_0003"
    qdir.mkdir()
    (qdir / "jackhmmer.out").write_text(JACKHMMER_OUT, encoding="utf-8")
    (qdir / INSTRUMENTED_FILE).write_text(INSTRUMENTED_OK, encoding="utf-8")

    profile = run_hmmer_cli._query_profile(
        query_idx=3,
        query_id="TEST",
        query_path="query.fa",
        qdir=qdir,
        status="ok",
        cost_s=500.0,
        iterations=3,
        nhits=12,
        commands={"jackhmmer": {"wall_s": 480.1}},
    )
    stage_timing = profile["stage_timing"]

    # The instrumented layer supplies per-stage times...
    assert stage_timing["stage_time_available"] is True
    assert stage_timing["time_source"] == "instrumented_build"
    assert stage_timing["contract_stages"]["dp_kernel"]["time_s"] == 160.25
    assert stage_timing["contract_stages"]["ssv_filter"]["time_s"] == 121.5
    # ...while the parse layer still supplies the per-iteration picture.
    assert stage_timing["jackhmmer_summary"]["slowest_iteration_round"] == 2
    assert stage_timing["contract_stages"]["ssv_filter"]["counts_total"]["n_past_msv"] == 2344
    # And the measured cost is untouched.
    assert profile["cost_s"] == 500.0

    # The stage times attribute work *within* jackhmmer, which is a subset of
    # cost_s (which also covers hmmsearch). They must not be read as the total.
    measured = sum(
        entry["time_s"] for entry in stage_timing["contract_stages"].values() if "time_s" in entry
    )
    assert measured < profile["cost_s"]
