import json
import inspect
import os
import subprocess
from pathlib import Path

import pytest

from seq_align import run_hmmer_cli, run_jack
from seq_align.c_patch_loop import controller
from seq_align.c_patch_loop.context import build_context_pack
from seq_align.c_patch_loop.experience import (
    algorithm_mechanism_projection,
    implementation_mechanism_projection,
    mechanism_fingerprint,
)
from seq_align.c_patch_loop.models import AlgorithmPlanProposal, AlgorithmPlanRecord, DurableWisdomRecord, EvalSummary, EvaluationRecord, ExperienceRecord, InvocationRecord, InvocationSummaryRecord, PatchPlanProposal, PatchProposal, PatchRecord, PatchsetRecord, PhaseSummaryRecord
from seq_align.c_patch_loop.proposer import (
    BaseProposer,
    FatalProposalError,
    FixtureProposer,
    OpenAICompatibleProposer,
    ProposalError,
    _is_fatal_model_status,
    parse_algorithm_plan_proposal,
    parse_patch_plan_proposal,
    parse_patch_proposal,
)
from seq_align.c_patch_loop.runner import PatchRunner, RuntimeIdentityUnavailable
from seq_align.c_patch_loop.store import LoopStore, canonical_json, record_hash, sha256_text, utc_now
from seq_align.eval_schema import EvalResult
from seq_align.run_hmmer_cli import _normalize_patch_paths


# Canonical measurement conditions for the synthetic summaries/results these
# tests build. A cost may only be subtracted/divided from another when every one
# of these fields matches, so both sides of a comparison use the same
# fingerprint; a test that wants a refusal overrides exactly one key.
_COMPARABILITY: dict = {
    "schema_version": "test_comparability",
    "backend": "hmmer_cli_local",
    "partition": "compute",
    "allocated_cpus": 45,
    "query_threads": 8,
    "queries_in_job": 5,
    "stage": "full",
    "query_set_sha256": "q" * 64,
    "max_iter": 1,
    "uniref_db_path": "/db/uniref90.fasta",
    "uniref_db_sha256": "u" * 64,
    "astral_db_path": "/db/astral.fa",
    "astral_db_sha256": "a" * 64,
    "scop_txt_path": "/db/scop.txt",
    "scop_txt_sha256": "s" * 64,
    "hmmer_source_sha256": "h" * 64,
    "eval_repeats": 1,
    "paired_current_best_repeats": 0,
    "bundle_id": "b" * 64,
    "build_context": "test",
}


def _comparable(**overrides: object) -> dict:
    """A fingerprint whose conditions match, so the decision logic is exercised.

    Override a key to construct a deliberate mismatch (cpus, query set, stage,
    database). ``stage`` must be the stage actually evaluated; derive it from the
    helper's argument when one helper serves both ``smoke`` and ``full``.
    """
    return {**_COMPARABILITY, **overrides}


def _profile(
    *,
    dominant: str = "jackhmmer",
    jack: float = 3.0,
    search: float = 1.0,
    top_query: str = "query1",
) -> dict:
    total = jack + search
    return {
        "schema_version": "hmmer_cli_eval_profile.v1",
        "dominant_command": dominant,
        "command_wall_s": {"jackhmmer": jack, "hmmsearch": search},
        "command_wall_fraction": {
            "jackhmmer": jack / total if total else 0.0,
            "hmmsearch": search / total if total else 0.0,
        },
        "query_cost_stats": {"count": 1, "max": total},
        "top_slowest_queries": [
            {"query_idx": 1, "query_id": top_query, "cost_s": total, "dominant_command": dominant}
        ],
    }


def test_parse_rejects_python_payload():
    raw = json.dumps({
        "component_name": "null2_correction",
        "rationale": "bad",
        "python_payload": {"rule": "evalue"},
        "unified_diff": "--- a/src/generic_null2.c\n+++ b/src/generic_null2.c\n",
    })
    with pytest.raises(ProposalError):
        parse_patch_proposal(raw)


def test_parse_rejects_diff_in_patch_plan():
    raw = json.dumps({
        "component_name": "null2_correction",
        "target_functions": ["p7_Null2_ByExpectation"],
        "algorithm_hypothesis": "bad",
        "intended_code_change": "bad",
        "runtime_effective_argument": "bad",
        "unified_diff": "--- a/src/impl_sse/null2.c\n+++ b/src/impl_sse/null2.c\n",
    })
    with pytest.raises(ProposalError):
        parse_patch_plan_proposal(raw)



def test_parse_rejects_diff_or_edits_in_algorithm_plan():
    raw = json.dumps({
        "algorithm_family": "pipeline_flow_strategy",
        "baseline_algorithm_summary": "Stock p7_Pipeline applies the full filter cascade before reporting hits.",
        "proposed_algorithm_change": "Add a candidate-pruning decision surface with exact fallback for borderline targets.",
        "speed_mechanism": "Reduce expensive downstream search work for confidently rejected candidates while exact fallback preserves borderline scoring.",
        "quality_preservation_argument": "TP and FP remain unchanged because reportable/includable candidates use the exact stock path.",
        "target_contracts": ["pipeline_flow"],
        "target_functions": ["p7_Pipeline"],
        "profile_evidence_citation": "No measured profile evidence is available in this context pack.",
        "edits": [{"old": "x", "new": "y"}],
    })
    with pytest.raises(ProposalError):
        parse_algorithm_plan_proposal(raw)



def test_hmmer_cli_query_profile_records_command_timings_and_artifacts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    query = tmp_path / "query.fasta"
    query.write_text(">d1abcA_\nAAAA\n", encoding="utf-8")

    def fake_run(cmd, *, cwd, label, env=None, timeout=None):
        if label == "jackhmmer":
            (cwd / "jackhmmer.tbl").write_text("# jack\n", encoding="utf-8")
            (cwd / "jackhmmer.out").write_text("jack\n", encoding="utf-8")
            (cwd / "chk-1.hmm").write_text("HMM\n", encoding="utf-8")
        elif label == "hmmsearch":
            (cwd / "hmmsearch.tbl").write_text(
                "d1abcA_ - query - 1e-10 50.0 0.0\n"
                "d2xyzA_ - query - 1e-5 20.0 0.0\n",
                encoding="utf-8",
            )
            (cwd / "hmmsearch.out").write_text("search\n", encoding="utf-8")
        return subprocess.CompletedProcess(cmd, 0, "stdout should not be profiled", "")

    monkeypatch.setattr(run_hmmer_cli, "_run", fake_run)

    result = run_hmmer_cli._run_one_query(
        str(query),
        1,
        jackhmmer=Path("/bin/jackhmmer"),
        hmmsearch=Path("/bin/hmmsearch"),
        run_root=tmp_path / "queries",
        cpus=2,
        astral_db="astral.fa",
        uniref_db="uniref.fa",
        scop={"d1abcA_": "a.1.1.1", "d2xyzA_": "b.1.1.1"},
        cli_opts={"max_iter": 1, "jackhmmer": [], "hmmsearch": []},
        stage="smoke",
    )

    profile = result.meta["profile"]
    assert result.status == "ok"
    assert result.tp == 1
    assert result.fp == 1
    assert profile["schema_version"] == "hmmer_cli_query_profile.v1"
    assert profile["commands"]["jackhmmer"]["status"] == "ok"
    assert profile["commands"]["hmmsearch"]["status"] == "ok"
    assert "stdout" not in json.dumps(profile)
    assert profile["artifacts"]["checkpoints"]["count"] == 1
    assert profile["artifacts"]["hmmsearch_tbl"]["bytes"] > 0


def test_hmmer_cli_aggregate_profile_summarizes_dominant_command():
    profiles = [
        {
            "query_idx": 1,
            "query_id": "q1",
            "status": "ok",
            "cost_s": 5.0,
            "iterations": 1,
            "nhits": 2,
            "commands": {
                "jackhmmer": {"wall_s": 4.0, "status": "ok"},
                "hmmsearch": {"wall_s": 1.0, "status": "ok"},
            },
        },
        {
            "query_idx": 2,
            "query_id": "q2",
            "status": "run_error",
            "cost_s": 2.0,
            "iterations": 0,
            "nhits": None,
            "commands": {"jackhmmer": {"wall_s": 2.0, "status": "error"}},
        },
    ]

    profile = run_hmmer_cli._aggregate_profiles(
        profiles,
        stage="full",
        status="run_error",
        cpus=2,
        max_iter=1,
        query_count=2,
        cost_s=7.0,
    )

    assert profile["schema_version"] == "hmmer_cli_eval_profile.v1"
    assert profile["dominant_command"] == "jackhmmer"
    assert profile["command_wall_s"]["jackhmmer"] == 6.0
    assert profile["command_status_counts"]["jackhmmer"]["error"] == 1
    assert profile["status_counts"]["run_error"] == 1
    assert profile["top_slowest_queries"][0]["query_id"] == "q1"


def test_perf_report_parser_extracts_self_symbol_rows():
    text = """
# Children      Self  Command    Shared Object      Symbol
    42.10%    37.50%  jackhmmer  jackhmmer          [.] forward_engine
     9.00%     9.00%  jackhmmer  libhmmer.so        [.] p7_Null2_ByExpectation
"""

    rows = run_hmmer_cli._parse_perf_report(text, top_n=2, command="jackhmmer", query_idx=3, query_id="q3")

    assert rows[0]["symbol"] == "forward_engine"
    assert rows[0]["self_pct"] == 37.5
    assert rows[0]["query_id"] == "q3"
    assert rows[1]["symbol"] == "p7_Null2_ByExpectation"


def test_external_perf_profile_fail_open_when_perf_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    aggregate = {"status": "ok", "command_wall_s": {"jackhmmer": 1.0}, "dominant_command": "jackhmmer"}
    monkeypatch.setattr(run_hmmer_cli.shutil, "which", lambda name: None)

    run_hmmer_cli._attach_external_profile(
        aggregate,
        [],
        jackhmmer=Path("/bin/jackhmmer"),
        hmmsearch=Path("/bin/hmmsearch"),
        eval_root=tmp_path,
        cpus=1,
        astral_db="astral.fa",
        uniref_db="uniref.fa",
        cli_opts={"max_iter": 1, "jackhmmer": [], "hmmsearch": []},
        external_profiler="perf",
        perf_top_queries=1,
        perf_top_symbols=5,
        perf_frequency=99,
        perf_command="dominant",
    )

    profiler = aggregate["external_profiler"]
    assert profiler["status"] == "skipped"
    assert profiler["excluded_from_cost_s"] is True
    assert profiler["reason"] == "perf executable not found"
    assert "symbol_hotspots" not in aggregate


def test_external_perf_profile_reports_command_errors_without_failing_eval(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    aggregate = {"status": "ok", "command_wall_s": {"jackhmmer": 1.0}, "dominant_command": "jackhmmer"}
    query_profiles = [
        {
            "query_idx": 1,
            "query_id": "q1",
            "status": "ok",
            "cost_s": 1.0,
            "commands": {"jackhmmer": {"wall_s": 1.0, "status": "ok"}},
        }
    ]
    monkeypatch.setattr(run_hmmer_cli.shutil, "which", lambda name: "/usr/bin/perf")
    monkeypatch.setattr(
        run_hmmer_cli,
        "_profile_query_command_with_perf",
        lambda *args, **kwargs: {
            "query_idx": 1,
            "query_id": "q1",
            "command": "jackhmmer",
            "status": "error",
            "reason": "perf record exited 2: permission denied",
            "symbols": [],
        },
    )

    run_hmmer_cli._attach_external_profile(
        aggregate,
        query_profiles,
        jackhmmer=Path("/bin/jackhmmer"),
        hmmsearch=Path("/bin/hmmsearch"),
        eval_root=tmp_path,
        cpus=1,
        astral_db="astral.fa",
        uniref_db="uniref.fa",
        cli_opts={"max_iter": 1, "jackhmmer": [], "hmmsearch": []},
        external_profiler="perf",
        perf_top_queries=1,
        perf_top_symbols=5,
        perf_frequency=99,
        perf_command="dominant",
    )

    profiler = aggregate["external_profiler"]
    assert profiler["status"] == "error"
    assert "permission denied" in profiler["reason"]
    assert profiler["profiled_commands"][0]["symbol_count"] == 0
    assert "symbol_hotspots" not in aggregate


def test_gprof_report_parser_extracts_flat_profile_rows():
    text = """
Flat profile:

Each sample counts as 0.01 seconds.
  %   cumulative   self              self     total
 time   seconds   seconds    calls  ns/call  ns/call  name
 42.50      1.70     1.70                             forward_engine
  9.00      2.06     0.36      1200   300.00   400.00  p7_Null2_ByExpectation

Call graph
"""

    rows = run_hmmer_cli._parse_gprof_report(text, top_n=2, command="jackhmmer", query_idx=3, query_id="q3")

    assert rows[0]["symbol"] == "forward_engine"
    assert rows[0]["self_pct"] == 42.5
    assert rows[0]["self_s"] == 1.7
    assert rows[0]["profile_source"] == "gprof_flat_profile"
    assert rows[1]["symbol"] == "p7_Null2_ByExpectation"


def test_external_gprof_profile_reports_build_errors_without_failing_eval(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    aggregate = {"status": "ok", "command_wall_s": {"jackhmmer": 1.0}, "dominant_command": "jackhmmer"}
    query_profiles = [
        {
            "query_idx": 1,
            "query_id": "q1",
            "status": "ok",
            "cost_s": 1.0,
            "commands": {"jackhmmer": {"wall_s": 1.0, "status": "ok"}},
        }
    ]
    monkeypatch.setattr(run_hmmer_cli.shutil, "which", lambda name: "/usr/bin/gprof")
    monkeypatch.setattr(
        run_hmmer_cli,
        "_build_gprof_hmmer_copy",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("instrumented build failed")),
    )

    run_hmmer_cli._attach_external_profile(
        aggregate,
        query_profiles,
        jackhmmer=Path("/bin/jackhmmer"),
        hmmsearch=Path("/bin/hmmsearch"),
        eval_root=tmp_path,
        cpus=1,
        astral_db="astral.fa",
        uniref_db="uniref.fa",
        cli_opts={"max_iter": 1, "jackhmmer": [], "hmmsearch": []},
        external_profiler="gprof",
        perf_top_queries=1,
        perf_top_symbols=5,
        perf_frequency=99,
        perf_command="dominant",
        cfg=object(),
        source_root=tmp_path / "hmmer-src",
    )

    profiler = aggregate["external_profiler"]
    assert profiler["tool"] == "gprof"
    assert profiler["status"] == "error"
    assert "instrumented build failed" in profiler["reason"]
    assert profiler["excluded_from_cost_s"] is True
    assert "symbol_hotspots" not in aggregate


def test_external_gprof_profile_attaches_symbol_hotspots(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    aggregate = {"status": "ok", "command_wall_s": {"jackhmmer": 10.0}, "dominant_command": "jackhmmer"}
    query_profiles = [
        {
            "query_idx": 1,
            "query_id": "q1",
            "status": "ok",
            "cost_s": 10.0,
            "commands": {"jackhmmer": {"wall_s": 10.0, "status": "ok"}},
        }
    ]
    monkeypatch.setattr(run_hmmer_cli.shutil, "which", lambda name: "/usr/bin/gprof")
    monkeypatch.setattr(run_hmmer_cli, "_build_gprof_hmmer_copy", lambda *args, **kwargs: (Path("/bin/jackhmmer"), Path("/bin/hmmsearch")))
    monkeypatch.setattr(
        run_hmmer_cli,
        "_profile_query_command_with_gprof",
        lambda *args, **kwargs: {
            "query_idx": 1,
            "query_id": "q1",
            "command": "jackhmmer",
            "status": "ok",
            "symbols": [
                {
                    "rank": 1,
                    "command": "jackhmmer",
                    "query_idx": 1,
                    "query_id": "q1",
                    "symbol": "forward_engine",
                    "dso": "gprof",
                    "self_pct": 40.0,
                    "self_s": 4.0,
                    "approx_command_wall_s": 4.0,
                    "profile_source": "gprof_flat_profile",
                }
            ],
        },
    )

    run_hmmer_cli._attach_external_profile(
        aggregate,
        query_profiles,
        jackhmmer=Path("/bin/jackhmmer"),
        hmmsearch=Path("/bin/hmmsearch"),
        eval_root=tmp_path,
        cpus=1,
        astral_db="astral.fa",
        uniref_db="uniref.fa",
        cli_opts={"max_iter": 1, "jackhmmer": [], "hmmsearch": []},
        external_profiler="gprof",
        perf_top_queries=1,
        perf_top_symbols=5,
        perf_frequency=99,
        perf_command="dominant",
        cfg=object(),
        source_root=tmp_path / "hmmer-src",
    )

    profiler = aggregate["external_profiler"]
    assert profiler["status"] == "ok"
    assert profiler["profiled_commands"][0]["symbol_count"] == 1
    assert aggregate["symbol_hotspots"][0]["symbol"] == "forward_engine"
    assert aggregate["symbol_hotspots"][0]["profile_source"] == "gprof_flat_profile"


def test_controller_repairs_run_error_final_records():
    record = EvaluationRecord(
        eval_id="eval_parse_error",
        patchset_id="patchset",
        round_idx=1,
        stage="smoke",
        status="run_error",
        decision="rejected_run_error",
        eval_result={"stage": "smoke", "status": "run_error", "errors": ["jackhmmer failed: illegal character"]},
        created_at=utc_now(),
    )

    assert controller._should_repair_final(record) is True
    assert controller._repairable_final_failure_kind(record) == "semantic_runtime_failure"



def test_controller_classifies_repairable_and_environmental_failures():
    compile_record = EvaluationRecord(
        eval_id="eval_compile",
        patchset_id="patchset",
        round_idx=1,
        stage="static_verify",
        status="compile_error",
        decision="rejected_static",
        errors=["error: expected ';' before '}' token"],
        created_at=utc_now(),
    )
    apply_record = EvaluationRecord(
        eval_id="eval_apply",
        patchset_id="patchset",
        round_idx=1,
        stage="smoke",
        status="run_error",
        decision="rejected_run_error",
        eval_result={"stage": "smoke", "status": "run_error", "errors": ["git apply --check failed: patch does not apply"]},
        created_at=utc_now(),
    )
    runtime_provenance = EvaluationRecord(
        eval_id="eval_runtime",
        patchset_id="patchset",
        round_idx=1,
        stage="smoke",
        status="ok",
        decision="rejected_not_runtime_effective",
        eval_result={"stage": "smoke", "status": "ok", "tp": 1, "fp": 0, "cost_s": 1.0},
        build={"runtime_effective": False},
        created_at=utc_now(),
    )
    environmental = EvaluationRecord(
        eval_id="eval_env",
        patchset_id="patchset",
        round_idx=1,
        stage="smoke",
        status="run_error",
        decision="rejected_run_error",
        eval_result={"stage": "smoke", "status": "run_error", "errors": ["jackhmmer failed: timeout after 120s"]},
        errors=["jackhmmer failed: timeout after 120s"],
        created_at=utc_now(),
    )

    assert controller._repairable_final_failure_kind(compile_record) == "compile_error"
    assert controller._repairable_final_failure_kind(apply_record) == "patch_apply_failure"
    assert controller._repairable_final_failure_kind(runtime_provenance) == "runtime_provenance_failure"
    assert controller._repairable_final_failure_kind(environmental) is None
    assert controller._failure_kind_for_feedback(environmental) == "environmental_failure"



def test_controller_main_reports_evaluation_interruption_without_traceback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
    def interrupted_loop(**kwargs):
        raise run_hmmer_cli.EvaluationInterrupted("received signal 15")

    monkeypatch.setattr(controller, "run_loop", interrupted_loop)
    monkeypatch.setattr(controller, "make_proposer", lambda args: object())

    with pytest.raises(SystemExit) as raised:
        controller.main(["--run-dir", str(tmp_path)])

    assert raised.value.code == 2
    assert capsys.readouterr().err == "evaluation interrupted: received signal 15\n"



def test_controller_interruption_before_slot_running_finalizes_invocation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    class NoopProposer(BaseProposer):
        def propose(self, pack):
            raise AssertionError("proposal should not run")

    def interrupted_baseline(self, *, stage, keep_work=False):
        raise run_hmmer_cli.EvaluationInterrupted("stop requested")

    monkeypatch.setattr(
        controller.PatchRunner,
        "evaluate_stock_baseline",
        interrupted_baseline,
    )

    run_dir = tmp_path / "run"
    with pytest.raises(run_hmmer_cli.EvaluationInterrupted):
        controller.run_hierarchical_loop(
            run_dir=run_dir,
            outer_rounds=1,
            inner_trials=1,
            proposer=NoopProposer(),
        )

    store = LoopStore(run_dir)
    invocation = InvocationRecord.model_validate(store.iter_invocations()[0])
    summary = controller._latest_invocation_summaries(store)[invocation.invocation_id]
    run_summary = controller._read_run_summary(store)

    assert summary.terminal_state == "interrupted"
    assert run_summary is not None
    assert run_summary.terminal_state == "interrupted"
    assert all(
        head.state == "pending"
        for head in store.slot_heads(invocation_id=invocation.invocation_id).values()
    )



def test_controller_accepts_gprof_external_profiler(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    captured = {}

    def fake_run_loop(**kwargs):
        captured.update(kwargs)
        return []

    monkeypatch.setattr(controller, "run_loop", fake_run_loop)
    monkeypatch.setattr(controller, "make_proposer", lambda args: object())

    assert controller.main([
        "--run-dir", str(tmp_path),
        "--external-profiler", "gprof",
        "--max-plan-reproposal-attempts", "3",
        "--pyhmmer-profile-prepass",
        "--pyhmmer-profile-prepass-stage", "smoke",
        "--pyhmmer-profile-max-target-bytes", "12345",
    ]) == 0
    assert captured["external_profiler"] == "gprof"
    assert captured["max_plan_reproposal_attempts"] == 3
    assert captured["pyhmmer_profile_prepass"] is True
    assert captured["pyhmmer_profile_prepass_stage"] == "smoke"
    assert captured["pyhmmer_profile_max_target_bytes"] == 12345


def test_pyhmmer_profile_prepass_records_phase_profile(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from seq_align import run_jack

    def fake_run_all_queries(cfg, **kwargs):
        return EvalResult(
            stage=kwargs["stage"],
            status="ok",
            tp=10,
            fp=1,
            cost_s=12.0,
            n_seqs=2,
            iterations=3,
            meta={"profile": _pyhmmer_phase_profile()},
        )

    monkeypatch.setattr(run_jack, "run_all_queries", fake_run_all_queries)
    runner = PatchRunner(store := LoopStore(tmp_path), cpus=2, max_iter=1, full_max_iter=3)

    record = runner.evaluate_pyhmmer_profile_prepass(stage="full")

    assert record.status == "ok"
    assert record.decision == "profile_prepass"
    assert record.eval_config["backend"] == "pyhmmer"
    assert record.eval_config["max_iter"] == 3
    assert record.eval_result["meta"]["profile"]["dominant_phase"] == "jackhmmer_iterations"
    assert store.read_index("pyhmmer_profile_prepass_full")["eval_id"] == record.eval_id


def test_controller_accepts_wisdom_cli_flags(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    captured = {}

    def fake_run_loop(**kwargs):
        captured.update(kwargs)
        return []

    wisdom_dir = tmp_path / "wisdom"
    monkeypatch.setattr(controller, "run_loop", fake_run_loop)
    monkeypatch.setattr(controller, "make_proposer", lambda args: object())

    assert controller.main([
        "--run-dir", str(tmp_path / "run"),
        "--wisdom-dir", str(wisdom_dir),
        "--disable-durable-wisdom",
        "--import-seed-wisdom",
        "--promote-wisdom",
    ]) == 0
    assert captured["wisdom_dir"] == str(wisdom_dir)
    assert captured["disable_durable_wisdom"] is True
    assert captured["import_seed_wisdom"] is True
    assert captured["promote_wisdom"] is True


def test_fixture_proposer_writes_patch_plan_record(tmp_path: Path):
    store = LoopStore(tmp_path)
    pack = build_context_pack(store, round_idx=1)
    plan_proposal, metadata = FixtureProposer().propose_plan(pack)
    runner = PatchRunner(store)

    plan_record = runner.build_patch_plan(plan_proposal, pack, metadata)

    assert plan_record.validation_status == "ok"
    assert plan_record.patch_plan_id
    assert (store.patch_plans_dir / f"{plan_record.patch_plan_id}.json").exists()



@pytest.mark.requires_hmmer_source
def test_fixture_patch_plan_follows_approved_algorithm_contract(tmp_path: Path):
    store = LoopStore(tmp_path)
    pack = build_context_pack(store, round_idx=1)
    algorithm_plan = AlgorithmPlanProposal(
        algorithm_family="pipeline_flow_strategy",
        baseline_algorithm_summary="Use stock pipeline flow.",
        proposed_algorithm_change="Add a guarded exact-fallback pipeline decision.",
        speed_mechanism="Avoid downstream work only on an unreachable fixture guard.",
        quality_preservation_argument="Normal execution remains on the exact stock path.",
        target_contracts=["pipeline_flow"],
        target_functions=["p7_evo_align_Run"],
        profile_evidence_citation="No measured profile evidence is available.",
        novelty_against_l2_l3="Fixture mechanism for contract-binding validation.",
        implementation_constraints=[
            "The volatile guard is false in normal execution."
        ],
    )

    plan, _ = FixtureProposer().propose_plan(
        pack,
        algorithm_plan=algorithm_plan,
    )
    proposal, _ = FixtureProposer().propose(
        pack,
        plan=plan,
        algorithm_plan=algorithm_plan,
    )

    assert plan.component_name == "pipeline_flow"
    assert plan.target_functions == ["p7_evo_align_Run"]
    assert "null2[0]" not in proposal.unified_diff



def _pyhmmer_phase_profile(*, dominant_phase: str = "jackhmmer_iterations", fraction: float = 0.9) -> dict:
    other_fraction = round(max(0.0, 1.0 - fraction), 6)
    return {
        "schema_version": "pyhmmer_eval_profile.v1",
        "backend": "pyhmmer",
        "dominant_phase": dominant_phase,
        "phase_wall_s": {dominant_phase: fraction * 10.0, "final_hmmsearch": other_fraction * 10.0},
        "phase_wall_fraction": {dominant_phase: fraction, "final_hmmsearch": other_fraction},
        "query_cost_stats": {"count": 1, "max": 10.0},
        "iteration_stats": {"count": 1, "max": 3},
        "top_slowest_queries": [
            {
                "query_idx": 1,
                "query_id": "q_pyhmmer",
                "cost_s": 10.0,
                "dominant_phase": dominant_phase,
                "phase_wall_fraction": {dominant_phase: fraction},
                "iteration_results": [{"iteration": 1, "hits": {"total": 10, "included": 4}}],
            }
        ],
    }



def _store_pyhmmer_phase_prepass(store: LoopStore) -> None:
    record = EvaluationRecord(
        eval_id="eval_pyhmmer_profile",
        patchset_id="stock_pyhmmer_patchset",
        round_idx=0,
        stage="full",
        status="ok",
        decision="profile_prepass",
        eval_result={
            "stage": "full",
            "status": "ok",
            "tp": 10,
            "fp": 1,
            "tp_fp_ratio": 10 / 11,
            "cost_s": 10.0,
            "meta": {"profile": _pyhmmer_phase_profile()},
        },
        created_at=utc_now(),
    )
    store.put_evaluation(record)



def _store_profile_baseline(store: LoopStore) -> EvalSummary:
    record = EvaluationRecord(
        eval_id="eval_profile",
        patchset_id="stock_patchset",
        round_idx=0,
        stage="full",
        status="ok",
        decision="accepted_baseline",
        eval_result={
            "stage": "full",
            "status": "ok",
            "tp": 10,
            "fp": 1,
            "tp_fp_ratio": 10 / 11,
            "cost_s": 10.0,
            "meta": {"profile": _profile(jack=9.0, search=1.0, top_query="q1")},
        },
        created_at=utc_now(),
    )
    store.put_evaluation(record)
    return EvalSummary(eval_id="eval_profile", patchset_id="stock_patchset", stage="full", status="ok", tp=10, fp=1, cost_s=10.0, comparability=_comparable(stage="full"))



def _store_gprof_hotspot_baseline(store: LoopStore) -> EvalSummary:
    profile = _profile(jack=29.0, search=1.0, top_query="q_gprof")
    profile.update({
        "external_profiler": {"tool": "gprof", "status": "ok", "excluded_from_cost_s": True},
        "symbol_hotspots": [
            {
                "rank": 1,
                "command": "jackhmmer",
                "symbol": "calc_band_8",
                "dso": "gprof",
                "approx_command_wall_s": 5.3,
                "max_self_pct": 17.75,
                "profile_source": "gprof_flat_profile",
            }
        ],
    })
    record = EvaluationRecord(
        eval_id="eval_gprof_hotspot",
        patchset_id="stock_patchset",
        round_idx=0,
        stage="full",
        status="ok",
        decision="accepted_baseline",
        eval_result={
            "stage": "full",
            "status": "ok",
            "tp": 10,
            "fp": 1,
            "tp_fp_ratio": 10 / 11,
            "cost_s": 30.0,
            "meta": {"profile": profile},
        },
        created_at=utc_now(),
    )
    store.put_evaluation(record)
    return EvalSummary(
        eval_id="eval_gprof_hotspot",
        patchset_id="stock_patchset",
        stage="full",
        status="ok",
        tp=10,
        fp=1,
        cost_s=30.0,
        comparability=_comparable(stage="full"),
    )



def _store_symbol_profile_baseline(store: LoopStore) -> EvalSummary:
    profile = _profile(jack=19.0, search=1.0, top_query="q_symbol")
    profile.update({
        "external_profiler": {"tool": "perf", "status": "ok", "excluded_from_cost_s": True},
        "symbol_hotspots": [
            {
                "rank": 1,
                "command": "jackhmmer",
                "symbol": "forward_engine",
                "dso": "jackhmmer",
                "approx_command_wall_s": 8.0,
                "max_self_pct": 42.0,
                "profile_source": "perf_report_stdio",
            }
        ],
    })
    record = EvaluationRecord(
        eval_id="eval_symbol_profile",
        patchset_id="stock_patchset",
        round_idx=0,
        stage="full",
        status="ok",
        decision="accepted_baseline",
        eval_result={
            "stage": "full",
            "status": "ok",
            "tp": 10,
            "fp": 1,
            "tp_fp_ratio": 10 / 11,
            "cost_s": 20.0,
            "meta": {"profile": profile},
        },
        created_at=utc_now(),
    )
    store.put_evaluation(record)
    return EvalSummary(
        eval_id="eval_symbol_profile",
        patchset_id="stock_patchset",
        stage="full",
        status="ok",
        tp=10,
        fp=1,
        cost_s=20.0,
        comparability=_comparable(stage="full"),
    )



def test_algorithm_plan_parse_accepts_sequence_streaming_family():
    raw = json.dumps({
        "algorithm_family": "sequence_streaming_strategy",
        "baseline_algorithm_summary": "Stock parser streams target FASTA into jackhmmer.",
        "proposed_algorithm_change": "Change parser streaming in addbuf while preserving residues.",
        "speed_mechanism": "Reduce parser work while preserving target bytes.",
        "quality_preservation_argument": "TP and FP remain unchanged because residue classification and thresholds are unchanged.",
        "target_contracts": ["sequence_io"],
        "target_functions": ["addbuf"],
        "profile_evidence_citation": "No measured profile evidence is available in this context pack.",
    })

    proposal = parse_algorithm_plan_proposal(raw)

    assert proposal.algorithm_family == "sequence_streaming_strategy"
    assert proposal.target_contracts == ["sequence_io"]


@pytest.mark.parametrize(
    ("family", "contract", "function"),
    [
        ("dp_recurrence_strategy", "dp_kernel_neon", "forward_engine"),
        ("ssv_filter_strategy", "ssv_filter_neon", "calc_band_8"),
        ("null2_scoring_strategy", "null2_correction_neon", "p7_Null2_ByExpectation"),
        ("sequence_streaming_strategy", "sequence_io", "addbuf"),
    ],
)
def test_algorithm_plan_binds_expanded_family_to_active_runtime_contract(
    tmp_path: Path,
    family: str,
    contract: str,
    function: str,
):
    store = LoopStore(tmp_path)
    pack = build_context_pack(store, round_idx=1)
    plan = AlgorithmPlanProposal(
        algorithm_family=family,
        baseline_algorithm_summary="Stock runtime behavior is the quality reference.",
        proposed_algorithm_change="Remove redundant measured work while preserving exact observable behavior.",
        speed_mechanism="Reduce repeated work and retain the exact stock fallback for uncertain cases.",
        quality_preservation_argument="TP and FP remain unchanged because scores, thresholds, targets, and hit decisions are preserved.",
        target_contracts=[contract],
        target_functions=[function],
        profile_evidence_citation="No measured profile evidence is available in this context pack.",
    )

    record = PatchRunner(store).build_algorithm_plan(plan, pack, {"provider": "test"})

    assert record.validation_status == "ok"
    assert record.target_contracts == [contract]
    assert function in record.target_functions


def test_new_dp_family_rejects_legacy_sse_contract_binding(tmp_path: Path):
    store = LoopStore(tmp_path)
    pack = build_context_pack(store, round_idx=1)
    plan = AlgorithmPlanProposal(
        algorithm_family="dp_recurrence_strategy",
        baseline_algorithm_summary="Stock runtime behavior is the quality reference.",
        proposed_algorithm_change="Remove redundant recurrence work with exact fallback.",
        speed_mechanism="Reduce repeated work while preserving exact scores.",
        quality_preservation_argument="TP and FP remain unchanged because recurrence outputs are preserved.",
        target_contracts=["dp_kernel"],
        target_functions=["forward_engine"],
        profile_evidence_citation="No measured profile evidence is available in this context pack.",
    )

    record = PatchRunner(store).build_algorithm_plan(plan, pack, {"provider": "test"})

    assert record.validation_status == "rejected"
    assert any("dp_kernel_neon" in error for error in record.validation_errors)



def test_algorithm_plan_validation_rejects_implementation_only_plan(tmp_path: Path):
    store = LoopStore(tmp_path)
    pack = build_context_pack(store, round_idx=1)
    plan = AlgorithmPlanProposal(
        algorithm_family="pipeline_flow_strategy",
        baseline_algorithm_summary="Stock HMMER behavior is the quality reference.",
        proposed_algorithm_change="Cache parsed targets and replay them from memory in p7_Pipeline.",
        speed_mechanism="Reduce parser, streaming, and dataflow work with a cache while preserving target bytes.",
        quality_preservation_argument="TP and FP remain unchanged because no thresholds or hit filters change.",
        target_contracts=["pipeline_flow"],
        target_functions=["p7_evo_align_Run"],
        profile_evidence_citation="No measured profile evidence is available in this context pack.",
    )

    record = PatchRunner(store).build_algorithm_plan(plan, pack, {"provider": "test"})

    assert record.validation_status == "rejected"
    assert any("implementation-only" in error for error in record.validation_errors)



def test_algorithm_plan_repair_records_rejected_parent_lineage(
    tmp_path: Path,
):
    bad = AlgorithmPlanProposal(
        algorithm_family="pipeline_flow_strategy",
        baseline_algorithm_summary=(
            "Stock HMMER behavior is the quality reference."
        ),
        proposed_algorithm_change=(
            "Cache parsed targets and replay them from memory."
        ),
        speed_mechanism=(
            "Reduce parser and streaming work with a cache."
        ),
        quality_preservation_argument=(
            "TP and FP remain unchanged because thresholds do not change."
        ),
        target_contracts=["pipeline_flow"],
        target_functions=["p7_evo_align_Run"],
        profile_evidence_citation=(
            "No measured profile evidence is available in this context pack."
        ),
    )
    good = AlgorithmPlanProposal(
        algorithm_family="null2_scoring_strategy",
        baseline_algorithm_summary=(
            "Stock null2 composition-bias correction is the quality reference."
        ),
        proposed_algorithm_change=(
            "Use a bounded composition-bias decision surface with exact "
            "fallback for borderline scores."
        ),
        speed_mechanism=(
            "Avoid downstream scoring work for decisive candidates while "
            "the exact fallback preserves borderline scores."
        ),
        quality_preservation_argument=(
            "TP and FP remain unchanged because thresholds and borderline "
            "decisions use the exact stock path."
        ),
        target_contracts=["null2_correction_neon"],
        target_functions=["p7_Null2_ByExpectation"],
        profile_evidence_citation=(
            "No measured profile evidence is available in this context pack."
        ),
    )

    class RepairingAlgorithmProposer(BaseProposer):
        provider_name = "algorithm-plan-repair-test"

        def __init__(self):
            self.calls = 0

        def propose_algorithm_plan(
            self,
            pack,
            *,
            feedback=None,
            attempt_idx=None,
        ):
            self.calls += 1
            return (
                bad if self.calls == 1 else good,
                {"provider": self.provider_name},
            )

    store = LoopStore(tmp_path)
    pack = build_context_pack(store, round_idx=1)
    proposer = RepairingAlgorithmProposer()

    _, approved = controller._propose_valid_algorithm_plan(
        proposer,
        PatchRunner(store),
        pack,
        comparison_patchset_id="stock",
        max_repair_attempts=1,
    )

    plans = store.iter_algorithm_plans()
    rejected = next(
        item for item in plans
        if item["validation_status"] == "rejected"
    )
    assert approved.validation_status == "ok"
    assert approved.parent_algorithm_plan_id == (
        rejected["algorithm_plan_id"]
    )



def test_algorithm_plan_validation_accepts_sequence_io_outer_target(tmp_path: Path):
    store = LoopStore(tmp_path)
    _store_pyhmmer_phase_prepass(store)
    summary = _store_profile_baseline(store)
    pack = build_context_pack(store, round_idx=1, baseline=summary, current_best=summary)
    plan = AlgorithmPlanProposal(
        algorithm_family="sequence_streaming_strategy",
        baseline_algorithm_summary="Stock parser streams target FASTA into jackhmmer.",
        proposed_algorithm_change="Add a filter cascade decision surface with exact fallback after parsing.",
        speed_mechanism="Reduce downstream search work for confidently rejected candidates while fallback preserves borderline targets.",
        quality_preservation_argument="TP and FP remain unchanged because reportable/includable candidates use the exact stock path.",
        target_contracts=["sequence_io"],
        target_functions=["addbuf"],
        profile_evidence_citation="eval_pyhmmer_profile reports dominant_phase=jackhmmer_iterations and phase_wall_fraction=0.9.",
    )

    record = PatchRunner(store).build_algorithm_plan(plan, pack, {"provider": "test"})

    assert record.validation_status == "ok"
    assert record.target_contracts == ["sequence_io"]
    assert "addbuf" in record.target_functions



def test_algorithm_plan_validation_accepts_algorithm_level_profile_plan(tmp_path: Path):
    store = LoopStore(tmp_path)
    summary = _store_gprof_hotspot_baseline(store)
    pack = build_context_pack(store, round_idx=1, baseline=summary, current_best=summary)
    plan = AlgorithmPlanProposal(
        algorithm_family="ssv_filter_strategy",
        baseline_algorithm_summary="Stock SSV filter computes calc_band_8 band scores in the jackhmmer prefilter path.",
        proposed_algorithm_change="Keep the same SSV thresholds and overflow fallback, but add a filter cascade decision surface that sends borderline band scores through the exact stock validation pass.",
        speed_mechanism="Reduce downstream SSV/DP work for confidently rejected candidates while exact fallback preserves borderline filter score semantics.",
        quality_preservation_argument="TP and FP remain unchanged because the same residues, scores, thresholds, and reported-hit policy are preserved.",
        target_contracts=["ssv_filter_neon"],
        target_functions=["calc_band_8"],
        profile_evidence_citation="eval_gprof_hotspot reports dominant_command=jackhmmer and top symbol hotspot calc_band_8 mapped to ssv_filter_neon.",
        novelty_against_l2_l3="Use a new SSV band-scoring mechanism rather than repeating recent implementation-only local hoisting.",
    )

    record = PatchRunner(store).build_algorithm_plan(plan, pack, {"provider": "test"})

    assert record.validation_status == "ok"
    assert record.algorithm_plan_id
    assert (store.algorithm_plans_dir / f"{record.algorithm_plan_id}.json").exists()



def test_algorithm_plan_validation_does_not_false_positive_on_io_inside_algorithm_words(tmp_path: Path):
    store = LoopStore(tmp_path)
    pack = build_context_pack(store, round_idx=1)
    plan = AlgorithmPlanProposal(
        algorithm_family="null2_scoring_strategy",
        baseline_algorithm_summary="Stock null2 composition-bias correction is applied after profile scoring.",
        proposed_algorithm_change="Change the composition-bias score correction decision surface with a validation pass for borderline scores.",
        speed_mechanism="Reduce downstream scoring work for non-borderline candidates while exact fallback preserves score correction semantics.",
        quality_preservation_argument="TP and FP remain unchanged because thresholds and report/include decisions are unchanged for borderline candidates.",
        target_contracts=["null2_correction_neon"],
        target_functions=["p7_Null2_ByExpectation"],
        profile_evidence_citation="No measured profile evidence is available in this context pack.",
        novelty_against_l2_l3="Uses a scoring decision surface rather than implementation cleanup.",
    )

    record = PatchRunner(store).build_algorithm_plan(plan, pack, {"provider": "test"})

    assert record.validation_status == "ok"



def test_algorithm_plan_validation_requires_high_level_decision_surface(tmp_path: Path):
    store = LoopStore(tmp_path)
    pack = build_context_pack(store, round_idx=1)
    plan = AlgorithmPlanProposal(
        algorithm_family="pipeline_flow_strategy",
        baseline_algorithm_summary="Stock pipeline flow is the quality reference.",
        proposed_algorithm_change="Preserve hit outcomes while keeping thresholds unchanged.",
        speed_mechanism="Reduce runtime in p7_Pipeline without naming a filter cascade, pruning, fallback, ranking, or scoring decision surface.",
        quality_preservation_argument="TP and FP remain unchanged because thresholds and reported-hit policy are unchanged.",
        target_contracts=["pipeline_flow"],
        target_functions=["p7_evo_align_Run"],
        profile_evidence_citation="No measured profile evidence is available in this context pack.",
    )

    record = PatchRunner(store).build_algorithm_plan(plan, pack, {"provider": "test"})

    assert record.validation_status == "rejected"
    assert any("high-level HMMER decision surface" in error for error in record.validation_errors)



def test_algorithm_plan_validation_rejects_identical_l3_mechanism_despite_novelty_prose(tmp_path: Path):
    store = LoopStore(tmp_path)
    plan = AlgorithmPlanProposal(
        algorithm_family="dp_recurrence_strategy",
        baseline_algorithm_summary="Stock forward_engine computes the DP recurrence for jackhmmer scoring.",
        proposed_algorithm_change="Change the DP recurrence decision surface by using a bounded pruning frontier with exact fallback for borderline cells while preserving filter and reporting behavior.",
        speed_mechanism="Reduce expensive recurrence work for confidently pruned cells while the fallback pass preserves exact forward_engine scoring.",
        quality_preservation_argument="TP and FP remain unchanged because score semantics and thresholds are unchanged.",
        target_contracts=["dp_kernel_neon"],
        target_functions=["forward_engine"],
        profile_evidence_citation="No measured profile evidence is available in this context pack.",
        novelty_against_l2_l3="This is new and different.",
    )
    projection = algorithm_mechanism_projection(plan)
    wisdom = DurableWisdomRecord(
        wisdom_id="wisdom-exact",
        wisdom_kind="algorithm_wisdom",
        algorithm_family=plan.algorithm_family,
        component_name="dp_kernel_neon",
        target_functions=["forward_engine"],
        lesson="Previously represented algorithm mechanism.",
        mechanism_projection=projection,
        mechanism_fingerprint=mechanism_fingerprint(projection),
    )
    pack = build_context_pack(store, round_idx=1)
    pack.l3_durable_wisdom = [wisdom]

    record = PatchRunner(store).build_algorithm_plan(plan, pack, {"provider": "test"})

    assert record.validation_status == "rejected"
    assert any("l3 wisdom-exact" in error and "exact fingerprint" in error for error in record.validation_errors)



def test_algorithm_plan_validation_ignores_malformed_l2_projection_and_stale_fingerprint(tmp_path: Path):
    store = LoopStore(tmp_path)
    plan = AlgorithmPlanProposal(
        algorithm_family="dp_recurrence_strategy",
        baseline_algorithm_summary="Stock forward_engine computes the DP recurrence for jackhmmer scoring.",
        proposed_algorithm_change="Use a bounded pruning frontier with exact fallback for borderline cells.",
        speed_mechanism="Reduce recurrence work while exact fallback preserves forward scores.",
        quality_preservation_argument="TP and FP remain unchanged because score semantics and thresholds are unchanged.",
        target_contracts=["dp_kernel_neon"],
        target_functions=["forward_engine"],
        profile_evidence_citation="No measured profile evidence is available in this context pack.",
    )
    pack = build_context_pack(store, round_idx=1)
    pack.l2_phase_summaries = [PhaseSummaryRecord(
        phase_summary_id="phase-malformed",
        run_dir=str(tmp_path),
        phase_kind="full",
        accepted_speedups=[{
            "scope": "outer_algorithm",
            "mechanism_projection": {"track": "not-a-supported-track", "mechanism": ["bad-shape"]},
            "mechanism_fingerprint": mechanism_fingerprint(algorithm_mechanism_projection(plan)),
        }],
    )]

    record = PatchRunner(store).build_algorithm_plan(plan, pack, {"provider": "test"})

    assert record.validation_status == "ok"



def test_algorithm_plan_validation_uses_projection_over_disagreeing_l2_fingerprint(tmp_path: Path):
    store = LoopStore(tmp_path)
    plan = AlgorithmPlanProposal(
        algorithm_family="dp_recurrence_strategy",
        baseline_algorithm_summary="Stock forward_engine computes the DP recurrence for jackhmmer scoring.",
        proposed_algorithm_change="Use a bounded pruning frontier with exact fallback for borderline cells.",
        speed_mechanism="Reduce recurrence work while exact fallback preserves forward scores.",
        quality_preservation_argument="TP and FP remain unchanged because score semantics and thresholds are unchanged.",
        target_contracts=["dp_kernel_neon"],
        target_functions=["forward_engine"],
        profile_evidence_citation="No measured profile evidence is available in this context pack.",
    )
    different = plan.model_copy(update={
        "proposed_algorithm_change": "Reorder exact recurrence tiles without pruning any cells.",
        "speed_mechanism": "Reduce scheduler overhead while evaluating every stock recurrence cell.",
    })
    pack = build_context_pack(store, round_idx=1)
    pack.l2_phase_summaries = [PhaseSummaryRecord(
        phase_summary_id="phase-stale-fingerprint",
        run_dir=str(tmp_path),
        phase_kind="full",
        accepted_speedups=[{
            "scope": "outer_algorithm",
            "mechanism_projection": algorithm_mechanism_projection(different),
            "mechanism_fingerprint": mechanism_fingerprint(algorithm_mechanism_projection(plan)),
        }],
    )]

    record = PatchRunner(store).build_algorithm_plan(plan, pack, {"provider": "test"})

    assert record.validation_status == "ok"



def test_algorithm_plan_validation_ignores_deprecated_l3_mechanism(tmp_path: Path):
    store = LoopStore(tmp_path)
    plan = AlgorithmPlanProposal(
        algorithm_family="dp_recurrence_strategy",
        baseline_algorithm_summary="Stock forward_engine computes the DP recurrence for jackhmmer scoring.",
        proposed_algorithm_change="Use a bounded pruning frontier with exact fallback for borderline cells.",
        speed_mechanism="Reduce recurrence work while exact fallback preserves forward scores.",
        quality_preservation_argument="TP and FP remain unchanged because score semantics and thresholds are unchanged.",
        target_contracts=["dp_kernel_neon"],
        target_functions=["forward_engine"],
        profile_evidence_citation="No measured profile evidence is available in this context pack.",
    )
    projection = algorithm_mechanism_projection(plan)
    pack = build_context_pack(store, round_idx=1)
    pack.l3_durable_wisdom = [DurableWisdomRecord(
        wisdom_id="deprecated-exact",
        wisdom_kind="algorithm_wisdom",
        confidence="deprecated",
        component_name="dp_kernel_neon",
        target_functions=["forward_engine"],
        lesson="Superseded mechanism.",
        mechanism_projection=projection,
        mechanism_fingerprint=mechanism_fingerprint(projection),
    )]

    record = PatchRunner(store).build_algorithm_plan(plan, pack, {"provider": "test"})

    assert record.validation_status == "ok"



def test_algorithm_plan_novelty_match_tie_break_is_deterministic(tmp_path: Path):
    store = LoopStore(tmp_path)
    plan = AlgorithmPlanProposal(
        algorithm_family="dp_recurrence_strategy",
        baseline_algorithm_summary="Stock forward_engine computes the DP recurrence for jackhmmer scoring.",
        proposed_algorithm_change="Use a bounded pruning frontier with exact fallback for borderline cells.",
        speed_mechanism="Reduce recurrence work while exact fallback preserves forward scores.",
        quality_preservation_argument="TP and FP remain unchanged because score semantics and thresholds are unchanged.",
        target_contracts=["dp_kernel_neon"],
        target_functions=["forward_engine"],
        profile_evidence_citation="No measured profile evidence is available in this context pack.",
    )
    projection = algorithm_mechanism_projection(plan)
    summaries = [
        PhaseSummaryRecord(
            phase_summary_id=source_id,
            run_dir=str(tmp_path),
            phase_kind="full",
            accepted_speedups=[{
                "scope": "outer_algorithm",
                "mechanism_projection": projection,
                "mechanism_fingerprint": mechanism_fingerprint(projection),
            }],
        )
        for source_id in ("phase-z", "phase-a")
    ]
    errors = []
    for ordered in (summaries, list(reversed(summaries))):
        pack = build_context_pack(store, round_idx=1)
        pack.l2_phase_summaries = ordered
        record = PatchRunner(store).build_algorithm_plan(plan, pack, {"provider": "test"})
        errors.append(next(error for error in record.validation_errors if "not novel against" in error))

    assert errors[0] == errors[1]
    assert "l2 phase-a" in errors[0]



def test_patch_plan_validation_rejects_similar_l2_mechanism(tmp_path: Path):
    store = LoopStore(tmp_path)
    prior = PatchPlanProposal(
        component_name="dp_kernel_neon",
        target_functions=["forward_engine"],
        target_symbols=["forward_engine"],
        algorithm_hypothesis="Prune bounded DP frontier cells while exact fallback preserves TP and recall.",
        intended_code_change="Use a bounded pruning frontier in forward_engine and route borderline cells through exact fallback.",
        runtime_effective_argument="Skip expensive DP recurrence work for confidently pruned cells in the rebuilt runtime binary.",
        expected_metric_effect="TP and FP remain unchanged.",
        anti_degenerate_check="Exact fallback preserves score semantics and TP recall.",
    )
    proposal = prior.model_copy(update={
        "intended_code_change": "Route borderline cells through exact fallback while forward_engine uses a bounded pruning frontier.",
        "runtime_effective_argument": "In the rebuilt binary, confidently pruned cells skip expensive DP recurrence work.",
        "novelty_argument": "This wording is novel and different.",
    })
    projection = implementation_mechanism_projection(prior)
    pack = build_context_pack(store, round_idx=1)
    pack.l2_phase_summaries = [PhaseSummaryRecord(
        phase_summary_id="phase-similar",
        run_dir=str(tmp_path),
        phase_kind="full",
        accepted_speedups=[{
            "scope": "inner_implementation",
            "mechanism_projection": projection,
            "mechanism_fingerprint": mechanism_fingerprint(projection),
        }],
    )]

    record = PatchRunner(store).build_patch_plan(proposal, pack, {"provider": "test"})

    assert record.validation_status == "rejected"
    assert any("l2 phase-similar" in error and "similarity=" in error for error in record.validation_errors)



def test_algorithm_plan_validation_rejects_identical_negative_mechanism_despite_novelty_prose(tmp_path: Path):
    store = LoopStore(tmp_path)
    plan = AlgorithmPlanProposal(
        algorithm_family="dp_recurrence_strategy",
        baseline_algorithm_summary="Stock forward_engine computes the DP recurrence for jackhmmer scoring.",
        proposed_algorithm_change="Change the DP recurrence decision surface by using a bounded pruning frontier with exact fallback for borderline cells while preserving filter and reporting behavior.",
        speed_mechanism="Reduce expensive recurrence work for confidently pruned cells while the fallback pass preserves exact forward_engine scoring.",
        quality_preservation_argument="TP and FP remain unchanged because score semantics and thresholds are unchanged.",
        target_contracts=["dp_kernel_neon"],
        target_functions=["forward_engine"],
        profile_evidence_citation="No measured profile evidence is available in this context pack.",
        novelty_against_l2_l3="This is a new and different mechanism.",
    )
    store.write_index("experience_recent_negatives", [{
        "scope": "outer_algorithm",
        "outcome": "rejected_not_faster",
        "component_name": "dp_kernel_neon",
        "target_functions": ["forward_engine"],
        "mechanism_fingerprint": mechanism_fingerprint(algorithm_mechanism_projection(plan)),
        "lesson": "Recent forward_engine algorithm plan preserved quality but was slower.",
    }])
    pack = build_context_pack(store, round_idx=2)

    record = PatchRunner(store).build_algorithm_plan(plan, pack, {"provider": "test"})

    assert record.validation_status == "rejected"
    assert any("identical fingerprint" in error for error in record.validation_errors)



def test_patch_plan_validation_rejects_noop_plan(tmp_path: Path):
    store = LoopStore(tmp_path)
    pack = build_context_pack(store, round_idx=1)
    plan_proposal = parse_patch_plan_proposal(json.dumps({
        "component_name": "null2_correction_neon",
        "target_functions": ["p7_Null2_ByExpectation"],
        "algorithm_hypothesis": "Insert a no-op marker.",
        "intended_code_change": "Add unreachable no-op code.",
        "runtime_effective_argument": "This runtime branch should affect the binary.",
        "anti_degenerate_check": "TP/recall should stay unchanged.",
    }))
    runner = PatchRunner(store)

    plan_record = runner.build_patch_plan(plan_proposal, pack, {"provider": "test"})

    assert plan_record.validation_status == "rejected"
    assert "no-op" in plan_record.validation_errors[0]



def test_patch_plan_validation_requires_profile_citation_when_profile_exists(tmp_path: Path):
    store = LoopStore(tmp_path)
    summary = _store_profile_baseline(store)
    pack = build_context_pack(store, round_idx=1, baseline=summary, current_best=summary)
    plan = PatchPlanProposal(
        component_name="null2_correction_neon",
        target_functions=["p7_Null2_ByExpectation"],
        algorithm_hypothesis="Remove redundant runtime null2 work while preserving TP/recall.",
        intended_code_change="Change executable C arithmetic in p7_Null2_ByExpectation without altering thresholds.",
        runtime_effective_argument="The runtime arithmetic changes rebuilt jackhmmer/hmmsearch binary executable code.",
        anti_degenerate_check="TP/recall remains unchanged because hit filters and thresholds are unchanged.",
    )

    record = PatchRunner(store).build_patch_plan(plan, pack, {"provider": "test"})

    assert record.validation_status == "rejected"
    assert any("profile_evidence_citation" in error for error in record.validation_errors)



def test_patch_plan_validation_counts_expected_metric_tp_preservation(tmp_path: Path):
    store = LoopStore(tmp_path)
    summary = _store_profile_baseline(store)
    pack = build_context_pack(store, round_idx=1, baseline=summary, current_best=summary)
    plan = PatchPlanProposal(
        component_name="sequence_io",
        target_functions=["seebuf"],
        target_symbols=["seebuf"],
        algorithm_hypothesis="Cache seebuf parser loads in the jackhmmer FASTA path.",
        intended_code_change="Change executable C pointer loads in seebuf without altering classification branches.",
        runtime_effective_argument="The runtime loop change alters rebuilt jackhmmer/hmmsearch binary executable code.",
        profile_evidence_citation="eval_profile reports dominant_command=jackhmmer wall_fraction=0.9 and measured symbol hotspot seebuf mapped to sequence_io.",
        expected_metric_effect="TP and FP are preserved exactly because parser classifications and hit filters are unchanged.",
        anti_degenerate_check="This is not a no-op.",
    )

    record = PatchRunner(store).build_patch_plan(plan, pack, {"provider": "test"})

    assert record.validation_status == "ok"



def test_patch_plan_validation_accepts_profile_cited_jackhmmer_plan(tmp_path: Path):
    store = LoopStore(tmp_path)
    summary = _store_profile_baseline(store)
    pack = build_context_pack(store, round_idx=1, baseline=summary, current_best=summary)
    plan = PatchPlanProposal(
        component_name="null2_correction_neon",
        target_functions=["p7_Null2_ByExpectation"],
        algorithm_hypothesis="Remove redundant runtime null2 work in the jackhmmer scoring path while preserving TP/recall.",
        intended_code_change="Change executable C arithmetic in p7_Null2_ByExpectation without altering thresholds.",
        runtime_effective_argument="The runtime arithmetic changes rebuilt jackhmmer/hmmsearch binary executable code.",
        profile_evidence_citation="eval_profile reports dominant_command=jackhmmer wall_fraction=0.9 top_slowest_query=q1.",
        anti_degenerate_check="TP/recall remains unchanged because hit filters and thresholds are unchanged.",
    )

    record = PatchRunner(store).build_patch_plan(plan, pack, {"provider": "test"})

    assert record.validation_status == "ok"
    assert record.profile_evidence_citation.startswith("eval_profile")



def test_patch_plan_validation_requires_symbol_hotspot_when_available(tmp_path: Path):
    store = LoopStore(tmp_path)
    summary = _store_symbol_profile_baseline(store)
    pack = build_context_pack(store, round_idx=1, baseline=summary, current_best=summary)
    plan = PatchPlanProposal(
        component_name="dp_kernel_neon",
        target_functions=["forward_engine"],
        algorithm_hypothesis="Simplify forward_engine runtime arithmetic in the jackhmmer DP path while preserving TP/recall.",
        intended_code_change="Change executable C arithmetic in forward_engine without altering thresholds or score semantics.",
        runtime_effective_argument="The runtime arithmetic changes rebuilt jackhmmer/hmmsearch binary executable code.",
        profile_evidence_citation="eval_symbol_profile reports dominant_command=jackhmmer wall_fraction=0.95 top_slowest_query=q_symbol.",
        anti_degenerate_check="TP/recall remains unchanged because hit filters and thresholds are unchanged.",
    )

    record = PatchRunner(store).build_patch_plan(plan, pack, {"provider": "test"})

    assert record.validation_status == "rejected"
    assert any("symbol/function/contract hotspot" in error for error in record.validation_errors)



def test_patch_plan_validation_accepts_calc_band_gprof_contract(tmp_path: Path):
    store = LoopStore(tmp_path)
    summary = _store_gprof_hotspot_baseline(store)
    pack = build_context_pack(store, round_idx=1, baseline=summary, current_best=summary)
    plan = PatchPlanProposal(
        component_name="ssv_filter_neon",
        target_functions=["calc_band_8"],
        target_symbols=["calc_band_8"],
        profile_evidence_refs=["eval_gprof_hotspot:calc_band_8"],
        algorithm_hypothesis="Simplify measured calc_band_8 vector-band runtime work in the jackhmmer SSV path while preserving TP/recall.",
        intended_code_change="Change executable C arithmetic in calc_band_8 without altering thresholds or filter score semantics.",
        runtime_effective_argument="The runtime arithmetic changes rebuilt jackhmmer/hmmsearch binary executable code in the measured SSV filter path.",
        profile_evidence_citation="eval_gprof_hotspot reports dominant_command=jackhmmer and top symbol hotspot calc_band_8 mapped to ssv_filter_neon.",
        anti_degenerate_check="TP/recall remains unchanged because hit filters, thresholds, overflow fallback, and score semantics are unchanged.",
    )

    record = PatchRunner(store).build_patch_plan(plan, pack, {"provider": "test"})

    assert record.validation_status == "ok"
    assert record.component_name == "ssv_filter_neon"
    assert record.c_target_functions == ["calc_band_8"]



def test_patch_plan_validation_accepts_symbol_cited_matching_contract(tmp_path: Path):
    store = LoopStore(tmp_path)
    summary = _store_symbol_profile_baseline(store)
    pack = build_context_pack(store, round_idx=1, baseline=summary, current_best=summary)
    plan = PatchPlanProposal(
        component_name="dp_kernel_neon",
        target_functions=["forward_engine"],
        target_symbols=["forward_engine"],
        profile_evidence_refs=["eval_symbol_profile:forward_engine"],
        algorithm_hypothesis="Simplify measured forward_engine runtime arithmetic in the jackhmmer DP path while preserving TP/recall.",
        intended_code_change="Change executable C arithmetic in forward_engine without altering thresholds or score semantics.",
        runtime_effective_argument="The runtime arithmetic changes rebuilt jackhmmer/hmmsearch binary executable code.",
        profile_evidence_citation="eval_symbol_profile reports dominant_command=jackhmmer wall_fraction=0.95 and top symbol hotspot forward_engine mapped to dp_kernel_neon.",
        anti_degenerate_check="TP/recall remains unchanged because hit filters and thresholds are unchanged.",
    )

    record = PatchRunner(store).build_patch_plan(plan, pack, {"provider": "test"})

    assert record.validation_status == "ok"
    assert record.target_symbols == ["forward_engine"]



def test_patch_plan_validation_rejects_symbol_hotspot_contract_mismatch(tmp_path: Path):
    store = LoopStore(tmp_path)
    summary = _store_symbol_profile_baseline(store)
    pack = build_context_pack(store, round_idx=1, baseline=summary, current_best=summary)
    plan = PatchPlanProposal(
        component_name="null2_correction_neon",
        target_functions=["p7_Null2_ByExpectation"],
        target_symbols=["forward_engine"],
        algorithm_hypothesis="Change null2 runtime arithmetic while preserving TP/recall.",
        intended_code_change="Change executable C arithmetic in p7_Null2_ByExpectation without altering thresholds.",
        runtime_effective_argument="The runtime arithmetic changes rebuilt jackhmmer/hmmsearch binary executable code.",
        profile_evidence_citation="eval_symbol_profile reports dominant_command=jackhmmer wall_fraction=0.95 and top symbol hotspot forward_engine mapped to dp_kernel_neon.",
        anti_degenerate_check="TP/recall remains unchanged because hit filters and thresholds are unchanged.",
    )

    record = PatchRunner(store).build_patch_plan(plan, pack, {"provider": "test"})

    assert record.validation_status == "rejected"
    assert any("maps top hotspot" in error for error in record.validation_errors)



def test_patch_plan_validation_rejects_non_dominant_command_only_plan(tmp_path: Path):
    store = LoopStore(tmp_path)
    summary = _store_profile_baseline(store)
    pack = build_context_pack(store, round_idx=1, baseline=summary, current_best=summary)
    plan = PatchPlanProposal(
        component_name="null2_correction_neon",
        target_functions=["p7_Null2_ByExpectation"],
        algorithm_hypothesis="Target hmmsearch-only post-processing while preserving TP/recall.",
        intended_code_change="Change executable C arithmetic in p7_Null2_ByExpectation for only hmmsearch.",
        runtime_effective_argument="The runtime arithmetic changes rebuilt jackhmmer/hmmsearch binary executable code.",
        profile_evidence_citation="eval_profile reports dominant_command=jackhmmer wall_fraction=0.9.",
        anti_degenerate_check="TP/recall remains unchanged because hit filters and thresholds are unchanged.",
    )

    record = PatchRunner(store).build_patch_plan(plan, pack, {"provider": "test"})

    assert record.validation_status == "rejected"
    assert any("non-dominant command" in error for error in record.validation_errors)



def test_patch_plan_validation_accepts_sequence_io_as_secondary_iteration_hot_path(tmp_path: Path):
    store = LoopStore(tmp_path)
    _store_pyhmmer_phase_prepass(store)
    summary = _store_profile_baseline(store)
    pack = build_context_pack(store, round_idx=1, baseline=summary, current_best=summary)
    plan = PatchPlanProposal(
        component_name="sequence_io",
        target_functions=["addbuf"],
        algorithm_hypothesis="Reduce target parser work while preserving TP and FP.",
        intended_code_change="Change executable C parser logic in addbuf while preserving residue classification.",
        runtime_effective_argument="The rebuilt jackhmmer/hmmsearch executable parser code changes at runtime.",
        profile_evidence_citation="eval_pyhmmer_profile reports dominant_phase=jackhmmer_iterations and phase_wall_fraction=0.9.",
        anti_degenerate_check="TP/recall remains unchanged because sequence bytes and thresholds are unchanged.",
    )

    record = PatchRunner(store).build_patch_plan(plan, pack, {"provider": "test"})

    assert record.validation_status == "ok"
    assert record.component_name == "sequence_io"



def test_patch_plan_validation_allows_negated_non_dominant_command_mention(tmp_path: Path):
    store = LoopStore(tmp_path)
    summary = _store_profile_baseline(store)
    pack = build_context_pack(store, round_idx=1, baseline=summary, current_best=summary)
    plan = PatchPlanProposal(
        component_name="null2_correction_neon",
        target_functions=["p7_Null2_ByExpectation"],
        algorithm_hypothesis="Remove redundant runtime null2 work in the jackhmmer scoring path while preserving TP/recall.",
        intended_code_change="Change executable C arithmetic in p7_Null2_ByExpectation without altering thresholds.",
        runtime_effective_argument="The runtime arithmetic changes rebuilt jackhmmer/hmmsearch binary executable code.",
        profile_evidence_citation=(
            "eval_profile reports dominant_command=jackhmmer wall_fraction=0.9; "
            "the selected function is a jackhmmer scoring path, not a hmmsearch-only path."
        ),
        anti_degenerate_check="TP/recall remains unchanged because hit filters and thresholds are unchanged.",
    )

    record = PatchRunner(store).build_patch_plan(plan, pack, {"provider": "test"})

    assert record.validation_status == "ok"



@pytest.mark.parametrize("anti_degenerate_check", [
    "The patch is not a comment, no-op, or unreachable marker; TP/recall remains unchanged.",
    "This is not a comment/no-op/unreachable change; it changes executable runtime C logic and preserves TP/recall.",
    "No no-op marker is added; the hot loop changes object code and preserves TP/recall.",
    "Do not add comment-only edits; the executable C arithmetic changes and preserves TP/recall.",
    "A patch containing only comments, unused variables, unreachable conditionals, or code optimized away would not satisfy this plan; TP/recall remains unchanged.",
    "Recent attempts failed validation because they appeared to rely on comments/no-ops/unreachable logic, so this executable C loop change preserves TP/recall.",
    "This differs from failed no-op ssv_filter plans by changing executable scheduling logic while preserving TP/recall.",
    "Reject the C diff if it only adds comments, hoists unused locals, adds unreachable code, or changes thresholds; this executable C loop change preserves TP/recall.",
    "A patch consisting only of comments, variable renames, unreachable code, or unused locals is explicitly not this plan; TP/recall remains unchanged.",
    "This differs from a pure local-variable hoisting no-op by changing the replay algorithm while preserving TP/recall.",
    "If the generated diff only adds comments, annotations, unused variables, unreachable code, or leaves the runtime path intact, it is invalid; TP/recall is preserved.",
    "A comment-only change, unused variable, or semantically unreachable branch is not acceptable; the runtime control-flow change preserves TP/recall.",
    "A valid implementation cannot consist only of comments, unused variables, unreachable branches, or final-result-preserving no-ops; TP/recall remains unchanged.",
    "Recent plan-validation failures were caused by plans that appeared comment-only/no-op or not executable; this repaired plan specifies runtime C logic and preserves TP/recall.",
    "A comment-only change, variable-only hoist, unreachable code block, or retained indirect-call scheduler would fail this check; TP/recall remains unchanged.",
    "The default path should be unreachable for valid partitioning but does not alter normal execution; TP/recall remains unchanged.",
])

def test_patch_plan_validation_allows_negated_degenerate_marker_mention(tmp_path: Path, anti_degenerate_check: str):
    store = LoopStore(tmp_path)
    pack = build_context_pack(store, round_idx=1)
    plan = PatchPlanProposal(
        component_name="dp_kernel_neon",
        target_functions=["forward_engine"],
        algorithm_hypothesis="Reuse one runtime reciprocal in forward_engine while preserving TP/recall.",
        intended_code_change="Change executable C arithmetic in the sparse rescaling block without altering thresholds.",
        runtime_effective_argument="The executable runtime arithmetic changes rebuilt jackhmmer/hmmsearch binary code.",
        profile_evidence_citation="No measured profile evidence is available in this context pack.",
        anti_degenerate_check=anti_degenerate_check,
    )

    record = PatchRunner(store).build_patch_plan(plan, pack, {"provider": "test"})

    assert record.validation_status == "ok"



def test_patch_plan_validation_rejects_identical_negative_mechanism(tmp_path: Path):
    store = LoopStore(tmp_path)
    plan = PatchPlanProposal(
        component_name="dp_kernel_neon",
        target_functions=["forward_engine"],
        algorithm_hypothesis="Reuse arithmetic in forward_engine while preserving TP/recall.",
        intended_code_change="Change executable C arithmetic in forward_engine without altering thresholds.",
        runtime_effective_argument="The executable runtime arithmetic changes rebuilt jackhmmer/hmmsearch binary code.",
        profile_evidence_citation="No measured profile evidence is available in this context pack.",
        novelty_argument="This is new and different.",
        anti_degenerate_check="TP/recall remains unchanged because hit filters and thresholds are unchanged.",
    )
    store.write_index("experience_recent_negatives", [{
        "scope": "inner_implementation",
        "outcome": "rejected_not_faster",
        "component_name": "dp_kernel_neon",
        "target_functions": ["forward_engine"],
        "mechanism_fingerprint": mechanism_fingerprint(implementation_mechanism_projection(plan)),
        "lesson": "Recent forward_engine patch preserved quality but was slower.",
    }])
    pack = build_context_pack(store, round_idx=2)

    record = PatchRunner(store).build_patch_plan(plan, pack, {"provider": "test"})

    assert record.validation_status == "rejected"
    assert any("identical fingerprint" in error for error in record.validation_errors)



def test_patch_plan_validation_rejects_inner_component_outside_outer_algorithm(tmp_path: Path):
    store = LoopStore(tmp_path)
    pack = build_context_pack(store, round_idx=1)
    outer = AlgorithmPlanProposal(
        algorithm_family="pipeline_flow_strategy",
        baseline_algorithm_summary="Stock pipeline filter cascade is the quality reference.",
        proposed_algorithm_change="Add a candidate-pruning decision surface with exact fallback for borderline targets.",
        speed_mechanism="Reduce downstream search work for confidently rejected candidates while fallback preserves borderline scoring.",
        quality_preservation_argument="TP and FP remain unchanged because reportable/includable candidates use the exact stock path.",
        target_contracts=["pipeline_flow"],
        target_functions=["p7_evo_align_Run"],
        profile_evidence_citation="No measured profile evidence is available in this context pack.",
    )
    outer_record = PatchRunner(store).build_algorithm_plan(outer, pack, {"provider": "test"})
    plan = PatchPlanProposal(
        component_name="sequence_io",
        target_functions=["addbuf"],
        algorithm_hypothesis="Implement the approved candidate pruning with parser cache support while preserving TP/recall.",
        intended_code_change="Change executable parser C logic in addbuf.",
        runtime_effective_argument="The rebuilt executable parser path changes at runtime.",
        profile_evidence_citation="No measured profile evidence is available in this context pack.",
        anti_degenerate_check="TP/recall remains unchanged because thresholds and hit filters are unchanged.",
    )

    record = PatchRunner(store).build_patch_plan(plan, pack, {"provider": "test"}, algorithm_plan=outer_record)

    assert record.validation_status == "rejected"
    assert any("not in approved algorithm plan target_contracts" in error for error in record.validation_errors)



def test_patch_plan_validation_rejects_inner_function_outside_outer_algorithm(tmp_path: Path):
    store = LoopStore(tmp_path)
    pack = build_context_pack(store, round_idx=1)
    outer = AlgorithmPlanProposal(
        algorithm_family="pipeline_flow_strategy",
        baseline_algorithm_summary="Stock pipeline filter cascade is the quality reference.",
        proposed_algorithm_change="Add a candidate-pruning decision surface with exact fallback for borderline targets.",
        speed_mechanism="Reduce downstream search work for confidently rejected candidates while fallback preserves borderline scoring.",
        quality_preservation_argument="TP and FP remain unchanged because reportable/includable candidates use the exact stock path.",
        target_contracts=["pipeline_flow"],
        target_functions=["p7_evo_align_Run"],
        profile_evidence_citation="No measured profile evidence is available in this context pack.",
    )
    outer_record = PatchRunner(store).build_algorithm_plan(outer, pack, {"provider": "test"})
    plan = PatchPlanProposal(
        component_name="pipeline_flow",
        target_functions=["p7_pli_NewSeq"],
        algorithm_hypothesis="Implement the approved pruning policy while preserving TP/recall.",
        intended_code_change="Change executable C logic in p7_pli_NewSeq.",
        runtime_effective_argument="The rebuilt executable pipeline path changes at runtime.",
        profile_evidence_citation="No measured profile evidence is available in this context pack.",
        anti_degenerate_check="TP/recall remains unchanged because thresholds and hit filters are unchanged.",
    )

    record = PatchRunner(store).build_patch_plan(plan, pack, {"provider": "test"}, algorithm_plan=outer_record)

    assert record.validation_status == "rejected"
    assert any("outside approved algorithm plan target_functions" in error for error in record.validation_errors)



def test_patch_plan_validation_accepts_inner_matching_outer_algorithm(tmp_path: Path):
    store = LoopStore(tmp_path)
    pack = build_context_pack(store, round_idx=1)
    outer = AlgorithmPlanProposal(
        algorithm_family="pipeline_flow_strategy",
        baseline_algorithm_summary="Stock pipeline filter cascade is the quality reference.",
        proposed_algorithm_change="Add a candidate-pruning decision surface with exact fallback for borderline targets.",
        speed_mechanism="Reduce downstream search work for confidently rejected candidates while fallback preserves borderline scoring.",
        quality_preservation_argument="TP and FP remain unchanged because reportable/includable candidates use the exact stock path.",
        target_contracts=["pipeline_flow"],
        target_functions=["p7_evo_align_Run"],
        profile_evidence_citation="No measured profile evidence is available in this context pack.",
    )
    outer_record = PatchRunner(store).build_algorithm_plan(outer, pack, {"provider": "test"})
    plan = PatchPlanProposal(
        component_name="pipeline_flow",
        target_functions=["p7_evo_align_Run"],
        algorithm_hypothesis="Implement the approved candidate-pruning decision surface while preserving TP/recall.",
        intended_code_change="Change executable C branch logic in p7_evo_align_Run without changing thresholds.",
        runtime_effective_argument="The rebuilt executable pipeline path changes at runtime.",
        profile_evidence_citation="No measured profile evidence is available in this context pack.",
        anti_degenerate_check="TP/recall remains unchanged because fallback preserves stock reported-hit behavior.",
    )

    record = PatchRunner(store).build_patch_plan(plan, pack, {"provider": "test"}, algorithm_plan=outer_record)

    assert record.validation_status == "ok"



def test_patch_plan_validation_rejects_rejected_outer_algorithm(tmp_path: Path):
    store = LoopStore(tmp_path)
    pack = build_context_pack(store, round_idx=1)
    outer = AlgorithmPlanProposal(
        algorithm_family="pipeline_flow_strategy",
        baseline_algorithm_summary="Stock pipeline flow is the quality reference.",
        proposed_algorithm_change="Cache parsed targets and replay them from memory.",
        speed_mechanism="Reduce parser and streaming work with a cache.",
        quality_preservation_argument="TP and FP remain unchanged because thresholds and hit filters are unchanged.",
        target_contracts=["pipeline_flow"],
        target_functions=["p7_evo_align_Run"],
        profile_evidence_citation="No measured profile evidence is available in this context pack.",
    )
    outer_record = PatchRunner(store).build_algorithm_plan(outer, pack, {"provider": "test"})
    plan = PatchPlanProposal(
        component_name="pipeline_flow",
        target_functions=["p7_evo_align_Run"],
        algorithm_hypothesis="Implement the rejected cache idea while preserving TP/recall.",
        intended_code_change="Change executable C logic in p7_evo_align_Run.",
        runtime_effective_argument="The rebuilt executable pipeline path changes at runtime.",
        profile_evidence_citation="No measured profile evidence is available in this context pack.",
        anti_degenerate_check="TP/recall remains unchanged because thresholds and hit filters are unchanged.",
    )

    record = PatchRunner(store).build_patch_plan(plan, pack, {"provider": "test"}, algorithm_plan=outer_record)

    assert outer_record.validation_status == "rejected"
    assert record.validation_status == "rejected"
    assert any("rejected outer algorithm plan" in error for error in record.validation_errors)



def test_patch_plan_validation_rejects_unknown_related_experience(tmp_path: Path):
    store = LoopStore(tmp_path)
    pack = build_context_pack(store, round_idx=2)
    plan = PatchPlanProposal(
        component_name="dp_kernel_neon",
        target_functions=["forward_engine"],
        algorithm_hypothesis="Reuse arithmetic in forward_engine while preserving TP/recall.",
        intended_code_change="Hoist one invariant reciprocal outside the sparse rescaling loop without altering thresholds.",
        runtime_effective_argument="The executable runtime arithmetic changes rebuilt jackhmmer/hmmsearch binary code.",
        profile_evidence_citation="No measured profile evidence is available in this context pack.",
        anti_degenerate_check="TP/recall remains unchanged because hit filters and thresholds are unchanged.",
        related_experience_ids=["missing-evidence"],
    )

    record = PatchRunner(store).build_patch_plan(plan, pack, {"provider": "test"})

    assert record.validation_status == "rejected"
    assert "related_experience_id does not exist: missing-evidence" in record.validation_errors



def test_patch_plan_validation_rejects_wrong_scope_related_experience(tmp_path: Path):
    store = LoopStore(tmp_path)
    evidence = ExperienceRecord(
        experience_id="outer-evidence",
        scope="outer_algorithm",
        polarity="negative",
        component_name="dp_kernel_neon",
        outcome="rejected_not_faster",
        lesson="Outer recurrence idea was slower.",
        patch_signature={"target_functions": ["forward_engine"]},
    )
    store.append_experience(evidence)
    pack = build_context_pack(store, round_idx=2)
    plan = PatchPlanProposal(
        component_name="dp_kernel_neon",
        target_functions=["forward_engine"],
        algorithm_hypothesis="Reuse arithmetic in forward_engine while preserving TP/recall.",
        intended_code_change="Hoist one invariant reciprocal outside the sparse rescaling loop without altering thresholds.",
        runtime_effective_argument="The executable runtime arithmetic changes rebuilt jackhmmer/hmmsearch binary code.",
        profile_evidence_citation="No measured profile evidence is available in this context pack.",
        anti_degenerate_check="TP/recall remains unchanged because hit filters and thresholds are unchanged.",
        related_experience_ids=["outer-evidence"],
    )

    record = PatchRunner(store).build_patch_plan(plan, pack, {"provider": "test"})

    assert record.validation_status == "rejected"
    assert any("expected 'inner_implementation'" in error for error in record.validation_errors)



def test_patch_plan_validation_rejects_related_experience_target_mismatch(tmp_path: Path):
    store = LoopStore(tmp_path)
    evidence = ExperienceRecord(
        experience_id="other-target-evidence",
        scope="inner_implementation",
        polarity="negative",
        component_name="null2_correction_neon",
        outcome="rejected_not_faster",
        lesson="A different target was slower.",
        patch_signature={"target_functions": ["p7_Null2_ByExpectation"]},
    )
    store.append_experience(evidence)
    pack = build_context_pack(store, round_idx=2)
    plan = PatchPlanProposal(
        component_name="dp_kernel_neon",
        target_functions=["forward_engine"],
        algorithm_hypothesis="Reuse arithmetic in forward_engine while preserving TP/recall.",
        intended_code_change="Hoist one invariant reciprocal outside the sparse rescaling loop without altering thresholds.",
        runtime_effective_argument="The executable runtime arithmetic changes rebuilt jackhmmer/hmmsearch binary code.",
        profile_evidence_citation="No measured profile evidence is available in this context pack.",
        anti_degenerate_check="TP/recall remains unchanged because hit filters and thresholds are unchanged.",
        related_experience_ids=["other-target-evidence"],
    )

    record = PatchRunner(store).build_patch_plan(plan, pack, {"provider": "test"})

    assert record.validation_status == "rejected"
    assert any("does not overlap the proposed component/contract and functions" in error for error in record.validation_errors)



def test_patch_plan_validation_rejects_legacy_negative_without_fingerprint(tmp_path: Path):
    store = LoopStore(tmp_path)
    plan = PatchPlanProposal(
        component_name="dp_kernel_neon",
        target_functions=["forward_engine"],
        algorithm_hypothesis="Reuse arithmetic in forward_engine while preserving TP/recall.",
        intended_code_change="Hoist one invariant reciprocal outside the sparse rescaling loop without altering thresholds.",
        runtime_effective_argument="The executable runtime arithmetic changes rebuilt jackhmmer/hmmsearch binary code.",
        profile_evidence_citation="No measured profile evidence is available in this context pack.",
        anti_degenerate_check="TP/recall remains unchanged because hit filters and thresholds are unchanged.",
    )
    store.write_index("experience_recent_negatives", [{
        "scope": "inner_implementation",
        "outcome": "rejected_not_faster",
        "component_name": "dp_kernel_neon",
        "target_functions": ["forward_engine"],
        "lesson": "Legacy same-target evidence has no comparable mechanism identity.",
    }])
    pack = build_context_pack(store, round_idx=2)

    record = PatchRunner(store).build_patch_plan(plan, pack, {"provider": "test"})

    assert record.validation_status == "rejected"
    assert any("legacy negative experience" in error and "without a comparable mechanism fingerprint" in error for error in record.validation_errors)



def test_patch_plan_validation_accepts_structurally_different_negative_mechanism(tmp_path: Path):
    store = LoopStore(tmp_path)
    plan = PatchPlanProposal(
        component_name="dp_kernel_neon",
        target_functions=["forward_engine"],
        algorithm_hypothesis="Reuse arithmetic in forward_engine while preserving TP/recall.",
        intended_code_change="Hoist one invariant reciprocal outside the sparse rescaling loop without altering thresholds.",
        runtime_effective_argument="The executable runtime arithmetic changes rebuilt jackhmmer/hmmsearch binary code.",
        profile_evidence_citation="No measured profile evidence is available in this context pack.",
        anti_degenerate_check="TP/recall remains unchanged because hit filters and thresholds are unchanged.",
    )
    prior = plan.model_copy(update={
        "intended_code_change": "Replace repeated table indexing with one local cached load without altering thresholds.",
    })
    store.write_index("experience_recent_negatives", [{
        "scope": "inner_implementation",
        "outcome": "rejected_not_faster",
        "component_name": "dp_kernel_neon",
        "target_functions": ["forward_engine"],
        "mechanism_fingerprint": mechanism_fingerprint(implementation_mechanism_projection(prior)),
        "lesson": "Recent forward_engine patch preserved quality but was slower.",
    }])
    pack = build_context_pack(store, round_idx=2)

    record = PatchRunner(store).build_patch_plan(plan, pack, {"provider": "test"})

    assert record.validation_status == "ok"



def test_openai_proposer_prefinish_repairs_parse_errors():
    class PrefinishProposer(OpenAICompatibleProposer):
        def __init__(self):
            super().__init__(api_key="test-key", prefinish_repair_attempts=1)
            self.prompts = []

        def _chat_json(self, prompt: str, *, attempt_kind: str, attempt_idx=None, system_content: str):
            self.prompts.append(prompt)
            if len(self.prompts) == 1:
                raw = json.dumps({
                    "component_name": "null2_correction",
                    "target_functions": ["p7_Null2_ByExpectation"],
                    "algorithm_hypothesis": "bad",
                    "intended_code_change": "bad",
                    "runtime_effective_argument": "bad",
                    "unified_diff": "--- a/src/impl_sse/null2.c\n+++ b/src/impl_sse/null2.c\n",
                })
            else:
                raw = json.dumps({
                    "component_name": "null2_correction",
                    "target_functions": ["p7_Null2_ByExpectation"],
                    "algorithm_hypothesis": "Adjust runtime null2 odds with a bounded data-dependent correction.",
                    "intended_code_change": "Change executable C logic in p7_Null2_ByExpectation before degeneracy averaging.",
                    "runtime_effective_argument": "The runtime branch and arithmetic alter rebuilt binary executable code.",
                    "anti_degenerate_check": "Preserve TP/recall by avoiding hit filtering or threshold changes.",
                })
            return raw, {"provider": self.provider_name, "attempt_kind": attempt_kind}

    proposal, metadata = PrefinishProposer()._complete_plan("PLAN", attempt_kind="plan")

    assert proposal.component_name == "null2_correction"
    assert metadata["prefinish_repair_attempts_used"] == 1
    assert "unified_diff" in metadata["prefinish_errors"][0]


def test_openai_proposer_repairs_missing_algorithm_fields_with_detailed_schema_feedback():
    class MissingFieldsProposer(OpenAICompatibleProposer):
        def __init__(self):
            super().__init__(api_key="test-key", prefinish_repair_attempts=1)
            self.prompts = []

        def _chat_json(self, prompt: str, *, attempt_kind: str, attempt_idx=None, system_content: str):
            self.prompts.append(prompt)
            if len(self.prompts) == 1:
                return json.dumps({"risk_notes": "missing required fields"}), {"provider": self.provider_name}
            return json.dumps({
                "algorithm_family": "profile_target_comparison",
                "baseline_algorithm_summary": "Stock HMMER runs the exact profile-target comparison.",
                "proposed_algorithm_change": "Use a certified bound with an exact fallback.",
                "speed_mechanism": "Skip recurrence work only after a safe proof.",
                "quality_preservation_argument": "Every uncertain comparison uses the stock exact path.",
            }), {"provider": self.provider_name}

    proposer = MissingFieldsProposer()
    proposal, metadata = proposer._complete_algorithm_plan("OUTER REQUEST", attempt_kind="algorithm_plan")

    repair_prompt_text = proposer.prompts[1]
    assert proposal.algorithm_family == "profile_target_comparison"
    assert metadata["prefinish_repair_attempts_used"] == 1
    assert "DETERMINISTIC_VALIDATION_ERRORS" in repair_prompt_text
    assert "algorithm_family" in repair_prompt_text
    assert "baseline_algorithm_summary" in repair_prompt_text
    assert "proposed_algorithm_change" in repair_prompt_text
    assert "speed_mechanism" in repair_prompt_text
    assert "quality_preservation_argument" in repair_prompt_text
    assert "REQUIRED_OUTPUT_JSON_SCHEMA" in repair_prompt_text
    assert '"required"' in repair_prompt_text
    assert '"risk_notes": "missing required fields"' in repair_prompt_text
    assert "OUTER REQUEST" in repair_prompt_text


def test_model_balance_and_access_errors_are_fatal_across_outer_rounds():
    class StatusError(Exception):
        def __init__(self, *, status_code, code):
            self.status_code = status_code
            self.code = code

    assert _is_fatal_model_status(StatusError(status_code=403, code="INSUFFICIENT_BALANCE"))
    assert _is_fatal_model_status(StatusError(status_code=402, code=None))
    assert _is_fatal_model_status(StatusError(status_code=400, code="insufficient_quota"))
    assert not _is_fatal_model_status(StatusError(status_code=429, code="rate_limit_exceeded"))
    assert not _is_fatal_model_status(StatusError(status_code=500, code="upstream_error"))


def test_openai_proposer_persists_sanitized_prefinish_traces(tmp_path: Path):
    store = LoopStore(tmp_path)

    class PrefinishProposer(OpenAICompatibleProposer):
        def __init__(self):
            super().__init__(api_key="test-key", trace_store=store, prefinish_repair_attempts=1)
            self.calls = 0

        def _chat_json(self, prompt: str, *, attempt_kind: str, attempt_idx=None, system_content: str):
            self.calls += 1
            if self.calls == 1:
                fake_key = "s" + "k-TEST_ONLY_NOT_A_REAL_SECRET"
                return f"Authorization: Bearer {fake_key}", {"usage": {"prompt_tokens": 1}}
            return json.dumps({
                "component_name": "null2_correction",
                "target_functions": ["p7_Null2_ByExpectation"],
                "algorithm_hypothesis": "Bounded runtime change.",
                "intended_code_change": "Change executable null2 arithmetic.",
                "runtime_effective_argument": "The rebuilt binary executes the arithmetic.",
                "anti_degenerate_check": "Preserve TP and FP.",
            }), {"usage": {"completion_tokens": 2}}

    proposer = PrefinishProposer()
    proposer._trace_context = {"stage": "inner", "memory_references": ["eval-1"], "artifact_references": ["ctx-1"]}
    proposal, _ = proposer._complete_plan(
        'token=secret-value json={"password": "quoted-secret", "nested": {"api_key": "quoted-api-secret"}}',
        attempt_kind="plan",
    )

    assert proposal.component_name == "null2_correction"
    traces = [json.loads(path.read_text()) for path in sorted(store.model_traces_dir.glob("*.json"))]
    assert len(traces) == 2
    assert {trace["validation"]["status"] for trace in traces} == {"ok", "rejected"}
    serialized = json.dumps(traces)
    assert "supersecret" not in serialized
    assert "secret-value" not in serialized
    assert "quoted-secret" not in serialized
    assert "quoted-api-secret" not in serialized
    assert "[REDACTED]" in serialized
    assert all(trace["stage"] == "inner" for trace in traces)



def test_normalize_patch_paths_strips_model_absolute_context_paths():
    diff = (
        "--- a/tools/hmmer-3.4/src/src/impl_sse/null2.c\n"
        "+++ b/tools/hmmer-3.4/src/src/impl_sse/null2.c\n"
        "@@ -1 +1 @@\n"
        "-old\n"
        "+new\n"
    )
    out = _normalize_patch_paths(diff, "impl_sse/null2.c")
    assert "a/src/impl_sse/null2.c" in out
    assert "b/src/impl_sse/null2.c" in out
    assert "tools/hmmer-3.4" not in out


@pytest.mark.requires_hmmer_source
def test_sequence_io_exact_edit_uses_easel_source_path(tmp_path: Path):
    store = LoopStore(tmp_path)
    pack = build_context_pack(store, round_idx=1)
    proposal = PatchProposal(
        component_name="sequence_io",
        rationale="Use exact old/new source replacement in the measured addbuf FASTA parser hotspot.",
        edits=[{
            "old": "        x  = sq->abc->inmap[(int) ascii->buf[ascii->bpos++]];\n",
            "new": "        x  = sq->abc->inmap[(int) ascii->buf[ascii->bpos++]]; /* measured addbuf hotspot */\n",
        }],
    )
    runner = PatchRunner(store, require_algorithmic_patch=False)
    patchset = runner.build_patchset(proposal, pack, {"provider": "test"})

    assert "a/easel/esl_sqio_ascii.c" in patchset.patches[0].unified_diff
    assert patchset.patches[0].edit_synthesis_errors == []

    record, cfg = runner.static_verify(patchset)

    assert record.status == "ok"
    assert cfg is not None
    assert cfg.c_patches[0].c_target_file == "easel/esl_sqio_ascii.c"



def test_patch_plan_accepts_high_level_pipeline_contract(tmp_path: Path):
    store = LoopStore(tmp_path)
    pack = build_context_pack(store, round_idx=1)
    proposal = PatchPlanProposal(
        component_name="pipeline_flow",
        target_functions=["p7_evo_align_Run"],
        algorithm_hypothesis="Pipeline-flow strategy keeps TP and FP unchanged while reducing redundant per-target runtime setup.",
        intended_code_change="Add executable, semantics-preserving branch structure inside p7_evo_align_Run without changing thresholds.",
        runtime_effective_argument="The change is compiled into rebuilt jackhmmer/hmmsearch runtime code, so executable hashes should change.",
        profile_evidence_citation="No measured profile evidence is available in this context pack.",
        expected_metric_effect="Lower runtime with unchanged TP and FP.",
        anti_degenerate_check="No target skipping, no threshold changes, and no hit suppression.",
    )
    runner = PatchRunner(store)

    record = runner.build_patch_plan(proposal, pack, {"provider": "test"})

    assert record.validation_status == "ok"
    assert record.component_name == "pipeline_flow"
    assert record.c_target_functions == ["p7_evo_align_Run"]



def test_patch_plan_rejects_high_level_out_of_contract_function(tmp_path: Path):
    store = LoopStore(tmp_path)
    pack = build_context_pack(store, round_idx=1)
    proposal = PatchPlanProposal(
        component_name="pipeline_flow",
        target_functions=["serial_loop"],
        algorithm_hypothesis="Bad pipeline-flow strategy that names a jackhmmer loop function but still mentions TP and FP.",
        intended_code_change="Modify the wrong function in a runtime-effective executable way.",
        runtime_effective_argument="The binary would change if this were allowed.",
        profile_evidence_citation="No measured profile evidence is available in this context pack.",
        expected_metric_effect="Lower runtime with unchanged TP and FP.",
        anti_degenerate_check="No threshold changes.",
    )
    runner = PatchRunner(store)

    record = runner.build_patch_plan(proposal, pack, {"provider": "test"})

    assert record.validation_status == "rejected"
    assert any("outside the component contract" in error for error in record.validation_errors)



@pytest.mark.requires_hmmer_source
def test_plan_diff_binding_rejects_actual_function_outside_inner_and_outer_plans(tmp_path: Path):
    store = LoopStore(tmp_path)
    pack = build_context_pack(store, round_idx=1)
    runner = PatchRunner(store, require_algorithmic_patch=False)
    outer = runner.build_algorithm_plan(AlgorithmPlanProposal(
        algorithm_family="pipeline_flow_strategy",
        baseline_algorithm_summary="Stock pipeline initialization and per-sequence setup execute before filtering.",
        proposed_algorithm_change="Optimize only per-sequence pipeline setup while preserving exact filtering decisions.",
        speed_mechanism="Reduce redundant setup work while retaining all stock scoring and reporting paths.",
        quality_preservation_argument="TP and FP remain unchanged because scores, thresholds, and hit reporting are unchanged.",
        target_contracts=["pipeline_flow"],
        target_functions=["p7_evo_align_Run"],
        profile_evidence_citation="No measured profile evidence is available in this context pack.",
    ), pack, {"provider": "test"})
    inner = runner.build_patch_plan(PatchPlanProposal(
        component_name="pipeline_flow",
        target_functions=["p7_evo_align_Run"],
        algorithm_hypothesis="Optimize per-sequence setup without changing TP or FP.",
        intended_code_change="Modify only p7_evo_align_Run runtime setup.",
        runtime_effective_argument="The rebuilt runtime executes this C function for each sequence.",
        profile_evidence_citation="No measured profile evidence is available in this context pack.",
        anti_degenerate_check="No threshold, score, reporting, or target coverage changes.",
    ), pack, {"provider": "test"}, algorithm_plan=outer)
    narrowed_outer = outer.model_copy(update={"target_functions": ["p7_evo_align_Run"]})
    proposal = PatchProposal(
        component_name="pipeline_flow",
        rationale="Final diff drifts to another function in the same component contract.",
        edits=[{
            "old": "  (void) state;\n  return eslOK;\n",
            "new": "  (void) state;\n  if (state == NULL) return eslOK;\n  return eslOK;\n",
        }],
    )

    patchset = runner.build_patchset(
        proposal,
        pack,
        {"provider": "test"},
        patch_plan=inner,
        algorithm_plan=narrowed_outer,
    )
    record, cfg = runner.static_verify(patchset)

    assert patchset.plan_diff_binding is not None
    assert patchset.plan_diff_binding.actual_footprint.extraction_status == "ok"
    assert "p7_evo_align_Reuse" in patchset.plan_diff_binding.actual_footprint.actual_functions
    assert patchset.plan_diff_binding.binding_status == "rejected"
    assert record.decision == "rejected_plan_diff_binding"
    assert controller._repairable_final_failure_kind(record) == "plan_diff_binding_failure"
    assert cfg is None
    assert any("inner patch plan" in error for error in record.errors)


@pytest.mark.requires_hmmer_source
def test_plan_diff_binding_accepts_actual_function_inside_inner_and_outer_plans(tmp_path: Path):
    store = LoopStore(tmp_path)
    pack = build_context_pack(store, round_idx=1)
    runner = PatchRunner(store, require_algorithmic_patch=False)
    outer = runner.build_algorithm_plan(AlgorithmPlanProposal(
        algorithm_family="pipeline_flow_strategy",
        baseline_algorithm_summary="Stock comparison delegates to the exact pipeline.",
        proposed_algorithm_change="Add a guarded comparison decision surface with exact fallback.",
        speed_mechanism="Reduce redundant setup while retaining exact fallback behavior.",
        quality_preservation_argument="TP and FP remain unchanged because scoring and reporting use the stock fallback.",
        target_contracts=["pipeline_flow"],
        target_functions=["p7_evo_align_Run"],
        profile_evidence_citation="No measured profile evidence is available in this context pack.",
    ), pack, {"provider": "test"})
    inner = runner.build_patch_plan(PatchPlanProposal(
        component_name="pipeline_flow",
        target_functions=["p7_evo_align_Run"],
        algorithm_hypothesis="Guard one exact fallback call while preserving TP and FP.",
        intended_code_change="Modify only p7_evo_align_Run.",
        runtime_effective_argument="The rebuilt runtime executes p7_evo_align_Run for each comparison.",
        profile_evidence_citation="No measured profile evidence is available in this context pack.",
        anti_degenerate_check="No threshold, score, reporting, or target coverage changes.",
    ), pack, {"provider": "test"}, algorithm_plan=outer)
    proposal = PatchProposal(
        component_name="pipeline_flow",
        rationale="Use exact old/new source replacement in the approved comparison function.",
        edits=[{
            "old": "  (void) state;\n  return p7_evo_align_stock_Run(pli, om, bg, sq, ntsq, hitlist);\n",
            "new": "  (void) state;\n  if (sq == NULL) return eslEINVAL;\n  return p7_evo_align_stock_Run(pli, om, bg, sq, ntsq, hitlist);\n",
        }],
        expected_metric_effect="No quality change; this verifies high-level static contract plumbing.",
        risk_notes="Test-only input guard before the exact stock fallback.",
    )
    patchset = runner.build_patchset(
        proposal,
        pack,
        {"provider": "test"},
        patch_plan=inner,
        algorithm_plan=outer,
    )

    record, cfg = runner.static_verify(patchset)

    assert patchset.plan_diff_binding is not None
    assert patchset.plan_diff_binding.binding_status == "ok"
    assert patchset.plan_diff_binding.actual_footprint.actual_functions == ["p7_evo_align_Run"]
    assert record.status == "ok"
    assert cfg is not None
    assert cfg.c_patches[0].c_target_file == "p7_evo_align.c"


@pytest.mark.requires_hmmer_source
def test_plan_diff_binding_rejects_final_component_drift_to_legacy_sse(tmp_path: Path):
    store = LoopStore(tmp_path)
    pack = build_context_pack(store, round_idx=1)
    runner = PatchRunner(store, require_algorithmic_patch=False)
    outer = runner.build_algorithm_plan(AlgorithmPlanProposal(
        algorithm_family="dp_recurrence_strategy",
        baseline_algorithm_summary="Stock NEON Forward recurrence is the quality reference.",
        proposed_algorithm_change="Remove redundant recurrence work with an exact fallback.",
        speed_mechanism="Reduce recurrence arithmetic while preserving exact scores.",
        quality_preservation_argument="TP and FP remain unchanged because recurrence outputs are preserved.",
        target_contracts=["dp_kernel_neon"],
        target_functions=["forward_engine"],
        profile_evidence_citation="No measured profile evidence is available in this context pack.",
    ), pack, {"provider": "test"})
    active_plan = PatchPlanProposal(
        component_name="dp_kernel_neon",
        target_functions=["forward_engine"],
        algorithm_hypothesis="Remove redundant NEON recurrence work while preserving TP and recall.",
        intended_code_change="Modify forward_engine arithmetic without changing scores.",
        runtime_effective_argument="The rebuilt AArch64 executable runs the changed NEON function.",
        profile_evidence_citation="No measured profile evidence is available in this context pack.",
        anti_degenerate_check="TP and recall remain unchanged because exact recurrence values are preserved.",
    )
    inner = runner.build_patch_plan(
        active_plan,
        pack,
        {"provider": "test"},
        algorithm_plan=outer,
    )
    legacy_plan = active_plan.model_copy(update={"component_name": "dp_kernel"})
    proposal, metadata = FixtureProposer().propose(pack, plan=legacy_plan)

    patchset = runner.build_patchset(
        proposal,
        pack,
        metadata,
        patch_plan=inner,
        algorithm_plan=outer,
    )
    record, cfg = runner.static_verify(patchset)

    assert patchset.plan_diff_binding is not None
    assert patchset.plan_diff_binding.binding_status == "rejected"
    assert patchset.plan_diff_binding.final_component_name == "dp_kernel"
    assert record.decision == "rejected_plan_diff_binding"
    assert cfg is None
    assert any("differs from approved inner patch plan" in error for error in record.errors)



@pytest.mark.requires_hmmer_source
def test_sequence_io_guardrail_rejects_removed_residue_check(tmp_path: Path):
    store = LoopStore(tmp_path)
    pack = build_context_pack(store, round_idx=1)
    proposal = PatchProposal(
        component_name="sequence_io",
        rationale="Unsafe addbuf shortcut that drops residue classification.",
        edits=[{
            "old": "        if (x <= 127) { nres--; sq->dsq[++sq->n] = x; }\n",
            "new": "        { nres--; sq->dsq[++sq->n] = x; }\n",
        }],
    )
    runner = PatchRunner(store)
    patchset = runner.build_patchset(proposal, pack, {"provider": "test"})

    errors = runner._algorithmic_patch_check(patchset)

    assert any("parser safety/classification" in error for error in errors)



@pytest.mark.requires_hmmer_source
def test_sequence_io_guardrail_rejects_new_simd_intrinsics(tmp_path: Path):
    store = LoopStore(tmp_path)
    pack = build_context_pack(store, round_idx=1)
    proposal = PatchProposal(
        component_name="sequence_io",
        rationale="Unsafe parser SIMD experiment.",
        unified_diff=(
            "--- a/easel/esl_sqio_ascii.c\n"
            "+++ b/easel/esl_sqio_ascii.c\n"
            "@@ -1 +1 @@\n"
            "+  __m128i zero = _mm_setzero_si128();\n"
        ),
    )
    runner = PatchRunner(store)
    patchset = runner.build_patchset(proposal, pack, {"provider": "test"})

    errors = runner._algorithmic_patch_check(patchset)

    assert any("SIMD/SSE intrinsic" in error for error in errors)


@pytest.mark.requires_hmmer_source
def test_sequence_io_guardrail_rejects_new_arm_neon_intrinsics(tmp_path: Path):
    store = LoopStore(tmp_path)
    pack = build_context_pack(store, round_idx=1)
    proposal = PatchProposal(
        component_name="sequence_io",
        rationale="Unsafe parser NEON experiment.",
        unified_diff=(
            "--- a/easel/esl_sqio_ascii.c\n"
            "+++ b/easel/esl_sqio_ascii.c\n"
            "@@ -1 +1 @@\n"
            "+  uint8x16_t residues = vld1q_u8(buf);\n"
        ),
    )
    runner = PatchRunner(store)
    patchset = runner.build_patchset(proposal, pack, {"provider": "test"})

    errors = runner._algorithmic_patch_check(patchset)

    assert any("ARM NEON" in error for error in errors)



@pytest.mark.requires_hmmer_source
def test_ssv_filter_guardrail_rejects_unguarded_high_calc_band_reference(tmp_path: Path):
    store = LoopStore(tmp_path)
    pack = build_context_pack(store, round_idx=1)
    proposal = PatchProposal(
        component_name="ssv_filter_neon",
        rationale="Unsafe get_xE dispatch shortcut that forgets MAX_BANDS guards.",
        unified_diff=(
            "--- a/src/impl_neon/ssvfilter.c\n"
            "+++ b/src/impl_neon/ssvfilter.c\n"
            "@@ -1 +1 @@\n"
            "+      xEv = calc_band_15(dsq, L, om, last_q, beginv, xEv);\n"
        ),
    )
    runner = PatchRunner(store)
    patchset = runner.build_patchset(proposal, pack, {"provider": "test"})

    errors = runner._algorithmic_patch_check(patchset)

    assert any("MAX_BANDS" in error for error in errors)



@pytest.mark.requires_hmmer_source
def test_nonapplying_proposal_keeps_its_text_for_the_guardrails(tmp_path: Path):
    """A proposal that cannot be materialized must still be judged as TEXT.

    ``_proposal_diff`` keeps the author's diff when it fails to apply, so the text
    guardrails (SIMD/NEON intrinsics, parser safety, MAX_BANDS, parameter-only,
    domain_def thresholds) still see it; the apply failure stays in
    ``edit_synthesis_errors``.
    """
    store = LoopStore(tmp_path)
    pack = build_context_pack(store, round_idx=1)
    proposal = PatchProposal(
        component_name="sequence_io",
        rationale="Unsafe parser SIMD experiment.",
        unified_diff=(
            "--- a/easel/esl_sqio_ascii.c\n"
            "+++ b/easel/esl_sqio_ascii.c\n"
            "@@ -1 +1 @@\n"
            "+  __m128i zero = _mm_setzero_si128();\n"
        ),
    )
    runner = PatchRunner(store)
    patchset = runner.build_patchset(proposal, pack, {"provider": "test"})

    assert patchset.patches[0].edit_synthesis_errors  # the failure is still recorded
    assert patchset.patches[0].unified_diff.strip()  # ... and the text survives
    errors = runner._algorithmic_patch_check(patchset)
    assert any("SIMD/SSE intrinsic" in error for error in errors)


@pytest.mark.requires_hmmer_source
def test_jackhmmer_loop_contract_rejects_frozen_driver_iteration_edit(tmp_path: Path):
    store = LoopStore(tmp_path)
    pack = build_context_pack(store, round_idx=1)
    proposal = PatchProposal(
        component_name="jackhmmer_loop",
        rationale="Unsafe iteration reduction.",
        unified_diff=(
            "--- a/src/jackhmmer.c\n"
            "+++ b/src/jackhmmer.c\n"
            "@@ -469,1 +469,1 @@\n"
            "-  maxiterations = esl_opt_GetInteger(go, \"-N\");\n"
            "+  maxiterations = 1;\n"
        ),
    )
    runner = PatchRunner(store)
    patchset = runner.build_patchset(proposal, pack, {"provider": "test"})

    record, cfg = runner.static_verify(patchset)

    assert cfg is None
    assert record.decision == "rejected_patch_apply"
    assert record.errors



@pytest.mark.requires_hmmer_source
def test_pipeline_flow_contract_rejects_frozen_pipeline_threshold_edit(tmp_path: Path):
    store = LoopStore(tmp_path)
    pack = build_context_pack(store, round_idx=1)
    proposal = PatchProposal(
        component_name="pipeline_flow",
        rationale="Unsafe report threshold tuning.",
        unified_diff=(
            "--- a/src/p7_pipeline.c\n"
            "+++ b/src/p7_pipeline.c\n"
            "@@ -409,1 +409,1 @@\n"
            "-      if ( !pli->long_targets  && exp(lnP) * pli->Z <= pli->E) return TRUE;\n"
            "+      if ( !pli->long_targets  && exp(lnP) * pli->Z <= pli->E * 2.0) return TRUE;\n"
        ),
    )
    runner = PatchRunner(store)
    patchset = runner.build_patchset(proposal, pack, {"provider": "test"})

    record, cfg = runner.static_verify(patchset)

    assert cfg is None
    assert record.decision == "rejected_patch_apply"
    assert record.errors



@pytest.mark.requires_hmmer_source
def test_exact_edit_blocks_are_synthesized_into_applyable_diff(tmp_path: Path):
    store = LoopStore(tmp_path)
    pack = build_context_pack(store, round_idx=1)
    proposal = PatchProposal(
        component_name="null2_correction",
        rationale="Use exact old/new source replacement so the controller generates the unified diff.",
        edits=[{
            "old": "  xfactor = XMXo(0, p7X_N) + XMXo(0, p7X_C) + XMXo(0, p7X_J); \n",
            "new": "  xfactor = (XMXo(0, p7X_N) + XMXo(0, p7X_C)) + XMXo(0, p7X_J); \n",
        }],
    )
    runner = PatchRunner(store, require_algorithmic_patch=False)
    patchset = runner.build_patchset(proposal, pack, {"provider": "test"})

    assert "@@" in patchset.patches[0].unified_diff
    assert "XMXo(0, p7X_N) + XMXo(0, p7X_C)" in patchset.patches[0].unified_diff
    assert patchset.patches[0].edit_synthesis_errors == []

    record, cfg = runner.static_verify(patchset)

    assert record.status == "ok"
    assert cfg is not None



@pytest.mark.requires_hmmer_source
def test_exact_edit_blocks_reject_non_unique_old_text(tmp_path: Path):
    store = LoopStore(tmp_path)
    pack = build_context_pack(store, round_idx=1)
    proposal = PatchProposal(
        component_name="null2_correction",
        rationale="Ambiguous exact old block should be rejected before assembling a diff.",
        edits=[{"old": "  {\n", "new": "  { /* ambiguous */\n"}],
    )
    runner = PatchRunner(store, require_algorithmic_patch=False)
    patchset = runner.build_patchset(proposal, pack, {"provider": "test"})

    record, cfg = runner.static_verify(patchset)

    assert cfg is None
    assert record.status == "compile_error"
    assert record.decision == "rejected_patch_apply"
    assert "old block matched" in record.errors[0]



@pytest.mark.requires_hmmer_source
def test_fixture_patch_static_verifies_and_writes_records(tmp_path: Path):
    store = LoopStore(tmp_path)
    pack = build_context_pack(store, round_idx=1)
    proposal, metadata = FixtureProposer().propose(pack)
    runner = PatchRunner(store)
    patchset = runner.build_patchset(proposal, pack, metadata)
    record, cfg = runner.static_verify(patchset)
    assert record.status == "ok"
    assert record.eval_id
    assert cfg is not None
    assert cfg.c_patches[0].c_target_file == "p7_evo_align.c"
    assert (store.patchsets_dir / f"{patchset.patchset_id}.json").exists()
    assert (store.evaluations_dir / f"{record.eval_id}.json").exists()
    assert (tmp_path / "round_001" / "pending_mods.json").exists()


@pytest.mark.requires_hmmer_source
def test_static_verify_rejects_non_applyable_diff(tmp_path: Path):
    store = LoopStore(tmp_path)
    pack = build_context_pack(store, round_idx=1)
    proposal = PatchProposal(
        component_name="domain_def",
        rationale="Malformed hunk context should be caught before smoke eval.",
        unified_diff=(
            "--- a/src/p7_domaindef.c\n"
            "+++ b/src/p7_domaindef.c\n"
            "@@ -106,10 +106,10 @@\n"
            "   /* default thresholds */\n"
            "-  ddef->rt1           = 0.25;\n"
            "+  ddef->rt1           = 0.35;\n"
        ),
    )
    runner = PatchRunner(store, require_algorithmic_patch=False)
    patchset = runner.build_patchset(proposal, pack, {"provider": "test"})

    record, cfg = runner.static_verify(patchset)

    assert cfg is None
    assert record.status == "compile_error"
    assert record.decision == "rejected_patch_apply"
    assert "git apply --check failed" in record.errors[0]


@pytest.mark.requires_hmmer_source
def test_static_verify_matches_runtime_isolated_apply_for_generated_diff(tmp_path: Path):
    store = LoopStore(tmp_path)
    pack = build_context_pack(store, round_idx=1)
    proposal = PatchProposal(
        component_name="dp_kernel",
        rationale="Generated diff that previously passed host git apply but failed runtime copied-source apply.",
        unified_diff=(
            "diff --git a/src/impl_sse/fwdback.c b/src/impl_sse/fwdback.c\n"
            "index 7f920c0..a1b2c3d 100644\n"
            "--- a/src/impl_sse/fwdback.c\n"
            "+++ b/src/impl_sse/fwdback.c\n"
            "@@ -313,8 +313,9 @@ forward_engine\n"
            " \t\t  sv   = _mm_add_ps(sv, _mm_mul_ps(ipv, *tp)); tp++;\n"
            " \t\t  sv   = _mm_add_ps(sv, _mm_mul_ps(dpv, *tp)); tp++;\n"
            " \t\t  sv   = _mm_mul_ps(sv, *rp);                  rp++;\n"
            "-\t\t  xEv  = _mm_add_ps(xEv, sv);\n"
            "-\t\t  \n"
            "+\t\t  dcv   = _mm_mul_ps(sv, *tp);\n"
            "+\t\t  xEv   = _mm_add_ps(xEv, sv);\n"
            "+\n"
            " \t\t  /* Load {MDI}(i-1,q) into mpv, dpv, ipv;\n"
            " \t\t   * {MDI}MX(q) is then the current, not the prev row\n"
            " \t\t   */\n"
            "@@ -330,7 +331,7 @@ forward_engine\n"
            " \t\t  /* Calculate the next D(i,q+1) partially: M->D only;\n"
            " \t           * delay storage, holding it in dcv\n"
            " \t\t   */\n"
            "-\t\t  dcv   = _mm_mul_ps(sv, *tp); tp++;\n"
            "+\t\t  tp++;\n"
            " \n"
            " \t\t  /* Calculate and store I(i,q); assumes odds ratio for emission is 1.0 */\n"
            " \t\t  sv         =                _mm_mul_ps(mpv, *tp);  tp++;\n"
        ),
    )
    runner = PatchRunner(store, require_algorithmic_patch=False)
    patchset = runner.build_patchset(proposal, pack, {"provider": "test"})

    record, cfg = runner.static_verify(patchset)

    assert cfg is None
    assert record.status == "compile_error"
    assert record.decision == "rejected_patch_apply"
    assert "while materializing source" in record.errors[0]
    assert "patch does not apply" in record.errors[0]



@pytest.mark.requires_hmmer_source
def test_static_verify_rejects_parameter_only_domain_threshold_patch(tmp_path: Path):
    store = LoopStore(tmp_path)
    pack = build_context_pack(store, round_idx=1)
    proposal = PatchProposal(
        component_name="domain_def",
        rationale="Parameter-only threshold tuning should be rejected for algorithmic runs.",
        unified_diff=(
            "--- a/src/p7_domaindef.c\n"
            "+++ b/src/p7_domaindef.c\n"
            "@@ -104,15 +104,15 @@\n"
            "   ddef->nenvelopes = 0;\n"
            " \n"
            "   /* default thresholds */\n"
            "   ddef->rt1           = 0.25;\n"
            "-  ddef->rt2           = 0.10;\n"
            "+  ddef->rt2           = 0.15;\n"
            "   ddef->rt3           = 0.20;\n"
            "   ddef->nsamples      = 200;\n"
            "   ddef->min_overlap   = 0.8;\n"
            "   ddef->of_smaller    = TRUE;\n"
            "   ddef->max_diagdiff  = 4;\n"
            "   ddef->min_posterior = 0.25;\n"
            "   ddef->min_endpointp = 0.02;\n"
            " \n"
            "   /* allocate reusable, growable objects that domain def reuses for each seq */\n"
            "   ddef->sp  = p7_spensemble_Create(1024, 64, 32); /* init allocs = # sampled pairs; max endpoint range; # of domains */\n"
        ),
    )
    runner = PatchRunner(store)
    patchset = runner.build_patchset(proposal, pack, {"provider": "test"})

    record, cfg = runner.static_verify(patchset)

    assert cfg is None
    assert record.status == "compile_error"
    assert record.decision == "rejected_parameter_tuning"
    assert "only changes default threshold" in record.errors[0]


def test_full_decision_rejects_faster_candidate_with_tp_loss(tmp_path: Path):
    runner = PatchRunner(LoopStore(tmp_path), accept_min_tp_fraction=1.0)
    current_best = EvalResult(stage="full", status="ok", tp=111, fp=1, tp_fp_ratio=111 / 112, cost_s=1.2, meta={"comparability": _comparable(stage="full")})
    degenerate = EvalResult(stage="full", status="ok", tp=5, fp=0, tp_fp_ratio=1.0, cost_s=0.8, meta={"comparability": _comparable(stage="full")})

    decision = runner._decision(degenerate, stage="full", current_best=controller.eval_summary_from_record(current_best.model_dump()))

    assert decision == "rejected_quality_tp_regression"



def test_full_decision_accepts_quality_preserving_speedup(tmp_path: Path):
    runner = PatchRunner(LoopStore(tmp_path), accept_min_tp_fraction=1.0, min_speedup_fraction=0.03)
    current_best = EvalResult(stage="full", status="ok", tp=111, fp=1, tp_fp_ratio=111 / 112, cost_s=1.3, meta={"comparability": _comparable(stage="full")})
    faster = EvalResult(stage="full", status="ok", tp=111, fp=1, tp_fp_ratio=111 / 112, cost_s=1.2, meta={"comparability": _comparable(stage="full")})

    decision = runner._decision(faster, stage="full", current_best=controller.eval_summary_from_record(current_best.model_dump()))

    assert decision == "accepted_speedup"



def test_full_decision_rejects_faster_candidate_with_fp_regression(tmp_path: Path):
    runner = PatchRunner(LoopStore(tmp_path), accept_min_tp_fraction=1.0, min_speedup_fraction=0.03)
    current_best = EvalResult(stage="full", status="ok", tp=111, fp=1, tp_fp_ratio=111 / 112, cost_s=1.3, meta={"comparability": _comparable(stage="full")})
    higher_fp = EvalResult(stage="full", status="ok", tp=111, fp=2, tp_fp_ratio=111 / 113, cost_s=1.0, meta={"comparability": _comparable(stage="full")})

    decision = runner._decision(higher_fp, stage="full", current_best=controller.eval_summary_from_record(current_best.model_dump()))

    assert decision == "rejected_quality_fp_regression"



def test_full_decision_rejects_quality_preserving_candidate_below_speedup_threshold(tmp_path: Path):
    runner = PatchRunner(LoopStore(tmp_path), accept_min_tp_fraction=1.0, min_speedup_fraction=0.03)
    current_best = EvalResult(stage="full", status="ok", tp=111, fp=1, tp_fp_ratio=111 / 112, cost_s=1.3, meta={"comparability": _comparable(stage="full")})
    too_small = EvalResult(stage="full", status="ok", tp=111, fp=1, tp_fp_ratio=111 / 112, cost_s=1.27, meta={"comparability": _comparable(stage="full")})

    decision = runner._decision(too_small, stage="full", current_best=controller.eval_summary_from_record(current_best.model_dump()))

    assert decision == "rejected_not_faster"



def test_full_decision_rejects_slower_candidate_with_better_ratio(tmp_path: Path):
    runner = PatchRunner(LoopStore(tmp_path), accept_min_tp_fraction=1.0, min_speedup_fraction=0.03)
    current_best = EvalResult(stage="full", status="ok", tp=111, fp=1, tp_fp_ratio=111 / 112, cost_s=1.3, meta={"comparability": _comparable(stage="full")})
    slower = EvalResult(stage="full", status="ok", tp=112, fp=0, tp_fp_ratio=1.0, cost_s=1.4, meta={"comparability": _comparable(stage="full")})

    decision = runner._decision(slower, stage="full", current_best=controller.eval_summary_from_record(current_best.model_dump()))

    assert decision == "rejected_not_faster"



def test_decision_rejects_non_runtime_effective_patch(tmp_path: Path):
    runner = PatchRunner(LoopStore(tmp_path), accept_min_tp_fraction=1.0)
    result = EvalResult(stage="full", status="ok", tp=111, fp=1, tp_fp_ratio=111 / 112, cost_s=1.2)

    decision = runner._decision(result, stage="full", current_best=None, build={"runtime_effective": False})

    assert decision == "rejected_not_runtime_effective"



def _paired_backend(
    outcome,
    *,
    query_result=None,
    queries_for=None,
    calls: list | None = None,
):
    """Fake backend for paired timing that commits real per-query evidence.

    ``outcome(role, occurrence)`` returns ``(cost_s, tp, fp, status)`` for one
    scheduled run, where ``role`` is ``initial``, ``candidate`` or
    ``reference``. Paired confirmation re-reads its evidence from the committed
    query completion records, so a paired test only proves something if the
    backend commits them the way a real run does.
    """
    counts: dict[str, int] = {}

    def fake_run_all_queries(cfg, **kwargs):
        work_root = Path(kwargs["work_root"])
        role = next(
            (
                name
                for name in ("candidate", "reference")
                if work_root.name.endswith(f"_{name}")
            ),
            "initial",
        )
        counts[role] = counts.get(role, 0) + 1
        occurrence = counts[role]
        cost_s, tp, fp, status = outcome(role, occurrence)
        kwargs["resume_provider"]({
            "jackhmmer_sha256": "jack-sha",
            "hmmsearch_sha256": "search-sha",
        })
        committed = (queries_for or (lambda *_: (1, 2)))(role, occurrence)
        for idx in committed:
            kwargs["on_query_complete"](
                idx,
                str(Path(kwargs["data_dir"]) / f"query{idx}.fasta"),
                work_root / "queries" / f"query_{idx:04d}",
                (query_result or (lambda r, o, i, c: _query_result(i, cost_s=c)))(
                    role, occurrence, idx, cost_s / 2.0
                ),
            )
        if calls is not None:
            calls.append({"role": role, "work_root": str(work_root)})
        return EvalResult(
            stage="full",
            status=status,
            tp=tp,
            fp=fp,
            tp_fp_ratio=tp / (tp + fp) if tp + fp else 0.0,
            n_seqs=2,
            cost_s=cost_s,
            meta={
                "profile": _profile(jack=cost_s - 1.0, search=1.0),
                "build": {"runtime_effective": True},
                "comparability": _comparable(stage="full"),
            },
        )

    return fake_run_all_queries


def _paired_current_best(
    runner: PatchRunner,
    *,
    cost_s: float = 100.0,
    tp: int = 10,
    fp: int = 1,
    eval_id: str = "best",
) -> EvalSummary:
    """Current best whose patchset is resolvable back into a runnable config."""
    patchset = runner.build_stock_patchset(stage="full")
    return EvalSummary(
        eval_id=eval_id,
        patchset_id=patchset.patchset_id,
        stage="full",
        status="ok",
        tp=tp,
        fp=fp,
        tp_fp_ratio=tp / (tp + fp),
        cost_s=cost_s,
        comparability=_comparable(stage="full"),
    )


def _fixed_costs(costs: dict[str, list[float]], *, tp: int = 10, fp: int = 1):
    """Outcome function driven by a per-role list of wall times."""

    def outcome(role: str, occurrence: int) -> tuple[float, int, int, str]:
        values = costs[role]
        return values[min(occurrence, len(values)) - 1], tp, fp, "ok"

    return outcome


def test_paired_current_best_can_reject_noisy_near_threshold_acceptance(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    runner = _checkpoint_runner(tmp_path, min_speedup_fraction=0.03, paired_current_best_repeats=1)
    current_best = _paired_current_best(runner, eval_id="stale_best")
    candidate_patchset = runner.build_stock_patchset(stage="full")
    monkeypatch.setattr(
        run_hmmer_cli,
        "run_all_queries",
        _paired_backend(_fixed_costs({
            "initial": [96.5],
            "candidate": [98.0],
            "reference": [99.0],
        })),
    )

    record = runner.evaluate(candidate_patchset, object(), stage="full", current_best=current_best, quality_reference=current_best)

    paired = record.metric_delta["paired_confirmation"]
    assert record.decision == "rejected_not_faster_paired"
    assert paired["status"] == "ok"
    assert paired["paired_cost_s"] == 99.0
    assert paired["confirmed"] is False
    assert paired["validation_errors"] == []
    # The stored cost is the paired median whichever way the verdict lands, so a
    # rejected record cannot advertise the faster pre-pairing measurement.
    assert (record.eval_result or {})["cost_s"] == 98.0
    assert paired["primary_candidate_cost_s"] == 96.5



def test_paired_current_best_records_median_cost_on_rejection(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """A rejection must not store a number the paired estimate contradicts.

    The unpaired measurement (90.0) is optimistic; pairing against the current
    best puts the candidate's real cost at 100.0. The record has to carry the
    paired median, and the pre-pairing value has to survive next to it.
    """
    runner = _checkpoint_runner(tmp_path, min_speedup_fraction=0.10, paired_current_best_repeats=3)
    current_best = _paired_current_best(runner, cost_s=110.0, eval_id="stale_best")
    candidate_patchset = runner.build_stock_patchset(stage="full")
    monkeypatch.setattr(
        run_hmmer_cli,
        "run_all_queries",
        _paired_backend(_fixed_costs({
            "initial": [90.0],
            "candidate": [98.0, 100.0, 102.0],
            "reference": [100.0, 100.0, 100.0],
        })),
    )

    record = runner.evaluate(candidate_patchset, object(), stage="full", current_best=current_best, quality_reference=current_best)

    paired = record.metric_delta["paired_confirmation"]
    assert record.decision == "rejected_not_faster_paired"
    assert paired["status"] == "ok"
    assert paired["confirmed"] is False
    assert paired["candidate_median_cost_s"] == 100.0
    assert paired["primary_candidate_cost_s"] == 90.0
    # The stored cost_s is the paired median, not the unpaired 90.0.
    assert paired["median_pair_speedup_fraction"] == pytest.approx(0.0)
    assert paired["median_pair_speedup_fraction"] <= runner.min_speedup_fraction
    assert (record.eval_result or {})["cost_s"] == 100.0
    assert runner.store.read_index("current_speed_best_full") is None



def test_paired_current_best_can_accept_noisy_near_threshold_rejection(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    runner = _checkpoint_runner(tmp_path, min_speedup_fraction=0.03, paired_current_best_repeats=1)
    current_best = _paired_current_best(runner, eval_id="stale_best")
    candidate_patchset = runner.build_stock_patchset(stage="full")
    monkeypatch.setattr(
        run_hmmer_cli,
        "run_all_queries",
        _paired_backend(_fixed_costs({
            "initial": [97.5],
            "candidate": [97.0],
            "reference": [101.0],
        })),
    )

    record = runner.evaluate(candidate_patchset, object(), stage="full", current_best=current_best, quality_reference=current_best)

    paired = record.metric_delta["paired_confirmation"]
    assert record.decision == "accepted_speedup_paired"
    assert paired["confirmed"] is True
    assert paired["speedup_fraction"] > runner.min_speedup_fraction



def test_paired_current_best_confirms_far_speedup_with_authoritative_median(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    runner = _checkpoint_runner(tmp_path, min_speedup_fraction=0.03, paired_current_best_repeats=3)
    current_best = _paired_current_best(runner, eval_id="stale_best")
    candidate_patchset = runner.build_stock_patchset(stage="full")
    calls: list = []
    monkeypatch.setattr(
        run_hmmer_cli,
        "run_all_queries",
        _paired_backend(
            _fixed_costs({
                "initial": [80.0],
                "candidate": [82.0, 78.0, 81.0],
                "reference": [100.0, 100.0, 100.0],
            }),
            calls=calls,
        ),
    )

    record = runner.evaluate(candidate_patchset, object(), stage="full", current_best=current_best, quality_reference=current_best)

    paired = record.metric_delta["paired_confirmation"]
    assert record.decision == "accepted_speedup_paired"
    assert paired["confirmed"] is True
    assert paired["trigger_reasons"] == ["accepted_best_confirmation"]
    assert paired["candidate_median_cost_s"] == 81.0
    # The paired median, not the single unpaired measurement, is what the rest
    # of the loop is allowed to treat as this candidate's cost.
    assert (record.eval_result or {})["cost_s"] == 81.0
    assert record.metric_delta["speedup_fraction"] == pytest.approx(0.19)
    indexed = runner.store.read_index("current_speed_best_full")
    assert indexed["cost_s"] == 81.0
    assert indexed["paired_confirmation"]["confirmed"] is True
    assert indexed["paired_confirmation"]["candidate_median_cost_s"] == 81.0
    assert indexed["paired_confirmation"]["current_best_eval_id"] == "stale_best"
    assert any("paired_current_best" in Path(call["work_root"]).parts for call in calls)



def test_paired_current_best_records_interleaved_schedule_and_median(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    runner = _checkpoint_runner(tmp_path, min_speedup_fraction=0.03, paired_current_best_repeats=3)
    current_best = _paired_current_best(runner)
    candidate_patchset = runner.build_stock_patchset(stage="full")
    monkeypatch.setattr(
        run_hmmer_cli,
        "run_all_queries",
        _paired_backend(_fixed_costs({
            "initial": [97.5],
            "candidate": [96.0, 96.0, 96.0],
            "reference": [100.0, 101.0, 99.0],
        })),
    )

    record = runner.evaluate(candidate_patchset, object(), stage="full", current_best=current_best, quality_reference=current_best)
    paired = record.metric_delta["paired_confirmation"]

    assert [item["role"] for item in paired["schedule"]] == [
        "candidate",
        "reference",
        "reference",
        "candidate",
        "candidate",
        "reference",
    ]
    assert [item["order"] for item in paired["schedule"]] == list(range(1, 7))
    assert paired["candidate_cost_s_values"] == [96.0, 96.0, 96.0]
    assert paired["reference_cost_s_values"] == [100.0, 101.0, 99.0]
    assert paired["statistic"] == "median_pair_speedup_fraction"
    assert paired["median_pair_speedup_fraction"] == pytest.approx(4.0 / 100.0)
    # One protocol fingerprint means candidate and reference were measured the
    # same way; the differing binaries are recorded per run instead.
    assert len(paired["timing_protocol_fingerprints"]) == 1
    assert all(item["binary_sha256"]["jackhmmer"] == "jack-sha" for item in paired["schedule"])
    assert record.decision == "accepted_speedup_paired"



def test_paired_current_best_fails_closed_on_pair_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    runner = _checkpoint_runner(tmp_path, min_speedup_fraction=0.03, paired_current_best_repeats=1)
    current_best = _paired_current_best(runner)
    candidate_patchset = runner.build_stock_patchset(stage="full")

    def outcome(role: str, occurrence: int) -> tuple[float, int, int, str]:
        if role == "reference":
            return 100.0, 10, 1, "run_error"
        return 96.5, 10, 1, "ok"

    monkeypatch.setattr(run_hmmer_cli, "run_all_queries", _paired_backend(outcome))

    record = runner.evaluate(candidate_patchset, object(), stage="full", current_best=current_best, quality_reference=current_best)

    paired = record.metric_delta["paired_confirmation"]
    assert record.decision == "rejected_paired_confirmation_error"
    assert paired["status"] == "run_error"
    assert paired["confirmed"] is False
    assert paired["run_errors"] and "reference" in paired["run_errors"][0]
    assert paired["evidence_errors"] == []
    assert (record.eval_result or {})["cost_s"] == 96.5



def test_paired_current_best_rejects_uncommitted_query_evidence(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    runner = _checkpoint_runner(tmp_path, min_speedup_fraction=0.03, paired_current_best_repeats=1)
    current_best = _paired_current_best(runner)
    candidate_patchset = runner.build_stock_patchset(stage="full")
    monkeypatch.setattr(
        run_hmmer_cli,
        "run_all_queries",
        _paired_backend(
            _fixed_costs({
                "initial": [96.5],
                "candidate": [96.0],
                "reference": [100.0],
            }),
            # The reference reports two queries but only ever commits one, the
            # shape an interrupted-then-aggregated run leaves behind.
            queries_for=lambda role, occurrence: (1,) if role == "reference" else (1, 2),
        ),
    )

    record = runner.evaluate(candidate_patchset, object(), stage="full", current_best=current_best, quality_reference=current_best)

    paired = record.metric_delta["paired_confirmation"]
    assert record.decision == "rejected_paired_incomplete_evidence"
    assert paired["status"] == "incomplete_evidence"
    assert paired["run_errors"] == []
    assert any("committed queries are 1" in error for error in paired["evidence_errors"])



def test_paired_current_best_rejects_unhashed_query_artifact(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    runner = _checkpoint_runner(tmp_path, min_speedup_fraction=0.03, paired_current_best_repeats=1)
    current_best = _paired_current_best(runner)
    candidate_patchset = runner.build_stock_patchset(stage="full")

    def query_result(role: str, occurrence: int, idx: int, cost_s: float) -> EvalResult:
        result = _query_result(idx, cost_s=cost_s)
        if role == "candidate" and idx == 2:
            # A zero-byte, unhashed table is exactly what a killed query leaves
            # on disk, and TP/FP are read back out of that table.
            result.meta["profile"]["artifacts"]["hmmsearch_tbl"] = {
                "exists": True,
                "bytes": 0,
                "sha256": "",
            }
        return result

    monkeypatch.setattr(
        run_hmmer_cli,
        "run_all_queries",
        _paired_backend(
            _fixed_costs({
                "initial": [96.5],
                "candidate": [96.0],
                "reference": [100.0],
            }),
            query_result=query_result,
        ),
    )

    record = runner.evaluate(candidate_patchset, object(), stage="full", current_best=current_best, quality_reference=current_best)

    paired = record.metric_delta["paired_confirmation"]
    assert record.decision == "rejected_paired_incomplete_evidence"
    assert any(
        "query 2 artifacts are incomplete: hmmsearch_tbl" in error
        for error in paired["evidence_errors"]
    )



def test_paired_current_best_rejects_quality_regression_inside_a_pair(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    runner = _checkpoint_runner(tmp_path, min_speedup_fraction=0.03, paired_current_best_repeats=1)
    current_best = _paired_current_best(runner)
    candidate_patchset = runner.build_stock_patchset(stage="full")

    def outcome(role: str, occurrence: int) -> tuple[float, int, int, str]:
        if role == "candidate":
            # Faster, but it invents a false positive the reference does not
            # report: speed evidence that costs quality is not acceptable.
            return 90.0, 10, 2, "ok"
        return (96.5 if role == "initial" else 100.0), 10, 1, "ok"

    monkeypatch.setattr(run_hmmer_cli, "run_all_queries", _paired_backend(outcome))

    record = runner.evaluate(candidate_patchset, object(), stage="full", current_best=current_best, quality_reference=current_best)

    paired = record.metric_delta["paired_confirmation"]
    assert record.decision == "rejected_paired_quality_regression"
    assert paired["status"] == "quality_regression"
    assert paired["run_errors"] == [] and paired["evidence_errors"] == []
    assert any("candidate FP 2 > reference FP 1" in error for error in paired["quality_errors"])
    assert runner.store.read_index("current_speed_best_full") is None



def test_paired_current_best_rejects_quality_unstable_across_repeats(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    runner = _checkpoint_runner(tmp_path, min_speedup_fraction=0.03, paired_current_best_repeats=3)
    current_best = _paired_current_best(runner, tp=9)
    candidate_patchset = runner.build_stock_patchset(stage="full")

    def outcome(role: str, occurrence: int) -> tuple[float, int, int, str]:
        if role == "candidate":
            # No single pair regresses against its own reference, but a
            # candidate whose hit set moves between repeats is not measurable.
            return 90.0, (9 if occurrence == 2 else 10), 1, "ok"
        return (96.5 if role == "initial" else 100.0), 9, 1, "ok"

    monkeypatch.setattr(run_hmmer_cli, "run_all_queries", _paired_backend(outcome))

    record = runner.evaluate(candidate_patchset, object(), stage="full", current_best=current_best, quality_reference=current_best)

    paired = record.metric_delta["paired_confirmation"]
    assert record.decision == "rejected_paired_quality_regression"
    assert paired["quality_errors"] == [
        "candidate TP/FP is unstable across repeats: tp=9 fp=1, tp=10 fp=1"
    ]



def test_first_candidate_is_confirmed_by_repeated_candidate_runs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    runner = _checkpoint_runner(tmp_path, min_speedup_fraction=0.03, paired_current_best_repeats=3)
    candidate_patchset = runner.build_stock_patchset(stage="full")
    calls: list = []
    monkeypatch.setattr(
        run_hmmer_cli,
        "run_all_queries",
        _paired_backend(
            _fixed_costs({"initial": [10.0], "candidate": [12.0, 10.0, 11.0]}),
            calls=calls,
        ),
    )

    record = runner.evaluate(candidate_patchset, object(), stage="full", current_best=None)

    paired = record.metric_delta["paired_confirmation"]
    assert record.decision == "accepted_full_paired"
    assert paired["mode"] == "repeated_candidate"
    assert paired["statistic"] == "median_candidate_cost_s"
    assert paired["confirmed"] is True
    assert paired["candidate_median_cost_s"] == 11.0
    assert paired["reference_cost_s_values"] == []
    assert paired["pair_speedup_fractions"] == []
    assert (record.eval_result or {})["cost_s"] == 11.0
    assert runner.store.read_index("current_speed_best_full")["cost_s"] == 11.0
    assert [call["role"] for call in calls] == ["initial", "candidate", "candidate", "candidate"]



def test_repeated_candidate_confirmation_rejects_unstable_quality(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    runner = _checkpoint_runner(tmp_path, min_speedup_fraction=0.03, paired_current_best_repeats=3)
    candidate_patchset = runner.build_stock_patchset(stage="full")

    def outcome(role: str, occurrence: int) -> tuple[float, int, int, str]:
        if role == "candidate":
            return 11.0, 10, (2 if occurrence == 3 else 1), "ok"
        return 10.0, 10, 1, "ok"

    monkeypatch.setattr(run_hmmer_cli, "run_all_queries", _paired_backend(outcome))

    record = runner.evaluate(candidate_patchset, object(), stage="full", current_best=None)

    paired = record.metric_delta["paired_confirmation"]
    assert record.decision == "rejected_paired_quality_regression"
    assert paired["confirmed"] is False
    assert runner.store.read_index("current_speed_best_full") is None



def _ledger_invocation(
    store: LoopStore,
    *,
    invocation_id: str,
    configuration_id: str = "config",
) -> InvocationRecord:
    invocation = InvocationRecord(
        invocation_id=invocation_id,
        configuration_id=configuration_id,
        configuration={},
        requested_outer=1,
        inner_budget=1,
        created_at="2026-01-01T00:00:00Z",
    )
    store.put_invocation(invocation)
    for inner_idx in (None, 1):
        store.append_slot_event(
            None,
            slot_id=controller._slot_id(configuration_id, 1, inner_idx),
            invocation_id=invocation_id,
            configuration_id=configuration_id,
            outer_idx=1,
            inner_idx=inner_idx,
            state="pending",
        )
    return invocation


def test_rebuild_run_summary_uses_canonical_heads_not_summary_episodes(
    tmp_path: Path,
):
    store = LoopStore(tmp_path)
    invocation = _ledger_invocation(store, invocation_id="invocation")
    controller._transition_slot(
        store,
        invocation,
        outer_idx=1,
        inner_idx=None,
        state="validated",
        result={"algorithm_plan_id": "algorithm-plan"},
    )
    controller._transition_slot(
        store,
        invocation,
        outer_idx=1,
        inner_idx=1,
        state="evaluated",
        result={
            "patch_plan_id": "patch-plan",
            "patchset_id": "patchset",
            "final_eval_id": "evaluation",
        },
    )
    interrupted = controller._finalize_invocation(
        store,
        invocation,
        terminal_state="interrupted",
    )
    complete = controller._finalize_invocation(
        store,
        invocation,
        terminal_state="complete",
    )

    summary = controller._rebuild_run_summary(store)

    assert interrupted.episode_idx == 1
    assert complete.episode_idx == 2
    assert complete.parent_invocation_summary_id == interrupted.invocation_summary_id
    assert summary.terminal_state == "complete"
    assert summary.proposal_attempts == 2
    assert summary.valid_plans == 1
    assert summary.evaluated_candidates == 1
    assert summary.invocation_ids == ["invocation"]


def test_read_run_summary_rebuilds_stale_projection(tmp_path: Path):
    store = LoopStore(tmp_path)
    invocation = _ledger_invocation(store, invocation_id="invocation")
    controller._finalize_invocation(
        store,
        invocation,
        terminal_state="partial",
    )
    original = controller._rebuild_run_summary(store)
    controller._transition_slot(
        store,
        invocation,
        outer_idx=1,
        inner_idx=None,
        state="blocked",
        result={"status": "algorithm_plan_proposal_error"},
    )
    controller._transition_slot(
        store,
        invocation,
        outer_idx=1,
        inner_idx=1,
        state="blocked",
        result={"status": "algorithm_plan_proposal_error"},
    )

    rebuilt = controller._read_run_summary(store)

    assert rebuilt is not None
    assert rebuilt.high_water_mark != original.high_water_mark
    assert rebuilt.blocked_slots == 1


def test_open_invocation_resumes_partial_but_not_terminal_episode(
    tmp_path: Path,
):
    class Runner:
        invocation_id = None

        def reuse_fingerprint(self, *, stage: str) -> str:
            return f"fingerprint:{stage}"

    store = LoopStore(tmp_path)
    runner = Runner()
    arguments = {"rounds": 1, "inner_rounds": 1, "stage": "full"}
    first = controller._open_invocation(store, runner, arguments)
    controller._finalize_invocation(
        store,
        first,
        terminal_state="interrupted",
    )

    resumed = controller._open_invocation(store, runner, arguments)
    assert resumed.invocation_id == first.invocation_id

    controller._finalize_invocation(
        store,
        resumed,
        terminal_state="complete",
    )
    replacement = controller._open_invocation(store, runner, arguments)
    assert replacement.invocation_id != first.invocation_id


def test_open_invocation_does_not_resume_incompatible_configuration(
    tmp_path: Path,
):
    class Runner:
        invocation_id = None

        def reuse_fingerprint(self, *, stage: str) -> str:
            return f"fingerprint:{stage}"

    store = LoopStore(tmp_path)
    runner = Runner()
    full = controller._open_invocation(
        store,
        runner,
        {"rounds": 1, "inner_rounds": 1, "stage": "full"},
    )
    smoke = controller._open_invocation(
        store,
        runner,
        {"rounds": 1, "inner_rounds": 1, "stage": "smoke"},
    )

    assert smoke.invocation_id != full.invocation_id
    assert smoke.configuration_id != full.configuration_id


def test_invocation_terminal_state_requires_outer_and_inner_terminal(
    tmp_path: Path,
):
    store = LoopStore(tmp_path)
    invocation = _ledger_invocation(store, invocation_id="invocation")
    controller._transition_slot(
        store,
        invocation,
        outer_idx=1,
        inner_idx=None,
        state="validated",
        result={"algorithm_plan_id": "algorithm-plan"},
    )
    controller._transition_slot(
        store,
        invocation,
        outer_idx=1,
        inner_idx=1,
        state="full_running",
        result={"patchset_id": "patchset"},
    )

    assert controller._invocation_terminal_state(store, invocation) == "partial"

    controller._transition_slot(
        store,
        invocation,
        outer_idx=1,
        inner_idx=1,
        state="evaluated",
        result={"patchset_id": "patchset", "final_eval_id": "evaluation"},
    )
    assert controller._invocation_terminal_state(store, invocation) == "complete"


def test_run_summary_requires_evaluated_terminal_outcome(tmp_path: Path):
    store = LoopStore(tmp_path)
    summary = controller._write_run_summary(
        store,
        [{"round_idx": 1, "status": "algorithm_plan_proposal_error", "error": "invalid plan"}],
        requested=1,
    )

    assert summary.terminal_state == "partial"
    assert summary.evaluated == 0
    assert summary.results[0]["terminal_outcome"] == "proposal_error"


def test_run_summary_records_distinct_terminal_outcomes(tmp_path: Path):
    store = LoopStore(tmp_path)
    summary = controller._write_run_summary(
        store,
        [
            {"round_idx": 1, "patchset_id": "p1", "final_eval_id": "e1", "status": "ok"},
            {"round_idx": 2, "patchset_id": "p2", "static_eval_id": "e2", "status": "compile_error"},
        ],
        requested=2,
    )

    assert summary.terminal_state == "complete"
    assert summary.evaluated == 1
    assert [row["terminal_outcome"] for row in summary.results] == ["evaluated", "static_rejected"]


def test_uniref_digest_cache_invalidates_same_path_same_size_content_change(tmp_path: Path):
    from seq_align.c_patch_loop.runner import _file_info

    uniref = tmp_path / "uniref.fasta"
    cache_dir = tmp_path / "digest-cache"
    uniref.write_bytes(b">a\nAAAA\n")
    first = _file_info(uniref, digest_cache_dir=cache_dir)
    first_stat = uniref.stat()

    replacement = tmp_path / "replacement.fasta"
    replacement.write_bytes(b">b\nCCCC\n")
    assert replacement.stat().st_size == first_stat.st_size
    os.replace(replacement, uniref)
    second = _file_info(uniref, digest_cache_dir=cache_dir)

    assert second["path"] == first["path"]
    assert second["size"] == first["size"]
    assert (second["device"], second["inode"], second["mtime_ns"]) != (
        first["device"], first["inode"], first["mtime_ns"]
    )
    assert second["sha256"] != first["sha256"]


def test_uniref_digest_cache_reuses_matching_stat_identity(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from seq_align.c_patch_loop import runner as runner_module

    uniref = tmp_path / "uniref.fasta"
    cache_dir = tmp_path / "digest-cache"
    uniref.write_bytes(b">a\nAAAA\n")
    first = runner_module._file_info(uniref, digest_cache_dir=cache_dir)
    monkeypatch.setattr(runner_module, "hash_file", lambda path: (_ for _ in ()).throw(AssertionError("rehash")))

    second = runner_module._file_info(uniref, digest_cache_dir=cache_dir)

    assert second["sha256"] == first["sha256"]


def test_controller_cached_eval_reuse_requires_matching_fingerprint(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    store = LoopStore(tmp_path)
    runner = PatchRunner(store)
    cached = {
        "eval_id": "cached",
        "patchset_id": "stock",
        "stage": "full",
        "status": "ok",
        "decision": "accepted_baseline",
        "eval_result": {"status": "ok", "tp": 10, "fp": 1, "tp_fp_ratio": 10 / 11, "cost_s": 1.0},
        "eval_config": {"reuse_fingerprint": "matching"},
    }
    store.write_index("baseline_full", cached)
    monkeypatch.setattr(runner, "reuse_fingerprint", lambda **kwargs: "matching")

    assert controller._baseline_summary(store, runner=runner).eval_id == "cached"

    cached["eval_config"] = {}
    store.write_index("baseline_full", cached)
    assert controller._baseline_summary(store, runner=runner) is None

    cached["eval_config"] = {"reuse_fingerprint": "stale"}
    store.write_index("baseline_full", cached)
    assert controller._baseline_summary(store, runner=runner) is None


def _legacy_slurm_baseline_fixture(tmp_path: Path):
    from seq_align.c_patch_loop import runner as runner_module

    hmmer_root = tmp_path / "hmmer"
    binary_dir = hmmer_root / "src"
    binary_dir.mkdir(parents=True)
    (binary_dir / "jackhmmer").write_bytes(b"current jackhmmer binary\n")
    (binary_dir / "hmmsearch").write_bytes(b"current hmmsearch binary\n")
    runner = _checkpoint_runner(
        tmp_path,
        eval_backend="slurm",
        full_max_iter=1,
        min_speedup_fraction=0.03,
        hmmer_src_root=hmmer_root,
    )
    binary_fingerprint = runner_module._stock_binary_fingerprint(runner.hmmer_src_root)
    binary_sha256 = {
        name: str(binary_fingerprint[name]["sha256"])
        for name in ("jackhmmer", "hmmsearch")
    }
    store = runner.store
    patchset = runner.build_stock_patchset(stage="full")
    job_dir = tmp_path / "legacy-slurm"
    query_dir = job_dir / "queries"
    result_dir = job_dir / "results"
    query_dir.mkdir(parents=True)
    result_dir.mkdir()
    manifest = runner._query_manifest(runner.data_dir)
    rows = []
    costs = []
    for zero_idx, query in enumerate(manifest):
        archived = query_dir / f"query_{zero_idx:03d}.fasta"
        archived.write_bytes(Path(query["query_path"]).read_bytes())
        cost_s = float(zero_idx + 1)
        costs.append(cost_s)
        query_result = EvalResult(
            stage="full",
            status="ok",
            tp=5,
            fp=0,
            tp_fp_ratio=1.0,
            cost_s=cost_s,
            n_seqs=1,
            iterations=1,
            meta={
                "query_index": zero_idx,
                "query_id": query["query_id"],
                "max_iter": 1,
                "jackhmmer_sha256": binary_sha256["jackhmmer"],
                "hmmsearch_sha256": binary_sha256["hmmsearch"],
                "build": {"runtime_effective": True},
            },
        )
        (result_dir / f"result_{zero_idx:03d}.json").write_text(
            json.dumps(query_result.model_dump(mode="json")), encoding="utf-8"
        )
        rows.append({
            "query_index": zero_idx,
            "query_id": query["query_id"],
            "status": "ok",
            "tp": 5,
            "fp": 0,
            "cost_s": cost_s,
            "iterations": 1,
        })
    eval_config = runner._eval_config(
        stage="full",
        max_iter=1,
        work_root=job_dir / "work",
        data_dir=runner.data_dir,
        uniref_db=runner.uniref_db,
    )
    eval_config.pop("query_timeout", None)
    request = {
        "stage": "full",
        "mods": [],
        "query_paths": [str(query_dir / f"query_{idx:03d}.fasta") for idx in range(len(manifest))],
        "max_iter": 1,
        "source_paths": {
            key: eval_config[key]
            for key in ("data_dir", "astral_db", "uniref_db", "scop_txt")
        },
    }
    (job_dir / "request.json").write_text(json.dumps(request), encoding="utf-8")
    (job_dir / "submission.json").write_text(json.dumps({"job_id": "legacy-job"}), encoding="utf-8")
    aggregate = EvalResult(
        stage="full",
        status="ok",
        tp=5 * len(manifest),
        fp=0,
        tp_fp_ratio=1.0,
        cost_s=round(sum(costs), 4),
        n_seqs=len(manifest),
        iterations=1,
        meta={
            "execution": "slurm_array",
            "job_dir": str(job_dir),
            "query_count": len(manifest),
            "query_results": rows,
            "query_cost_s": costs,
            "build": {"runtime_effective": True},
        },
    )
    dataset_manifest = {
        "data_dir": str(runner.data_dir),
        "query_count": len(manifest),
        "queries": [Path(item["query_path"]).name for item in manifest],
        "astral_db": runner_module._file_info(runner.astral_db),
        "uniref_db": runner_module._file_info(runner.uniref_db, sha256=False),
        "scop_txt": runner_module._file_info(runner.scop_txt),
    }
    record = EvaluationRecord(
        patchset_id=patchset.patchset_id or "",
        round_idx=0,
        stage="full",
        status="ok",
        decision="accepted_baseline",
        eval_result=aggregate.model_dump(mode="json"),
        eval_config=eval_config,
        dataset_manifest=dataset_manifest,
        source_hashes=runner_module._stock_source_hashes(runner.hmmer_src_root),
        created_at=utc_now(),
    )
    record.eval_id = record_hash(record, id_field="eval_id")
    store.put_evaluation(record)
    # Legacy indexes were compact summaries: the immutable evaluation carries
    # status/config/evidence and must be resolved by eval_id before backfill.
    store.write_index("baseline_full", {
        "baseline": True,
        "eval_id": record.eval_id,
        "patchset_id": record.patchset_id,
        "round_idx": 0,
        "stage": "full",
        "decision": "accepted_baseline",
        "tp": aggregate.tp,
        "fp": aggregate.fp,
        "tp_fp_ratio": aggregate.tp_fp_ratio,
        "cost_s": aggregate.cost_s,
    })
    return runner, record, job_dir


def test_legacy_slurm_baseline_is_strictly_attested_and_backfilled(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("SEQ_ALIGN_SLURM_RUNTIME", "host")
    runner, record, _ = _legacy_slurm_baseline_fixture(tmp_path)

    summary = controller._baseline_summary(runner.store, runner=runner, stage="full")

    assert summary is not None and summary.eval_id == record.eval_id
    migrated = runner.store.read_index("baseline_full")
    fingerprint = migrated["eval_config"]["reuse_fingerprint"]
    assert fingerprint == runner.reuse_fingerprint(stage="full")
    assert migrated["legacy_reuse_attestation"]["eval_id"] == record.eval_id
    attestation = runner.store.read_index("legacy_baseline_reuse_full")
    assert attestation["query_count"] == 2
    assert len(attestation["artifact_hashes"]) == 6
    # The content-addressed evaluation stays immutable; only indexes are backfilled.
    immutable = json.loads(
        (runner.store.evaluations_dir / f"{record.eval_id}.json").read_text(encoding="utf-8")
    )
    assert "reuse_fingerprint" not in immutable["eval_config"]


@pytest.mark.parametrize(
    "damage",
    [
        "missing_result",
        "query_drift",
        "config_drift",
        "binary_drift",
        "current_binary_drift",
        "uniref_identity_drift",
    ],
)
def test_legacy_slurm_baseline_backfill_fails_closed_on_inconsistent_evidence(
    tmp_path: Path,
    damage: str,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("SEQ_ALIGN_SLURM_RUNTIME", "host")
    runner, _, job_dir = _legacy_slurm_baseline_fixture(tmp_path)
    if damage == "missing_result":
        (job_dir / "results" / "result_001.json").unlink()
    elif damage == "query_drift":
        (job_dir / "queries" / "query_001.fasta").write_text(">different\nAAAA\n", encoding="utf-8")
    elif damage == "config_drift":
        runner.cpus += 1
    elif damage == "binary_drift":
        result_path = job_dir / "results" / "result_001.json"
        result = json.loads(result_path.read_text(encoding="utf-8"))
        result["meta"]["jackhmmer_sha256"] = "c" * 64
        result_path.write_text(json.dumps(result), encoding="utf-8")
    elif damage == "current_binary_drift":
        (runner.hmmer_src_root / "src" / "jackhmmer").write_bytes(
            b"modified jackhmmer binary\n"
        )
    else:
        stat = runner.uniref_db.stat()
        os.utime(
            runner.uniref_db,
            ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000),
        )

    assert controller._baseline_summary(runner.store, runner=runner, stage="full") is None
    assert runner.store.read_index("legacy_baseline_reuse_full") is None
    assert "reuse_fingerprint" not in (
        runner.store.read_index("baseline_full").get("eval_config") or {}
    )


def test_slurm_docker_runtime_identity_probes_pinned_image_once_and_validates_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    from seq_align.c_patch_loop import runner as runner_module

    image_ref = f"registry.example/hmmer@sha256:{'d' * 64}"
    monkeypatch.setenv("SEQ_ALIGN_SLURM_RUNTIME", "docker")
    monkeypatch.setenv("SEQ_ALIGN_SLURM_DOCKER_IMAGE", image_ref)
    runner = _checkpoint_runner(tmp_path, eval_backend="slurm")
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(
            command,
            0,
            (
                f"{'a' * 64}  /opt/hmmer-src/src/jackhmmer\n"
                f"{'b' * 64}  /opt/hmmer-src/src/hmmsearch\n"
            ),
            "",
        )

    monkeypatch.setattr(runner_module.subprocess, "run", fake_run)

    identity = runner.runtime_identity()
    assert runner.runtime_identity() == identity
    assert identity["image_ref"] == image_ref
    assert identity["binaries"]["jackhmmer"]["sha256"] == "a" * 64
    assert len(calls) == 1
    assert calls[0] == (
        [
            "docker",
            "run",
            "--pull=never",
            "--rm",
            "--network",
            "none",
            "--read-only",
            "--entrypoint",
            "sha256sum",
            image_ref,
            "/opt/hmmer-src/src/jackhmmer",
            "/opt/hmmer-src/src/hmmsearch",
        ],
        {"capture_output": True, "text": True, "timeout": 60},
    )

    cache_path = runner.store.root / "runtime_identities" / f"{'d' * 64}.json"
    cached = json.loads(cache_path.read_text(encoding="utf-8"))
    assert record_hash(cached, id_field="runtime_identity_id") == cached["runtime_identity_id"]
    cache_path.unlink()
    assert runner.runtime_identity() == identity
    assert len(calls) == 2

    cached = json.loads(cache_path.read_text(encoding="utf-8"))
    cached["binaries"]["jackhmmer"]["sha256"] = "c" * 64
    cache_path.write_text(json.dumps(cached), encoding="utf-8")
    with pytest.raises(RuntimeIdentityUnavailable, match="failed validation"):
        runner.runtime_identity()
    assert len(calls) == 2


def test_slurm_docker_reuse_fingerprint_changes_with_pinned_image_digest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    from seq_align.c_patch_loop import runner as runner_module

    monkeypatch.setenv("SEQ_ALIGN_SLURM_RUNTIME", "docker")
    runner = _checkpoint_runner(tmp_path, eval_backend="slurm")
    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(
            command,
            0,
            (
                f"{'a' * 64}  /opt/hmmer-src/src/jackhmmer\n"
                f"{'b' * 64}  /opt/hmmer-src/src/hmmsearch\n"
            ),
            "",
        )

    monkeypatch.setattr(runner_module.subprocess, "run", fake_run)
    monkeypatch.setenv(
        "SEQ_ALIGN_SLURM_DOCKER_IMAGE",
        f"registry.example/hmmer@sha256:{'1' * 64}",
    )
    first = runner.reuse_fingerprint(stage="full")
    monkeypatch.setenv(
        "SEQ_ALIGN_SLURM_DOCKER_IMAGE",
        f"registry.example/hmmer@sha256:{'2' * 64}",
    )
    second = runner.reuse_fingerprint(stage="full")

    assert first != second
    assert len(calls) == 2


def test_legacy_slurm_docker_baseline_matches_probed_runtime_not_host_binaries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    from seq_align.c_patch_loop import runner as runner_module

    runner, record, _ = _legacy_slurm_baseline_fixture(tmp_path)
    host_fingerprint = runner_module._stock_binary_fingerprint(runner.hmmer_src_root)
    jack_hash = host_fingerprint["jackhmmer"]["sha256"]
    search_hash = host_fingerprint["hmmsearch"]["sha256"]
    (runner.hmmer_src_root / "src" / "jackhmmer").unlink()
    (runner.hmmer_src_root / "src" / "hmmsearch").unlink()
    image_ref = f"registry.example/hmmer@sha256:{'3' * 64}"
    monkeypatch.setenv("SEQ_ALIGN_SLURM_RUNTIME", "docker")
    monkeypatch.setenv("SEQ_ALIGN_SLURM_DOCKER_IMAGE", image_ref)

    def fake_run(command, **kwargs):
        return subprocess.CompletedProcess(
            command,
            0,
            (
                f"{jack_hash}  /opt/hmmer-src/src/jackhmmer\n"
                f"{search_hash}  /opt/hmmer-src/src/hmmsearch\n"
            ),
            "",
        )

    monkeypatch.setattr(runner_module.subprocess, "run", fake_run)

    summary = controller._baseline_summary(runner.store, runner=runner, stage="full")

    assert summary is not None and summary.eval_id == record.eval_id
    attestation = runner.store.read_index("legacy_baseline_reuse_full")
    assert attestation["runtime_identity"]["image_ref"] == image_ref
    assert attestation["binary_sha256"]["jackhmmer"] == jack_hash


def test_legacy_slurm_docker_baseline_rejects_runtime_probe_binary_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    from seq_align.c_patch_loop import runner as runner_module

    runner, _, _ = _legacy_slurm_baseline_fixture(tmp_path)
    monkeypatch.setenv("SEQ_ALIGN_SLURM_RUNTIME", "docker")
    monkeypatch.setenv(
        "SEQ_ALIGN_SLURM_DOCKER_IMAGE",
        f"registry.example/hmmer@sha256:{'4' * 64}",
    )

    def fake_run(command, **kwargs):
        return subprocess.CompletedProcess(
            command,
            0,
            (
                f"{'c' * 64}  /opt/hmmer-src/src/jackhmmer\n"
                f"{'d' * 64}  /opt/hmmer-src/src/hmmsearch\n"
            ),
            "",
        )

    monkeypatch.setattr(runner_module.subprocess, "run", fake_run)

    assert controller._baseline_summary(runner.store, runner=runner, stage="full") is None
    assert runner.store.read_index("legacy_baseline_reuse_full") is None


def test_runtime_identity_unavailable_prevents_automatic_full_baseline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    from seq_align.c_patch_loop import runner as runner_module

    monkeypatch.setenv("SEQ_ALIGN_SLURM_RUNTIME", "docker")
    monkeypatch.setenv(
        "SEQ_ALIGN_SLURM_DOCKER_IMAGE",
        f"registry.example/hmmer@sha256:{'5' * 64}",
    )
    monkeypatch.setattr(
        runner_module.subprocess,
        "run",
        lambda *args, **kwargs: (_ for _ in ()).throw(FileNotFoundError("docker")),
    )
    baseline_calls = []

    def forbidden_baseline(self, **kwargs):
        baseline_calls.append(kwargs)
        raise AssertionError("Full baseline must not be submitted without runtime identity")

    monkeypatch.setattr(PatchRunner, "evaluate_stock_baseline", forbidden_baseline)

    with pytest.raises(RuntimeIdentityUnavailable, match="could not probe"):
        controller.run_loop(
            run_dir=tmp_path / "controller-run",
            rounds=0,
            proposer=FixtureProposer(),
            stage="full",
            eval_backend="slurm",
        )
    assert baseline_calls == []


def test_local_runtime_identity_keeps_host_binary_fingerprint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    from seq_align.c_patch_loop import runner as runner_module

    hmmer_root = tmp_path / "local-hmmer"
    (hmmer_root / "src").mkdir(parents=True)
    (hmmer_root / "src" / "jackhmmer").write_bytes(b"local-jack")
    (hmmer_root / "src" / "hmmsearch").write_bytes(b"local-search")
    runner = _checkpoint_runner(
        tmp_path,
        eval_backend="local",
        hmmer_src_root=hmmer_root,
    )
    monkeypatch.setenv("SEQ_ALIGN_SLURM_RUNTIME", "docker")
    monkeypatch.setattr(
        runner_module.subprocess,
        "run",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("unexpected Docker probe")),
    )

    identity = runner.runtime_identity()

    assert identity["runtime"] == "host"
    assert identity["binaries"]["jackhmmer"]["sha256"] == runner_module.hash_file(
        hmmer_root / "src" / "jackhmmer"
    )



def test_reuse_fingerprint_changes_with_each_provenance_group(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    runner = PatchRunner(LoopStore(tmp_path))
    monkeypatch.setattr("seq_align.c_patch_loop.runner._stock_source_hashes", lambda root: {"source": "a"})
    monkeypatch.setattr("seq_align.c_patch_loop.runner._stock_binary_fingerprint", lambda root: {"binary": "a"})
    monkeypatch.setattr(runner, "_dataset_manifest", lambda **kwargs: {"dataset": "a"})
    monkeypatch.setattr(runner, "_eval_config", lambda **kwargs: {"option": "a", "work_root": None})
    baseline = runner.reuse_fingerprint(stage="full")

    monkeypatch.setattr("seq_align.c_patch_loop.runner._stock_source_hashes", lambda root: {"source": "b"})
    assert runner.reuse_fingerprint(stage="full") != baseline
    monkeypatch.setattr("seq_align.c_patch_loop.runner._stock_source_hashes", lambda root: {"source": "a"})
    monkeypatch.setattr("seq_align.c_patch_loop.runner._stock_binary_fingerprint", lambda root: {"binary": "b"})
    assert runner.reuse_fingerprint(stage="full") != baseline
    monkeypatch.setattr("seq_align.c_patch_loop.runner._stock_binary_fingerprint", lambda root: {"binary": "a"})
    monkeypatch.setattr(runner, "_dataset_manifest", lambda **kwargs: {"dataset": "b"})
    assert runner.reuse_fingerprint(stage="full") != baseline
    monkeypatch.setattr(runner, "_dataset_manifest", lambda **kwargs: {"dataset": "a"})
    monkeypatch.setattr(runner, "_eval_config", lambda **kwargs: {"option": "b", "work_root": None})
    assert runner.reuse_fingerprint(stage="full") != baseline
    monkeypatch.setattr(runner, "_eval_config", lambda **kwargs: {"option": "a", "work_root": None})
    monkeypatch.setattr("seq_align.c_patch_loop.runner.os.cpu_count", lambda: 123456)
    assert runner.reuse_fingerprint(stage="full") != baseline



def test_median_eval_result_uses_median_cost_for_repeated_timing():
    results = [
        EvalResult(stage="full", status="ok", tp=10, fp=1, tp_fp_ratio=10 / 11, cost_s=1.5, meta={"profile": _profile(jack=1.2, search=0.3, top_query="median")}),
        EvalResult(stage="full", status="ok", tp=10, fp=1, tp_fp_ratio=10 / 11, cost_s=1.0, meta={"profile": _profile(jack=0.8, search=0.2, top_query="fast")}),
        EvalResult(stage="full", status="ok", tp=10, fp=1, tp_fp_ratio=10 / 11, cost_s=3.0, meta={"profile": _profile(jack=2.5, search=0.5, top_query="slow")}),
    ]

    from seq_align.c_patch_loop.runner import _median_eval_result
    collapsed = _median_eval_result(results)

    assert collapsed.cost_s == 1.5
    assert collapsed.meta["timing"]["eval_repeats"] == 3
    assert collapsed.meta["profile"]["repeat_strategy"] == "median_nearest_cost_s"
    assert collapsed.meta["profile"]["selected_repeat_idx"] == 1
    assert collapsed.meta["profile"]["top_slowest_queries"][0]["query_id"] == "median"
    assert len(collapsed.meta["profile"]["repeat_summary"]) == 3
    assert collapsed.meta["profile"]["repeat_aggregate"]["dominant_command_counts"]["jackhmmer"] == 3



@pytest.mark.requires_hmmer_source
def test_stage_specific_indexes_do_not_let_smoke_overwrite_full_best(tmp_path: Path):
    store = LoopStore(tmp_path)
    runner = PatchRunner(store)
    pack = build_context_pack(store, round_idx=1)
    proposal, metadata = FixtureProposer().propose(pack)
    patchset = runner.build_patchset(proposal, pack, metadata)
    full_record = EvaluationRecord(
        patchset_id=patchset.patchset_id or "",
        round_idx=1,
        stage="full",
        status="ok",
        decision="accepted_full",
        eval_result={"stage": "full", "status": "ok", "tp": 111, "fp": 1, "tp_fp_ratio": 111 / 112, "cost_s": 1.2},
        created_at=utc_now(),
    )
    full_record.eval_id = "eval_full"
    smoke_record = EvaluationRecord(
        patchset_id=patchset.patchset_id or "",
        round_idx=2,
        stage="smoke",
        status="ok",
        decision="accepted_smoke",
        eval_result={"stage": "smoke", "status": "ok", "tp": 54, "fp": 2, "tp_fp_ratio": 54 / 56, "cost_s": 0.7},
        created_at=utc_now(),
    )
    smoke_record.eval_id = "eval_smoke"

    runner._update_indexes(patchset, full_record)
    runner._update_indexes(patchset, smoke_record)

    assert store.read_index("current_best")["stage"] == "full"
    assert store.read_index("current_best_full")["eval_id"] == "eval_full"
    assert store.read_index("current_best_smoke")["eval_id"] == "eval_smoke"



def test_controller_reads_full_best_when_smoke_best_exists(tmp_path: Path):
    store = LoopStore(tmp_path)
    store.write_index("current_best_full", {
        "eval_id": "eval_full",
        "patchset_id": "patch_full",
        "stage": "full",
        "status": "ok",
        "tp": 111,
        "fp": 1,
        "tp_fp_ratio": 111 / 112,
        "cost_s": 1.2,
    })
    store.write_index("current_best", {
        "eval_id": "eval_smoke",
        "patchset_id": "patch_smoke",
        "stage": "smoke",
        "status": "ok",
        "tp": 54,
        "fp": 2,
        "tp_fp_ratio": 54 / 56,
        "cost_s": 0.7,
    })

    current_best = controller._current_best_summary(store)

    assert current_best is not None
    assert current_best.stage == "full"
    assert current_best.eval_id == "eval_full"



def test_controller_recovers_full_best_from_evals_with_tp_floor(tmp_path: Path):
    store = LoopStore(tmp_path)
    store.write_index("current_best", {
        "eval_id": "eval_smoke",
        "patchset_id": "patch_smoke",
        "stage": "smoke",
        "status": "ok",
        "tp": 54,
        "fp": 2,
        "tp_fp_ratio": 54 / 56,
        "cost_s": 0.7,
    })
    good = EvaluationRecord(
        eval_id="eval_good",
        patchset_id="patch_good",
        round_idx=6,
        stage="full",
        status="ok",
        decision="accepted_full",
        eval_result={"stage": "full", "status": "ok", "tp": 111, "fp": 1, "tp_fp_ratio": 111 / 112, "cost_s": 1.2},
        created_at=utc_now(),
    )
    degenerate = EvaluationRecord(
        eval_id="eval_degenerate",
        patchset_id="patch_degenerate",
        round_idx=7,
        stage="full",
        status="ok",
        decision="accepted_full",
        eval_result={"stage": "full", "status": "ok", "tp": 5, "fp": 0, "tp_fp_ratio": 1.0, "cost_s": 0.8},
        created_at=utc_now(),
    )
    store.put_evaluation(good)
    store.put_evaluation(degenerate)

    current_best = controller._current_best_summary(store, accept_min_tp_fraction=0.95)

    assert current_best is not None
    assert current_best.eval_id == "eval_good"
    assert current_best.tp == 111



def test_controller_initializes_speed_best_from_stock_baseline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    class NoopProposer(BaseProposer):
        def propose(self, pack):
            raise FatalProposalError("stop after baseline")

    def fake_baseline(self, *, stage, keep_work=False):
        record = EvaluationRecord(
            patchset_id="stock_patchset",
            round_idx=0,
            stage=stage,
            status="ok",
            decision="accepted_baseline",
            eval_result={"stage": stage, "status": "ok", "tp": 10, "fp": 1, "tp_fp_ratio": 10 / 11, "cost_s": 1.0, "meta": {"comparability": _comparable(stage=stage)}},
            created_at=utc_now(),
        )
        record.eval_id = "stock_eval"
        self.store.put_evaluation(record)
        self.store.write_index(f"baseline_{stage}", {"eval_id": "stock_eval", "patchset_id": "stock_patchset", "stage": stage, "status": "ok", "tp": 10, "fp": 1, "tp_fp_ratio": 10 / 11, "cost_s": 1.0})
        self.store.write_index(f"current_speed_best_{stage}", {"eval_id": "stock_eval", "patchset_id": "stock_patchset", "stage": stage, "status": "ok", "tp": 10, "fp": 1, "tp_fp_ratio": 10 / 11, "cost_s": 1.0})
        if stage == "full":
            self.store.write_index("baseline", {"eval_id": "stock_eval", "patchset_id": "stock_patchset", "stage": stage, "status": "ok", "tp": 10, "fp": 1, "tp_fp_ratio": 10 / 11, "cost_s": 1.0})
            self.store.write_index("current_speed_best", {"eval_id": "stock_eval", "patchset_id": "stock_patchset", "stage": stage, "status": "ok", "tp": 10, "fp": 1, "tp_fp_ratio": 10 / 11, "cost_s": 1.0})
        return record

    monkeypatch.setattr(controller.PatchRunner, "evaluate_stock_baseline", fake_baseline)

    results = controller.run_loop(run_dir=tmp_path / "run", rounds=1, proposer=NoopProposer(), stage="full")
    store = LoopStore(tmp_path / "run")

    assert results[0]["decision"] == "accepted_baseline"
    assert store.read_index("baseline")["eval_id"] == "stock_eval"
    assert store.read_index("current_speed_best")["eval_id"] == "stock_eval"



def test_controller_stops_on_fatal_proposal_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    def fake_baseline(self, *, stage, keep_work=False):
        record = EvaluationRecord(
            patchset_id="stock_patchset",
            round_idx=0,
            stage=stage,
            status="ok",
            decision="accepted_baseline",
            eval_result={"stage": stage, "status": "ok", "tp": 10, "fp": 1, "tp_fp_ratio": 10 / 11, "cost_s": 1.0, "meta": {"comparability": _comparable(stage=stage)}},
            created_at=utc_now(),
        )
        record.eval_id = "stock_eval"
        return record

    monkeypatch.setattr(controller.PatchRunner, "evaluate_stock_baseline", fake_baseline)

    class FatalProposer(BaseProposer):
        def __init__(self):
            self.calls = 0

        def propose(self, pack):
            self.calls += 1
            raise FatalProposalError("model authentication failed: status=401")

    proposer = FatalProposer()

    results = controller.run_loop(run_dir=tmp_path / "run", rounds=3, proposer=proposer)

    assert len(results) == 2
    assert results[0]["decision"] == "accepted_baseline"
    assert results[-1]["status"] == "fatal_proposal_error"
    assert "status=401" in results[-1]["error"]
    assert proposer.calls == 1



@pytest.mark.requires_hmmer_source
def test_controller_repairs_plan_validation_before_patch_generation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    bad_plan = PatchPlanProposal(
        component_name="null2_correction",
        target_functions=["p7_Null2_ByExpectation"],
        algorithm_hypothesis="Insert a no-op marker.",
        intended_code_change="Add unreachable no-op code.",
        runtime_effective_argument="This runtime branch should affect the binary.",
        profile_evidence_citation="No measured profile evidence is available in this context pack.",
        anti_degenerate_check="TP/recall should stay unchanged.",
    )
    good_plan = PatchPlanProposal(
        component_name="null2_correction",
        target_functions=["p7_Null2_ByExpectation"],
        algorithm_hypothesis="Adjust runtime null2 odds with a bounded data-dependent correction.",
        intended_code_change="Change executable C logic in p7_Null2_ByExpectation before degeneracy averaging.",
        runtime_effective_argument="The runtime branch and arithmetic alter rebuilt binary executable code.",
        profile_evidence_citation="No measured profile evidence is available in this context pack.",
        anti_degenerate_check="Preserve TP/recall by avoiding hit filtering or threshold changes.",
    )
    good_patch, _ = FixtureProposer().propose(build_context_pack(LoopStore(tmp_path / "fixture"), round_idx=1), plan=good_plan)

    class PlanRepairingProposer(BaseProposer):
        provider_name = "plan-repairing-test"

        def __init__(self):
            self.plan_calls = 0
            self.feedback = ""

        def propose_plan(self, pack, *, feedback=None, attempt_idx=None):
            self.plan_calls += 1
            self.feedback = feedback or ""
            plan = bad_plan if self.plan_calls == 1 else good_plan
            return plan, {"provider": self.provider_name}

        def propose(self, pack, *, plan=None):
            assert plan is good_plan
            return good_patch, {"provider": self.provider_name}

    def fake_evaluate(self, patchset, cfg, *, stage, current_best=None, quality_reference=None, keep_work=False, baseline=False):
        record = EvaluationRecord(
            patchset_id=patchset.patchset_id or "",
            round_idx=patchset.round_idx,
            stage="smoke",
            status="ok",
            decision="accepted_smoke",
            eval_result={"stage": "smoke", "status": "ok", "tp": 1, "fp": 0, "tp_fp_ratio": 1.0, "cost_s": 0.1, "meta": {"comparability": _comparable(stage="smoke")}},
            created_at=utc_now(),
        )
        record.eval_id = "eval_smoke"
        return record

    def fake_baseline(self, *, stage, keep_work=False):
        record = EvaluationRecord(
            patchset_id="stock_patchset",
            round_idx=0,
            stage=stage,
            status="ok",
            decision="accepted_baseline",
            eval_result={"stage": stage, "status": "ok", "tp": 1, "fp": 0, "tp_fp_ratio": 1.0, "cost_s": 0.2, "meta": {"comparability": _comparable(stage=stage)}},
            created_at=utc_now(),
        )
        record.eval_id = "stock_eval"
        return record

    proposer = PlanRepairingProposer()
    monkeypatch.setattr(controller.PatchRunner, "evaluate_stock_baseline", fake_baseline)
    monkeypatch.setattr(controller.PatchRunner, "evaluate", fake_evaluate)

    results = controller.run_loop(
        run_dir=tmp_path / "run",
        rounds=1,
        proposer=proposer,
        max_repair_attempts=1,
        require_algorithmic_patch=False,
    )

    result = results[-1]
    assert result["status"] == "ok"
    assert result["decision"] == "accepted_smoke"
    assert proposer.plan_calls == 2
    assert "validation_errors" in proposer.feedback
    assert "no-op" in proposer.feedback
    patch_plans = LoopStore(tmp_path / "run").iter_patch_plans()
    assert len(patch_plans) == 2
    rejected = next(
        item for item in patch_plans
        if item["validation_status"] == "rejected"
    )
    approved = next(
        item for item in patch_plans
        if item["validation_status"] == "ok"
    )
    assert approved["parent_patch_plan_id"] == rejected["patch_plan_id"]



@pytest.mark.requires_hmmer_source
def test_controller_repairs_static_validation_failures(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    bad = PatchProposal(
        component_name="pipeline_flow",
        rationale="first invalid patch",
        unified_diff=(
            "--- a/src/p7_evo_align.c\n"
            "+++ b/src/p7_evo_align.c\n"
            "@@ -40,3 +40,3 @@\n"
            "-  this context does not exist;\n"
            "+  return eslOK;\n"
        ),
    )
    fixture_pack = build_context_pack(LoopStore(tmp_path / "fixture"), round_idx=1)
    good, _ = FixtureProposer().propose(fixture_pack, plan=PatchPlanProposal(
        component_name="pipeline_flow",
        target_functions=["p7_evo_align_Run"],
        algorithm_hypothesis="Exercise a valid comparison-island repair while preserving TP/recall.",
        intended_code_change="Insert a guarded runtime marker inside the approved comparison function.",
        runtime_effective_argument="The rebuilt runtime executes the changed C function.",
        profile_evidence_citation="No measured profile evidence is available in this context pack.",
        anti_degenerate_check="No thresholds, hit reporting, or target coverage change.",
    ))

    class RepairingProposer(BaseProposer):
        provider_name = "repairing-test"

        def __init__(self):
            self.feedback = ""

        def propose_algorithm_plan(self, pack, *, feedback=None, attempt_idx=None):
            return AlgorithmPlanProposal(
                algorithm_family="pipeline_flow_strategy",
                baseline_algorithm_summary="Stock comparison delegates to the exact pipeline.",
                proposed_algorithm_change="Use a guarded comparison decision surface with exact fallback.",
                speed_mechanism="Reduce redundant setup while exact fallback preserves borderline decisions.",
                quality_preservation_argument="TP and FP remain unchanged because report/include decisions use the exact stock path.",
                target_contracts=["pipeline_flow"],
                target_functions=["p7_evo_align_Run"],
                profile_evidence_citation="No measured profile evidence is available in this context pack.",
            ), {"provider": self.provider_name}

        def propose(self, pack):
            return bad, {"provider": self.provider_name}

        def repair(self, pack, *, previous, feedback, attempt_idx):
            self.feedback = feedback
            return good, {"provider": self.provider_name, "attempt_kind": "repair"}

    def fake_evaluate(self, patchset, cfg, *, stage, current_best=None, quality_reference=None, keep_work=False, baseline=False):
        record = EvaluationRecord(
            patchset_id=patchset.patchset_id or "",
            round_idx=patchset.round_idx,
            stage="smoke",
            status="ok",
            decision="accepted_smoke",
            eval_result={"stage": "smoke", "status": "ok", "tp": 1, "fp": 0, "tp_fp_ratio": 1.0, "cost_s": 0.1, "meta": {"comparability": _comparable(stage="smoke")}},
            created_at=utc_now(),
        )
        record.eval_id = "eval_smoke"
        return record

    def fake_baseline(self, *, stage, keep_work=False):
        record = EvaluationRecord(
            patchset_id="stock_patchset",
            round_idx=0,
            stage=stage,
            status="ok",
            decision="accepted_baseline",
            eval_result={"stage": stage, "status": "ok", "tp": 1, "fp": 0, "tp_fp_ratio": 1.0, "cost_s": 0.2, "meta": {"comparability": _comparable(stage=stage)}},
            created_at=utc_now(),
        )
        record.eval_id = "stock_eval"
        return record

    proposer = RepairingProposer()
    monkeypatch.setattr(controller.PatchRunner, "evaluate_stock_baseline", fake_baseline)
    monkeypatch.setattr(controller.PatchRunner, "evaluate", fake_evaluate)

    results = controller.run_loop(
        run_dir=tmp_path / "run",
        rounds=1,
        proposer=proposer,
        max_repair_attempts=1,
        require_algorithmic_patch=False,
    )

    result = results[-1]
    assert result["status"] == "ok"
    assert result["decision"] == "accepted_smoke"
    assert result["repair_attempts_used"] == 1
    assert "git apply --check failed" in proposer.feedback



@pytest.mark.requires_hmmer_source
def test_controller_repairs_runtime_patch_apply_failures(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    bad, _ = FixtureProposer().propose(build_context_pack(LoopStore(tmp_path / "bad_fixture"), round_idx=1))
    good, _ = FixtureProposer().propose(build_context_pack(LoopStore(tmp_path / "good_fixture"), round_idx=1))

    class RuntimeRepairingProposer(BaseProposer):
        provider_name = "runtime-repairing-test"

        def __init__(self):
            self.feedback = ""
            self.repair_plan = None
            self.repair_algorithm_plan = None

        def propose_plan(self, pack, *, feedback=None, attempt_idx=None):
            return PatchPlanProposal(
                component_name="null2_correction",
                target_functions=["p7_Null2_ByExpectation"],
                algorithm_hypothesis="Remove redundant runtime work while preserving null2 correction semantics.",
                intended_code_change="Change executable arithmetic inside p7_Null2_ByExpectation without altering thresholds or suppressing hits.",
                runtime_effective_argument="The changed arithmetic is in rebuilt runtime C code, so jackhmmer and hmmsearch binary hashes should change.",
                profile_evidence_citation="No measured profile evidence is available in this context pack.",
                anti_degenerate_check="TP/recall and FP should remain unchanged because reported hits and score thresholds are not changed.",
            ), {"provider": self.provider_name}

        def propose(self, pack, *, plan=None):
            return bad, {"provider": self.provider_name}

        def repair(
            self,
            pack,
            *,
            previous,
            feedback,
            attempt_idx,
            plan=None,
            algorithm_plan=None,
        ):
            self.feedback = feedback
            self.repair_plan = plan
            self.repair_algorithm_plan = algorithm_plan
            return good, {"provider": self.provider_name, "attempt_kind": "repair"}

    def fake_static_verify(self, patchset):
        record = EvaluationRecord(
            patchset_id=patchset.patchset_id or "",
            round_idx=patchset.round_idx,
            stage="static_verify",
            status="ok",
            decision="needs_smoke",
            created_at=utc_now(),
        )
        record.eval_id = f"static_{patchset.proposal_metadata.get('attempt_idx', 0)}"
        return record, object()

    def fake_evaluate(self, patchset, cfg, *, stage, current_best=None, quality_reference=None, keep_work=False, baseline=False):
        if baseline:
            record = EvaluationRecord(
                patchset_id="stock_patchset",
                round_idx=0,
                stage=stage,
                status="ok",
                decision="accepted_baseline",
                eval_result={"stage": stage, "status": "ok", "tp": 1, "fp": 0, "tp_fp_ratio": 1.0, "cost_s": 0.2, "meta": {"comparability": _comparable(stage=stage)}},
                created_at=utc_now(),
            )
            record.eval_id = "stock_eval"
            return record
        attempt = patchset.proposal_metadata.get("attempt_idx", 0)
        if attempt == 0:
            result = EvalResult(
                stage=stage,
                status="run_error",
                errors=["compiled HMMER setup failed: RuntimeError: git apply --check exited 1; stderr='error: patch failed: src/impl_sse/null2.c:12\\nerror: patch does not apply'"],
                meta={
                    "repeat_results": [
                        {"repeat_idx": 1, "status": "compile_error", "errors": ["git apply --check failed: patch does not apply"]}
                    ]
                },
            )
            record = EvaluationRecord(
                patchset_id=patchset.patchset_id or "",
                round_idx=patchset.round_idx,
                stage=stage,
                status="run_error",
                decision="rejected_run_error",
                eval_result=result.model_dump(mode="json"),
                errors=list(result.errors),
                created_at=utc_now(),
            )
            record.eval_id = "eval_apply_fail"
            return record
        record = EvaluationRecord(
            patchset_id=patchset.patchset_id or "",
            round_idx=patchset.round_idx,
            stage=stage,
            status="ok",
            decision="accepted_smoke",
            eval_result={"stage": stage, "status": "ok", "tp": 1, "fp": 0, "tp_fp_ratio": 1.0, "cost_s": 0.1, "meta": {"comparability": _comparable(stage=stage)}},
            created_at=utc_now(),
        )
        record.eval_id = "eval_repaired"
        return record

    proposer = RuntimeRepairingProposer()
    monkeypatch.setattr(controller.PatchRunner, "evaluate_stock_baseline", lambda self, *, stage, keep_work=False: fake_evaluate(self, None, None, stage=stage, baseline=True))
    monkeypatch.setattr(controller.PatchRunner, "static_verify", fake_static_verify)
    monkeypatch.setattr(controller.PatchRunner, "evaluate", fake_evaluate)

    results = controller.run_loop(
        run_dir=tmp_path / "run",
        rounds=1,
        proposer=proposer,
        max_repair_attempts=1,
        require_algorithmic_patch=False,
    )

    assert results[-1]["status"] == "ok"
    assert results[-1]["decision"] == "accepted_smoke"
    assert results[-1]["repair_attempts_used"] == 1
    assert "patch does not apply" in proposer.feedback
    assert isinstance(proposer.repair_plan, PatchPlanProposal)
    assert isinstance(proposer.repair_algorithm_plan, AlgorithmPlanProposal)
    assert proposer.repair_plan.component_name == "null2_correction"
    assert (
        proposer.repair_algorithm_plan.target_contracts
        == ["null2_correction"]
    )



def test_controller_does_not_repair_normal_runtime_failures():
    record = EvaluationRecord(
        patchset_id="patch",
        round_idx=1,
        stage="smoke",
        status="run_error",
        decision="rejected_run_error",
        eval_result={"stage": "smoke", "status": "run_error", "errors": ["jackhmmer failed: signal 9"]},
        errors=["jackhmmer failed: signal 9"],
        created_at=utc_now(),
    )

    assert controller._should_repair_final(record) is False



def test_repair_feedback_is_structured_sanitized_json():
    fake_bearer = "s" + "k-TEST_ONLY_BEARER_NOT_A_REAL_SECRET"
    fake_env_key = "s" + "k-TEST_ONLY_ENV_NOT_A_REAL_SECRET"
    record = EvaluationRecord(
        eval_id="eval_secret",
        patchset_id="patchset",
        round_idx=1,
        stage="smoke",
        status="run_error",
        decision="rejected_run_error",
        eval_result={
            "stage": "smoke",
            "status": "run_error",
            "tp": 1,
            "fp": 0,
            "tp_fp_ratio": 1.0,
            "cost_s": 1.5,
            "errors": [f"Authorization: Bearer {fake_bearer} illegal character"],
            "meta": {"repeat_results": [{"errors": ["repeat parser failed: bad file format"]}]},
        },
        metric_delta={"cost_s_delta": 0.5, "tp_delta": 0, "fp_delta": 0},
        build={"runtime_effective": False, "comparisons": {"binaries": {"jackhmmer": {"changed": False}}}},
        errors=[f"OPENAI_API_KEY={fake_env_key} should not leak"],
        created_at=utc_now(),
    )

    feedback = controller._repair_feedback(record)
    payload = json.loads(feedback)

    assert payload["feedback_kind"] == "patch_repair"
    item = payload["validation_records"][0]
    assert item["eval_id"] == "eval_secret"
    assert item["failure_kind"] == "semantic_runtime_failure"
    assert item["metrics"]["cost_s"] == 1.5
    assert item["metric_delta"]["cost_s_delta"] == 0.5
    assert item["runtime_effective"] is False
    assert item["build_comparisons"]["binaries"]["jackhmmer"] is False
    assert "bad file format" in json.dumps(item)
    assert fake_bearer not in feedback
    assert fake_env_key not in feedback
    assert "<redacted" in feedback



@pytest.mark.requires_hmmer_source
def test_controller_replans_after_static_repair_exhaustion(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    first_plan = PatchPlanProposal(
        component_name="null2_correction",
        target_functions=["p7_Null2_ByExpectation"],
        algorithm_hypothesis="Change null2 runtime arithmetic while preserving TP/recall.",
        intended_code_change="Try the first executable null2 implementation path.",
        runtime_effective_argument="The arithmetic change should alter rebuilt executable code.",
        profile_evidence_citation="No measured profile evidence is available in this context pack.",
        anti_degenerate_check="TP/recall remains unchanged because thresholds and hit filters are unchanged.",
    )
    second_plan = PatchPlanProposal(
        component_name="null2_correction",
        target_functions=["p7_Null2_ByExpectation"],
        algorithm_hypothesis="Use a different null2 arithmetic path while preserving TP/recall.",
        intended_code_change="Avoid the failed diff mechanics and change executable arithmetic in a different block.",
        runtime_effective_argument="The new runtime arithmetic should alter rebuilt executable code.",
        profile_evidence_citation="No measured profile evidence is available in this context pack.",
        anti_degenerate_check="TP/recall remains unchanged because thresholds and hit filters are unchanged.",
    )
    patch, _ = FixtureProposer().propose(build_context_pack(LoopStore(tmp_path / "fixture"), round_idx=1), plan=second_plan)

    class ReplanningProposer(BaseProposer):
        provider_name = "replanning-test"

        def __init__(self):
            self.plan_calls = 0
            self.feedback = ""

        def propose_plan(self, pack, *, feedback=None, attempt_idx=None):
            self.plan_calls += 1
            self.feedback = feedback or self.feedback
            return (second_plan if feedback else first_plan), {"provider": self.provider_name}

        def propose(self, pack, *, plan=None):
            return patch, {"provider": self.provider_name}

    def fake_baseline(self, *, stage, keep_work=False):
        record = EvaluationRecord(
            patchset_id="stock_patchset",
            round_idx=0,
            stage=stage,
            status="ok",
            decision="accepted_baseline",
            eval_result={"stage": stage, "status": "ok", "tp": 1, "fp": 0, "tp_fp_ratio": 1.0, "cost_s": 0.2, "meta": {"comparability": _comparable(stage=stage)}},
            created_at=utc_now(),
        )
        record.eval_id = "stock_eval"
        return record

    def fake_static_verify(self, patchset):
        if patchset.proposal_metadata.get("plan_reproposal_idx") == 0:
            record = EvaluationRecord(
                patchset_id=patchset.patchset_id or "",
                round_idx=patchset.round_idx,
                stage="static_verify",
                status="compile_error",
                decision="rejected_static",
                errors=["compile failed: expected ';' before '}' token"],
                created_at=utc_now(),
            )
            record.eval_id = "static_fail"
            return record, None
        record = EvaluationRecord(
            patchset_id=patchset.patchset_id or "",
            round_idx=patchset.round_idx,
            stage="static_verify",
            status="ok",
            decision="needs_smoke",
            created_at=utc_now(),
        )
        record.eval_id = "static_ok"
        return record, object()

    def fake_evaluate(self, patchset, cfg, *, stage, current_best=None, quality_reference=None, keep_work=False, baseline=False):
        record = EvaluationRecord(
            patchset_id=patchset.patchset_id or "",
            round_idx=patchset.round_idx,
            stage="smoke",
            status="ok",
            decision="accepted_smoke",
            eval_result={"stage": "smoke", "status": "ok", "tp": 1, "fp": 0, "tp_fp_ratio": 1.0, "cost_s": 0.1, "meta": {"comparability": _comparable(stage="smoke")}},
            created_at=utc_now(),
        )
        record.eval_id = "eval_replanned"
        return record

    proposer = ReplanningProposer()
    monkeypatch.setattr(controller.PatchRunner, "evaluate_stock_baseline", fake_baseline)
    monkeypatch.setattr(controller.PatchRunner, "static_verify", fake_static_verify)
    monkeypatch.setattr(controller.PatchRunner, "evaluate", fake_evaluate)

    results = controller.run_loop(
        run_dir=tmp_path / "run",
        rounds=1,
        proposer=proposer,
        max_repair_attempts=0,
        max_plan_reproposal_attempts=1,
        require_algorithmic_patch=False,
    )

    assert proposer.plan_calls == 2
    assert "implementation_path_failed_after_patch_repairs" in proposer.feedback
    assert "compile_error" in proposer.feedback
    assert results[-2]["replanned_after_exhaustion"] is True
    assert results[-1]["status"] == "ok"
    assert results[-1]["plan_reproposal_idx"] == 1



@pytest.mark.requires_hmmer_source
def test_controller_replans_after_semantic_smoke_failure_exhaustion(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    first_plan = PatchPlanProposal(
        component_name="null2_correction",
        target_functions=["p7_Null2_ByExpectation"],
        algorithm_hypothesis="Change null2 runtime arithmetic while preserving TP/recall.",
        intended_code_change="Try a runtime path that may fail semantic smoke execution.",
        runtime_effective_argument="The arithmetic change should alter rebuilt executable code.",
        profile_evidence_citation="No measured profile evidence is available in this context pack.",
        anti_degenerate_check="TP/recall remains unchanged because thresholds and hit filters are unchanged.",
    )
    second_plan = PatchPlanProposal(
        component_name="null2_correction",
        target_functions=["p7_Null2_ByExpectation"],
        algorithm_hypothesis="Use a safer null2 runtime path while preserving TP/recall.",
        intended_code_change="Avoid the previous parser-crashing implementation path.",
        runtime_effective_argument="The arithmetic change should alter rebuilt executable code.",
        profile_evidence_citation="No measured profile evidence is available in this context pack.",
        anti_degenerate_check="TP/recall remains unchanged because thresholds and hit filters are unchanged.",
    )
    patch, _ = FixtureProposer().propose(build_context_pack(LoopStore(tmp_path / "fixture"), round_idx=1), plan=second_plan)

    class SmokeReplanningProposer(BaseProposer):
        provider_name = "smoke-replanning-test"

        def __init__(self):
            self.plan_calls = 0
            self.feedback = ""

        def propose_plan(self, pack, *, feedback=None, attempt_idx=None):
            self.plan_calls += 1
            self.feedback = feedback or self.feedback
            return (second_plan if feedback else first_plan), {"provider": self.provider_name}

        def propose(self, pack, *, plan=None):
            return patch, {"provider": self.provider_name}

    def fake_baseline(self, *, stage, keep_work=False):
        record = EvaluationRecord(
            patchset_id="stock_patchset",
            round_idx=0,
            stage=stage,
            status="ok",
            decision="accepted_baseline",
            eval_result={"stage": stage, "status": "ok", "tp": 1, "fp": 0, "tp_fp_ratio": 1.0, "cost_s": 0.2, "meta": {"comparability": _comparable(stage=stage)}},
            created_at=utc_now(),
        )
        record.eval_id = "stock_eval"
        return record

    def fake_static_verify(self, patchset):
        record = EvaluationRecord(
            patchset_id=patchset.patchset_id or "",
            round_idx=patchset.round_idx,
            stage="static_verify",
            status="ok",
            decision="needs_smoke",
            created_at=utc_now(),
        )
        record.eval_id = f"static_{patchset.proposal_metadata.get('plan_reproposal_idx')}"
        return record, object()

    def fake_evaluate(self, patchset, cfg, *, stage, current_best=None, quality_reference=None, keep_work=False, baseline=False):
        if patchset.proposal_metadata.get("plan_reproposal_idx") == 0:
            record = EvaluationRecord(
                patchset_id=patchset.patchset_id or "",
                round_idx=patchset.round_idx,
                stage="smoke",
                status="run_error",
                decision="rejected_run_error",
                eval_result={"stage": "smoke", "status": "run_error", "errors": ["jackhmmer failed: illegal character"]},
                errors=["jackhmmer failed: illegal character"],
                created_at=utc_now(),
            )
            record.eval_id = "eval_semantic_fail"
            return record
        record = EvaluationRecord(
            patchset_id=patchset.patchset_id or "",
            round_idx=patchset.round_idx,
            stage="smoke",
            status="ok",
            decision="accepted_smoke",
            eval_result={"stage": "smoke", "status": "ok", "tp": 1, "fp": 0, "tp_fp_ratio": 1.0, "cost_s": 0.1, "meta": {"comparability": _comparable(stage="smoke")}},
            created_at=utc_now(),
        )
        record.eval_id = "eval_replanned"
        return record

    proposer = SmokeReplanningProposer()
    monkeypatch.setattr(controller.PatchRunner, "evaluate_stock_baseline", fake_baseline)
    monkeypatch.setattr(controller.PatchRunner, "static_verify", fake_static_verify)
    monkeypatch.setattr(controller.PatchRunner, "evaluate", fake_evaluate)

    results = controller.run_loop(
        run_dir=tmp_path / "run",
        rounds=1,
        proposer=proposer,
        max_repair_attempts=0,
        max_plan_reproposal_attempts=1,
        require_algorithmic_patch=False,
    )

    assert proposer.plan_calls == 2
    assert "semantic_runtime_failure" in proposer.feedback
    assert results[-2]["replanned_after_exhaustion"] is True
    assert results[-1]["status"] == "ok"
    assert results[-1]["plan_reproposal_idx"] == 1



@pytest.mark.requires_hmmer_source
def test_controller_does_not_repair_or_replan_environmental_run_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    plan = PatchPlanProposal(
        component_name="null2_correction",
        target_functions=["p7_Null2_ByExpectation"],
        algorithm_hypothesis="Change null2 runtime arithmetic while preserving TP/recall.",
        intended_code_change="Change executable arithmetic in the selected runtime path.",
        runtime_effective_argument="The arithmetic change should alter rebuilt executable code.",
        profile_evidence_citation="No measured profile evidence is available in this context pack.",
        anti_degenerate_check="TP/recall remains unchanged because thresholds and hit filters are unchanged.",
    )
    patch, _ = FixtureProposer().propose(build_context_pack(LoopStore(tmp_path / "fixture"), round_idx=1), plan=plan)

    class EnvFailProposer(BaseProposer):
        provider_name = "env-fail-test"

        def __init__(self):
            self.plan_calls = 0
            self.repair_calls = 0

        def propose_plan(self, pack, *, feedback=None, attempt_idx=None):
            self.plan_calls += 1
            return plan, {"provider": self.provider_name}

        def propose(self, pack, *, plan=None):
            return patch, {"provider": self.provider_name}

        def repair(self, pack, *, previous, feedback, attempt_idx, plan=None):
            self.repair_calls += 1
            return patch, {"provider": self.provider_name}

    def fake_baseline(self, *, stage, keep_work=False):
        record = EvaluationRecord(
            patchset_id="stock_patchset",
            round_idx=0,
            stage=stage,
            status="ok",
            decision="accepted_baseline",
            eval_result={"stage": stage, "status": "ok", "tp": 1, "fp": 0, "tp_fp_ratio": 1.0, "cost_s": 0.2, "meta": {"comparability": _comparable(stage=stage)}},
            created_at=utc_now(),
        )
        record.eval_id = "stock_eval"
        return record

    def fake_static_verify(self, patchset):
        record = EvaluationRecord(
            patchset_id=patchset.patchset_id or "",
            round_idx=patchset.round_idx,
            stage="static_verify",
            status="ok",
            decision="needs_smoke",
            created_at=utc_now(),
        )
        record.eval_id = "static_ok"
        return record, object()

    def fake_evaluate(self, patchset, cfg, *, stage, current_best=None, quality_reference=None, keep_work=False, baseline=False):
        record = EvaluationRecord(
            patchset_id=patchset.patchset_id or "",
            round_idx=patchset.round_idx,
            stage="smoke",
            status="run_error",
            decision="rejected_run_error",
            eval_result={"stage": "smoke", "status": "run_error", "errors": ["jackhmmer failed: signal 9"]},
            errors=["jackhmmer failed: signal 9"],
            created_at=utc_now(),
        )
        record.eval_id = "eval_env_fail"
        return record

    proposer = EnvFailProposer()
    monkeypatch.setattr(controller.PatchRunner, "evaluate_stock_baseline", fake_baseline)
    monkeypatch.setattr(controller.PatchRunner, "static_verify", fake_static_verify)
    monkeypatch.setattr(controller.PatchRunner, "evaluate", fake_evaluate)

    results = controller.run_loop(
        run_dir=tmp_path / "run",
        rounds=1,
        proposer=proposer,
        max_repair_attempts=1,
        max_plan_reproposal_attempts=1,
        require_algorithmic_patch=False,
    )

    assert proposer.plan_calls == 1
    assert proposer.repair_calls == 0
    assert results[-1]["status"] == "run_error"
    assert results[-1]["failure_kind"] == "environmental_failure"



@pytest.mark.requires_hmmer_source
def test_static_failure_experience_uses_workflow_identity_and_plan_provenance(
    tmp_path: Path,
):
    from seq_align.c_patch_loop.experience import (
        implementation_mechanism_projection,
        mechanism_fingerprint,
    )

    store = LoopStore(tmp_path)
    pack = build_context_pack(store, round_idx=1)
    runner = PatchRunner(store, require_algorithmic_patch=False)
    patch_plan = runner.build_patch_plan(
        PatchPlanProposal(
            component_name="null2_correction",
            target_functions=["p7_Null2_ByExpectation"],
            algorithm_hypothesis="Preserve exact null2 scores while reducing repeated arithmetic.",
            intended_code_change="Hoist one invariant arithmetic term inside p7_Null2_ByExpectation.",
            runtime_effective_argument="The rebuilt binary executes the changed arithmetic for every scored domain.",
            profile_evidence_citation="No measured profile evidence is available in this context pack.",
            expected_metric_effect="Reduce runtime while preserving TP and FP exactly.",
            anti_degenerate_check="No thresholds, scores, or reporting decisions change.",
        ),
        pack,
        {"provider": "test"},
    )
    patchset = runner.build_patchset(
        PatchProposal(
            component_name="null2_correction",
            rationale="A deliberately different final-patch rationale that must not define experience identity.",
            unified_diff="--- a/src/generic_null2.c\n+++ b/src/generic_null2.c\n@@ -1 +1 @@\n-missing\n+replacement\n",
        ),
        pack,
        {"provider": "test"},
        patch_plan=patch_plan,
    )

    record, cfg = runner.static_verify(patchset)

    assert record.status != "ok"
    assert cfg is None
    experiences = store.iter_experience()
    persisted = next(
        item for item in experiences
        if item.get("source_patchset_ids") == [patchset.patchset_id]
    )
    patch_plan_fingerprint = mechanism_fingerprint(
        implementation_mechanism_projection(patch_plan)
    )
    final_patch_fingerprint = mechanism_fingerprint(
        implementation_mechanism_projection(patchset.patches[0])
    )
    assert persisted["scope"] == "workflow"
    assert persisted["mechanism_projection"]["track"] == "workflow"
    assert (
        persisted["mechanism_projection"]["outcome"]
        == "patch_apply_failure"
    )
    assert (
        persisted["mechanism_projection"]["failure_stage"]
        == "static_verify"
    )
    assert persisted["source_patch_plan_ids"] == [
        patch_plan.patch_plan_id
    ]
    assert persisted["mechanism_fingerprint"] not in {
        patch_plan_fingerprint,
        final_patch_fingerprint,
    }


@pytest.mark.requires_hmmer_source
def test_experience_indexes_separate_workflow_and_algorithm_lessons(tmp_path: Path):
    store = LoopStore(tmp_path)
    pack = build_context_pack(store, round_idx=1)
    proposal, metadata = FixtureProposer().propose(pack)
    runner = PatchRunner(store, require_algorithmic_patch=False)
    patchset = runner.build_patchset(proposal, pack, metadata)
    apply_failure = EvaluationRecord(
        patchset_id=patchset.patchset_id or "",
        round_idx=1,
        stage="smoke",
        status="run_error",
        decision="rejected_run_error",
        eval_result={"stage": "smoke", "status": "run_error", "errors": ["git apply --check failed: patch does not apply"]},
        errors=["git apply --check failed: patch does not apply"],
        created_at=utc_now(),
    )
    apply_failure.eval_id = "eval_apply_failure"

    from seq_align.c_patch_loop.experience import append_and_compress, experience_from_evaluation

    record = experience_from_evaluation(patchset, apply_failure)
    append_and_compress(store, record)

    workflow = store.read_index("experience_workflow_failures")
    stats = store.read_index("experience_component_stats")
    summary = (store.experience_dir / "compressed_summary.md").read_text(encoding="utf-8")

    assert record.outcome == "patch_apply_failure"
    assert record.scope == "workflow"
    assert record.lesson_kind == "workflow"
    assert record.mechanism_projection["track"] == "workflow"
    assert (
        record.mechanism_projection["outcome"]
        == "patch_apply_failure"
    )
    assert workflow[0]["outcome"] == "patch_apply_failure"
    assert stats[proposal.component_name]["patch_apply_failures"] == 1
    assert "Workflow / repair lessons" in summary


@pytest.mark.requires_hmmer_source
def test_workflow_failure_identity_distinguishes_apply_and_compile_failures(
    tmp_path: Path,
):
    from seq_align.c_patch_loop.experience import experience_from_evaluation

    store = LoopStore(tmp_path)
    pack = build_context_pack(store, round_idx=1)
    proposal, metadata = FixtureProposer().propose(pack)
    patchset = PatchRunner(
        store,
        require_algorithmic_patch=False,
    ).build_patchset(proposal, pack, metadata)
    apply_failure = EvaluationRecord(
        eval_id="eval_apply_failure",
        patchset_id=patchset.patchset_id or "",
        round_idx=1,
        stage="smoke",
        status="run_error",
        decision="rejected_run_error",
        eval_result={
            "stage": "smoke",
            "status": "run_error",
            "errors": ["git apply --check failed: patch does not apply"],
        },
        errors=["git apply --check failed: patch does not apply"],
        created_at=utc_now(),
    )
    compile_failure = EvaluationRecord(
        eval_id="eval_compile_failure",
        patchset_id=patchset.patchset_id or "",
        round_idx=1,
        stage="smoke",
        status="compile_error",
        decision="rejected_compile_error",
        eval_result={
            "stage": "smoke",
            "status": "compile_error",
            "errors": ["compiler failed"],
        },
        errors=["compiler failed"],
        created_at=utc_now(),
    )

    apply_record = experience_from_evaluation(
        patchset,
        apply_failure,
    )
    compile_record = experience_from_evaluation(
        patchset,
        compile_failure,
    )

    assert apply_record.scope == compile_record.scope == "workflow"
    assert apply_record.mechanism_projection["track"] == "workflow"
    assert compile_record.mechanism_projection["track"] == "workflow"
    assert apply_record.mechanism_fingerprint != (
        compile_record.mechanism_fingerprint
    )
    assert apply_record.semantic_id != compile_record.semantic_id


def test_baseline_posthoc_profile_leaves_instrumented_evidence(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """The baseline's profiling pass must land as evidence, like the candidate's.

    ``evaluate_stock_baseline`` calls ``run_posthoc_profile`` whenever the plain
    baseline evaluation returns ok. On run-prof-verify-1 only the *candidate* left
    a profiling record, so the digest's baseline line was stuck on the
    no-stage-split fallback. This pins the baseline half of the chain end to end.
    """
    store = LoopStore(tmp_path / "run")
    runner = PatchRunner(store, posthoc_profile="local")
    monkeypatch.setattr(runner, "evaluate", lambda *a, **k: _baseline_ok_record())
    monkeypatch.setattr(runner, "_run_profiling_eval", lambda ps, **_kwargs: _baseline_profiling_record())

    runner.evaluate_stock_baseline(stage="full")

    evidence = store.read_index("experience_profile_evidence") or []
    instrumented = [
        item
        for item in evidence
        if ((item.get("profile") or {}).get("stage_timing") or {}).get("time_source") == "instrumented_build"
    ]
    assert instrumented, "baseline profiling produced no instrumented experience record"

    summary = (store.experience_dir / "compressed_summary.md").read_text(encoding="utf-8")
    focus_lines = [line for line in summary.splitlines() if "stage_focus=" in line]
    assert any("ssv_filter:0.850" in line for line in focus_lines), focus_lines


def _baseline_ok_record() -> EvaluationRecord:
    return EvaluationRecord.model_construct(
        status="ok",
        stage="full",
        decision="accepted_baseline",
        patchset_id="stock",
        eval_id="baseline-eval",
        eval_result={"stage": "full", "status": "ok", "tp": 33, "fp": 0, "cost_s": 16.0, "meta": {}},
        metric_delta={},
        build={},
    )


def _baseline_profiling_record() -> EvaluationRecord:
    import time as _time

    profile = {
        "schema_version": "hmmer_cli_eval_profile.v1",
        "dominant_command": "jackhmmer",
        "command_wall_s": {"jackhmmer": 16.0},
        "command_wall_fraction": {"jackhmmer": 1.0},
        "query_cost_stats": {"count": 5, "max": 3.0},
        "top_slowest_queries": [],
        "stage_timing": {
            "time_source": "instrumented_build",
            "stage_time_available": True,
            "measured_stage_total_s": 9.0,
            "stage_time_s": {"ssv_filter": 7.65, "dp_kernel": 1.35},
            "stage_time_fraction": {"ssv_filter": 0.85, "dp_kernel": 0.15},
        },
    }
    return EvaluationRecord.model_construct(
        status="ok",
        stage="full",
        decision="accepted_baseline",
        patchset_id="stock",
        eval_id="baseline-profiling-eval",
        eval_result={"stage": "full", "status": "ok", "tp": 33, "fp": 0, "cost_s": 16.0, "meta": {"profile": profile}},
        metric_delta={"stage": "full", "decision": "accepted_baseline", "profile": profile},
        build={},
        created_at=_time.strftime("%Y-%m-%dT%H:%M:%SZ", _time.gmtime()),
    )


@pytest.mark.requires_hmmer_source
def test_experience_summarizes_profile_derived_algorithm_evidence(tmp_path: Path):
    store = LoopStore(tmp_path)
    pack = build_context_pack(store, round_idx=1)
    proposal, metadata = FixtureProposer().propose(pack)
    runner = PatchRunner(store, require_algorithmic_patch=False)
    patchset = runner.build_patchset(proposal, pack, metadata)
    evaluation = EvaluationRecord(
        patchset_id=patchset.patchset_id or "",
        round_idx=1,
        stage="full",
        status="ok",
        decision="rejected_not_faster",
        eval_result={
            "stage": "full",
            "status": "ok",
            "tp": 10,
            "fp": 1,
            "tp_fp_ratio": 10 / 11,
            "cost_s": 12.0,
            "meta": {"profile": _profile(top_query="slow_query")},
        },
        created_at=utc_now(),
    )
    evaluation.eval_id = "eval_profile"

    from seq_align.c_patch_loop.experience import append_and_compress, experience_from_evaluation

    record = experience_from_evaluation(patchset, evaluation)
    append_and_compress(store, record)

    profile_index = store.read_index("experience_profile_evidence")
    summary = (store.experience_dir / "compressed_summary.md").read_text(encoding="utf-8")

    assert record.metric_summary["profile"]["dominant_command"] == "jackhmmer"
    assert profile_index[0]["profile"]["top_slowest_queries"][0]["query_id"] == "slow_query"
    assert "Profile-derived algorithm evidence" in summary
    assert "dominant_command=jackhmmer" in summary


@pytest.mark.requires_hmmer_source
def test_experience_digest_renders_stage_time_fraction_for_the_planner(tmp_path: Path):
    """Stage attribution must reach the text the agent reads.

    ``_stage_focus`` was a dead branch until the profiling build started emitting
    real stage times: every profile carried only per-iteration elapsed, so the
    digest could never say which stage dominated. Nothing asserted the rendered
    line, so the dead branch went unnoticed. This pins the rendered form.
    """
    import json as _json

    from seq_align.c_patch_loop.experience import append_and_compress, experience_from_evaluation

    store = LoopStore(tmp_path)
    pack = build_context_pack(store, round_idx=1)
    proposal, metadata = FixtureProposer().propose(pack)
    runner = PatchRunner(store, require_algorithmic_patch=False)
    patchset = runner.build_patchset(proposal, pack, metadata)

    profile = _profile(top_query="slow_query")
    profile["stage_timing"] = {
        "time_source": "instrumented_build",
        "stage_time_available": True,
        "measured_stage_total_s": 9.3852,
        "stage_time_s": {"ssv_filter": 7.8602, "dp_kernel": 1.3608, "null2_correction": 0.1642},
        "stage_time_fraction": {"ssv_filter": 0.83751, "dp_kernel": 0.144994, "null2_correction": 0.017496},
    }
    evaluation = EvaluationRecord(
        patchset_id=patchset.patchset_id or "",
        round_idx=1,
        stage="full",
        status="ok",
        decision="rejected_not_faster",
        eval_result={
            "stage": "full",
            "status": "ok",
            "tp": 33,
            "fp": 0,
            "cost_s": 12.0,
            "meta": {"profile": profile},
        },
        created_at=utc_now(),
    )
    evaluation.eval_id = "eval_stage_focus"

    append_and_compress(store, experience_from_evaluation(patchset, evaluation))

    summary = (store.experience_dir / "compressed_summary.md").read_text(encoding="utf-8")
    focus_lines = [line for line in summary.splitlines() if "stage_focus=" in line]

    assert focus_lines, "the digest never rendered a stage attribution line"
    focus = focus_lines[0]
    # Dominant stage first, with its share, so the planner can act on it.
    assert "stage_focus=ssv_filter:0.838" in focus
    assert "dp_kernel:0.145" in focus
    assert "null2_correction:0.017" in focus
    assert _json.dumps(profile)  # serialisable into the stored evidence


@pytest.mark.requires_hmmer_source
def test_single_full_speedup_produces_typed_inner_outer_candidate_evidence(tmp_path: Path):
    import json

    from seq_align.c_patch_loop.experience import append_and_compress, experience_from_evaluation
    from seq_align.c_patch_loop.store import record_hash

    store = LoopStore(tmp_path)
    pack = build_context_pack(store, round_idx=1)
    proposal, metadata = FixtureProposer().propose(pack)
    patchset = PatchRunner(store, require_algorithmic_patch=False).build_patchset(proposal, pack, metadata)
    algorithm_plan = AlgorithmPlanRecord(
        algorithm_plan_id="algorithm-plan",
        round_idx=1,
        context_pack_id=pack.context_pack_id,
        comparison_patchset_id="stock",
        algorithm_family="pipeline_flow_strategy",
        baseline_algorithm_summary="Stock pipeline flow is the quality reference.",
        proposed_algorithm_change="Use candidate pruning with exact fallback for borderline targets.",
        speed_mechanism="Reduce downstream work for confidently rejected candidates.",
        quality_preservation_argument="Exact fallback preserves reportable and includable candidates.",
        target_contracts=["pipeline_flow"],
        target_functions=["p7_evo_align_Run"],
        profile_evidence_citation="No measured profile evidence is available.",
    )
    store.put_algorithm_plan(algorithm_plan)
    patchset = patchset.model_copy(update={"patchset_id": None, "algorithm_plan_id": "algorithm-plan", "patch_plan_id": "patch-plan"})
    patchset.patchset_id = record_hash(patchset, id_field="patchset_id")
    evaluation = EvaluationRecord(
        eval_id="eval-full",
        patchset_id=patchset.patchset_id or "patchset",
        round_idx=1,
        stage="full",
        status="ok",
        decision="accepted_speedup",
        eval_result={"stage": "full", "status": "ok", "tp": 10, "fp": 1, "cost_s": 8.0, "meta": {"timing": {"eval_repeats": 1}}},
        metric_delta={"tp_delta": 0, "fp_delta": 0, "speedup_fraction": 0.1},
    )
    store.put_evaluation(evaluation)
    store.put_patchset(patchset)

    inner = experience_from_evaluation(patchset, evaluation)
    append_and_compress(store, inner)

    records = [json.loads(line) for line in (store.experience_dir / "experience.jsonl").read_text(encoding="utf-8").splitlines()]
    assert {item["scope"] for item in records} == {"outer_algorithm", "inner_implementation"}
    assert inner.source_algorithm_plan_ids == ["algorithm-plan"]
    assert inner.source_patch_plan_ids == ["patch-plan"]
    assert inner.mechanism_fingerprint and inner.semantic_id
    candidate = store.read_index("l2_promotion_candidates")[0]
    assert candidate["promotion_confidence"] == "candidate"
    assert candidate["promotion_confirmation"]["reason"] == "single_full_eval_only"


def _checkpoint_runner(tmp_path: Path, **kwargs) -> PatchRunner:
    """Build a runner whose datasets are small local fixtures."""
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    for idx in (1, 2):
        (data_dir / f"query{idx}.fasta").write_text(f">d{idx}abcA_\nMKV\n", encoding="utf-8")
    astral = tmp_path / "astral.fasta"
    astral.write_text(">d1abcA_\nMKV\n", encoding="utf-8")
    uniref = tmp_path / "uniref.fasta"
    uniref.write_text(">u1\nMKV\n", encoding="utf-8")
    scop = tmp_path / "scop.txt"
    scop.write_text("d1abcA_\ta.1.1.1\n", encoding="utf-8")
    return PatchRunner(
        LoopStore(tmp_path / "run"),
        data_dir=data_dir,
        astral_db=astral,
        uniref_db=uniref,
        scop_txt=scop,
        **kwargs,
    )


def _query_result(idx: int, *, cost_s: float = 1.0) -> EvalResult:
    return EvalResult(
        stage="full",
        status="ok",
        tp=5,
        fp=0,
        n_seqs=1,
        iterations=1,
        cost_s=cost_s,
        meta={
            "profile": {
                "commands": {"jackhmmer": {"wall_s": cost_s}},
                "artifacts": {"hmmsearch_tbl": {"exists": True, "bytes": 10, "sha256": f"sha-{idx}"}},
            },
            "comparability": _comparable(stage="full"),
        },
    )


def _checkpointing_backend(per_query: dict[int, EvalResult], *, cost_s: float = 2.0, calls: list | None = None):
    """Fake backend that exercises the resume and per-query commit hooks."""

    def fake_run_all_queries(cfg, **kwargs):
        resumed = kwargs["resume_provider"]({
            "jackhmmer_sha256": "jack-sha",
            "hmmsearch_sha256": "search-sha",
        })
        ran: list[int] = []
        for idx in (1, 2):
            if idx in resumed:
                continue
            ran.append(idx)
            kwargs["on_query_complete"](
                idx,
                str(Path(kwargs["data_dir"]) / f"query{idx}.fasta"),
                Path(kwargs["work_root"]) / "queries" / f"query_{idx:04d}",
                per_query[idx],
            )
        if calls is not None:
            calls.append({"work_root": str(kwargs["work_root"]), "resumed": sorted(resumed), "ran": ran})
        return EvalResult(
            stage="full",
            status="ok",
            tp=10,
            fp=0,
            tp_fp_ratio=10.0,
            n_seqs=2,
            cost_s=cost_s,
            meta={"profile": _profile(jack=cost_s - 1.0, search=1.0), "build": {"runtime_effective": True}, "comparability": _comparable(stage="full")},
        )

    return fake_run_all_queries


def test_evaluation_commits_query_completions_and_attempt_lifecycle(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    runner = _checkpoint_runner(tmp_path)
    patchset = runner.build_stock_patchset(stage="full")
    per_query = {1: _query_result(1), 2: _query_result(2)}
    monkeypatch.setattr(run_hmmer_cli, "run_all_queries", _checkpointing_backend(per_query))

    record = runner.evaluate(patchset, object(), stage="full", baseline=True)

    completions = runner.store.iter_query_completions()
    attempts = runner.store.iter_evaluation_attempts()
    started = [item for item in attempts if item["status"] == "started"]
    terminal = [item for item in attempts if item["status"] == "complete"]

    assert [item["query_idx"] for item in sorted(completions, key=lambda x: x["query_idx"])] == [1, 2]
    assert {item["artifacts"]["hmmsearch_tbl"]["sha256"] for item in completions} == {"sha-1", "sha-2"}
    assert all(item["query_sha256"] for item in completions)
    assert len(started) == 1 and len(terminal) == 1
    assert terminal[0]["parent_attempt_id"] == started[0]["attempt_id"]
    assert terminal[0]["completed_queries"] == 2
    assert terminal[0]["resumed_queries"] == 0
    assert record.primary_attempt_id == terminal[0]["attempt_id"]
    assert record.evaluation_fingerprint == terminal[0]["evaluation_fingerprint"]
    assert record.resumed_queries == 0
    assert sorted(record.query_completion_ids) == sorted(
        item["query_completion_id"] for item in completions
    )
    assert record.requested_queries == 2


def test_failed_baseline_is_rejected_and_keeps_indexes_clean(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    runner = _checkpoint_runner(tmp_path)
    patchset = runner.build_stock_patchset(stage="full")

    def failing_backend(cfg, **kwargs):
        return EvalResult(
            stage="full",
            status="compile_error",
            n_seqs=0,
            cost_s=0.0,
            errors=["configure failed"],
        )

    monkeypatch.setattr(run_hmmer_cli, "run_all_queries", failing_backend)

    record = runner.evaluate(patchset, object(), stage="full", baseline=True)

    assert record.status == "compile_error"
    assert record.decision == "rejected_baseline_compile_error"
    for name in (
        "baseline_full",
        "baseline",
        "current_best_full",
        "current_best",
        "current_speed_best_full",
        "current_speed_best",
    ):
        assert runner.store.read_index(name) is None


def test_incomplete_baseline_evidence_is_not_indexed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    runner = _checkpoint_runner(tmp_path)
    patchset = runner.build_stock_patchset(stage="full")

    def incomplete_backend(cfg, **kwargs):
        kwargs["resume_provider"]({
            "jackhmmer_sha256": "jack-sha",
            "hmmsearch_sha256": "search-sha",
        })
        return EvalResult(
            stage="full",
            status="ok",
            tp=10,
            fp=0,
            tp_fp_ratio=10.0,
            n_seqs=2,
            cost_s=2.0,
            meta={"build": {"runtime_effective": True}},
        )

    monkeypatch.setattr(run_hmmer_cli, "run_all_queries", incomplete_backend)

    record = runner.evaluate(patchset, object(), stage="full", baseline=True)

    assert record.status == "ok"
    assert record.decision == "rejected_baseline_incomplete_evidence"
    assert record.completed_queries == 2
    assert record.query_completion_ids == []
    assert runner.store.read_index("baseline_full") is None
    assert runner.store.read_index("current_best_full") is None
    assert runner.store.read_index("current_speed_best_full") is None


def test_second_evaluation_resumes_committed_queries(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    runner = _checkpoint_runner(tmp_path)
    patchset = runner.build_stock_patchset(stage="full")
    per_query = {1: _query_result(1), 2: _query_result(2)}
    monkeypatch.setattr(run_hmmer_cli, "run_all_queries", _checkpointing_backend(per_query))
    runner.evaluate(patchset, object(), stage="full", baseline=True)

    calls: list = []
    resumed_runner = _checkpoint_runner(tmp_path)
    monkeypatch.setattr(
        run_hmmer_cli,
        "run_all_queries",
        _checkpointing_backend(per_query, calls=calls),
    )
    resumed_record = resumed_runner.evaluate(patchset, object(), stage="full", baseline=True)

    assert calls[0]["resumed"] == [1, 2]
    assert calls[0]["ran"] == []
    assert resumed_record.resumed_queries == 2
    assert resumed_record.evaluation_fingerprint
    assert len(resumed_runner.store.iter_query_completions()) == 2


def test_resume_rejects_completions_after_query_content_changes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    runner = _checkpoint_runner(tmp_path)
    patchset = runner.build_stock_patchset(stage="full")
    per_query = {1: _query_result(1), 2: _query_result(2)}
    monkeypatch.setattr(run_hmmer_cli, "run_all_queries", _checkpointing_backend(per_query))
    runner.evaluate(patchset, object(), stage="full", baseline=True)

    calls: list = []
    reloaded = _checkpoint_runner(tmp_path)
    (tmp_path / "data" / "query2.fasta").write_text(">d2abcA_\nMKVQ\n", encoding="utf-8")
    monkeypatch.setattr(
        run_hmmer_cli,
        "run_all_queries",
        _checkpointing_backend(per_query, calls=calls),
    )
    reloaded.evaluate(patchset, object(), stage="full", baseline=True)

    assert calls[0]["resumed"] == []
    assert calls[0]["ran"] == [1, 2]


def test_interrupted_evaluation_records_attempt_and_keeps_indexes_clean(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    runner = _checkpoint_runner(tmp_path)
    patchset = runner.build_stock_patchset(stage="full")

    def interrupting_backend(cfg, **kwargs):
        kwargs["resume_provider"]({"jackhmmer_sha256": "jack-sha", "hmmsearch_sha256": "search-sha"})
        kwargs["on_query_complete"](
            1,
            str(Path(kwargs["data_dir"]) / "query1.fasta"),
            Path(kwargs["work_root"]) / "queries" / "query_0001",
            _query_result(1),
        )
        raise run_hmmer_cli.EvaluationInterrupted("stop requested")

    monkeypatch.setattr(run_hmmer_cli, "run_all_queries", interrupting_backend)

    with pytest.raises(run_hmmer_cli.EvaluationInterrupted):
        runner.evaluate(patchset, object(), stage="full", baseline=True)

    attempts = runner.store.iter_evaluation_attempts()
    interrupted = [item for item in attempts if item["status"] == "interrupted"]

    assert len(interrupted) == 1
    assert interrupted[0]["completed_queries"] == 1
    assert interrupted[0]["interruption"]["requested_queries"] == 2
    assert interrupted[0]["aggregate_result"] is None
    assert runner.store.iter_evaluations() == []
    assert runner.store.read_index("current_best_full") is None
    assert len(runner.store.iter_query_completions()) == 1


def test_repeat_and_paired_attempts_use_separate_checkpoint_namespaces(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    runner = _checkpoint_runner(tmp_path, eval_repeats=2, paired_current_best_repeats=1, min_speedup_fraction=0.03)
    reference_patchset = runner.build_stock_patchset(stage="full")
    candidate_patchset = runner.build_stock_patchset(stage="full")
    current_best = EvalSummary(
        eval_id="best",
        patchset_id=reference_patchset.patchset_id,
        stage="full",
        status="ok",
        tp=10,
        fp=0,
        tp_fp_ratio=10.0,
        cost_s=100.0,
        comparability=_comparable(stage="full"),
    )
    per_query = {1: _query_result(1), 2: _query_result(2)}
    calls: list = []

    def fake_run_all_queries(cfg, **kwargs):
        role = Path(kwargs["work_root"]).name
        cost = {"pair_01_candidate": 97.0, "pair_01_reference": 101.0}.get(role, 97.5)
        return _checkpointing_backend(per_query, cost_s=cost, calls=calls)(cfg, **kwargs)

    monkeypatch.setattr(run_hmmer_cli, "run_all_queries", fake_run_all_queries)

    runner.evaluate(candidate_patchset, object(), stage="full", current_best=current_best, quality_reference=current_best)

    attempts = runner.store.iter_evaluation_attempts()
    terminal = [item for item in attempts if item["status"] != "started"]
    fingerprints = {item["evaluation_fingerprint"] for item in terminal}

    assert {item["attempt_kind"] for item in terminal} == {"repeat", "paired_candidate", "paired_reference"}
    assert len(fingerprints) == len(terminal)
    assert all(call["resumed"] == [] for call in calls)


def test_reevaluate_patchset_full_runs_full_only_from_stored_patchset(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    runner = _checkpoint_runner(tmp_path)
    patchset = runner.build_stock_patchset(stage="full")
    per_query = {1: _query_result(1), 2: _query_result(2)}
    calls: list = []
    monkeypatch.setattr(run_hmmer_cli, "run_all_queries", _checkpointing_backend(per_query, calls=calls))

    record = runner.reevaluate_patchset_full(patchset.patchset_id or "")

    attempts = [item for item in runner.store.iter_evaluation_attempts() if item["status"] != "started"]

    assert record.stage == "full"
    assert len(calls) == 1
    assert [item["attempt_kind"] for item in attempts] == ["existing_patchset_full_reevaluation"]
    assert [item["stage"] for item in runner.store.iter_evaluations()] == ["full"]


# ---------------------------------------------------------------------------
# P1: smoke stage fail-closed
# ---------------------------------------------------------------------------

def test_smoke_stage_fails_closed_without_smoke_subdir(tmp_path: Path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    runner = PatchRunner(
        LoopStore(tmp_path / "run"),
        data_dir=data_dir,
        uniref_db=tmp_path / "uniref.fasta",
    )
    with pytest.raises(ValueError, match="smoke"):
        runner._data_dir_for_stage("smoke")
    # full stage never falls back or probes the smoke subset
    assert runner._data_dir_for_stage("full") == data_dir


def test_smoke_stage_fails_closed_without_smoke_uniref(tmp_path: Path):
    data_dir = tmp_path / "data"
    (data_dir / "smoke").mkdir(parents=True)
    runner = PatchRunner(
        LoopStore(tmp_path / "run"),
        data_dir=data_dir,
        uniref_db=tmp_path / "uniref.fasta",
    )
    with pytest.raises(ValueError, match="uniref_smoke"):
        runner._uniref_for_stage("smoke", data_dir)


def test_smoke_stage_uses_smoke_subset_when_present(tmp_path: Path):
    data_dir = tmp_path / "data"
    smoke = data_dir / "smoke"
    smoke.mkdir(parents=True)
    (data_dir / "uniref_smoke.fasta").write_text(">u1\nMKV\n", encoding="utf-8")
    uniref_full = tmp_path / "uniref.fasta"
    runner = PatchRunner(
        LoopStore(tmp_path / "run"),
        data_dir=data_dir,
        uniref_db=uniref_full,
    )
    assert runner._data_dir_for_stage("smoke") == smoke
    assert runner._uniref_for_stage("smoke", data_dir) == data_dir / "uniref_smoke.fasta"
    assert runner._uniref_for_stage("full", data_dir) == uniref_full


# ---------------------------------------------------------------------------
# P2: query fasta validation fail-closed
# ---------------------------------------------------------------------------

def test_validate_query_fastas_rejects_multi_record_file(tmp_path: Path):
    multi = tmp_path / "multi.fasta"
    multi.write_text(">d1abcA_\nMKV\n>d2xyzA_\nMKV\n", encoding="utf-8")
    with pytest.raises(ValueError, match="2 sequence records"):
        run_jack.validate_query_fastas([str(multi)])


def test_validate_query_fastas_rejects_duplicate_display_ids_across_files(tmp_path: Path):
    a = tmp_path / "a.fasta"
    a.write_text(">d1abcA_\nMKV\n", encoding="utf-8")
    b = tmp_path / "b.fasta"
    b.write_text(">d1abcA_\nMKV\n", encoding="utf-8")
    with pytest.raises(ValueError, match="d1abcA_"):
        run_jack.validate_query_fastas([str(a), str(b)])


def test_validate_query_fastas_accepts_clean_single_record_files(tmp_path: Path):
    a = tmp_path / "a.fasta"
    a.write_text(">d1abcA_\nMKV\n", encoding="utf-8")
    b = tmp_path / "b.fasta"
    b.write_text(">d2xyzA_\nMKV\n", encoding="utf-8")
    # must not raise
    run_jack.validate_query_fastas([str(a), str(b)])


def test_validate_query_fastas_rejects_empty_file(tmp_path: Path):
    empty = tmp_path / "empty.fasta"
    empty.write_text("", encoding="utf-8")
    with pytest.raises(ValueError):
        run_jack.validate_query_fastas([str(empty)])


def test_query_manifest_fails_closed_on_multi_record_dir_and_does_not_cache(tmp_path: Path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "bad.fasta").write_text(">d1abcA_\nMKV\n>d2xyzA_\nMKV\n", encoding="utf-8")
    runner = PatchRunner(LoopStore(tmp_path / "run"))
    with pytest.raises(ValueError):
        runner._query_manifest(data_dir)
    assert str(data_dir) not in runner._query_manifest_cache


# ---------------------------------------------------------------------------
# P3: per-query timeout
# ---------------------------------------------------------------------------

def test_run_passes_timeout_to_subprocess(monkeypatch: pytest.MonkeyPatch):
    captured = {}

    def fake_subprocess_run(cmd, **kwargs):
        captured.update(kwargs)
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(subprocess, "run", fake_subprocess_run)
    run_hmmer_cli._run(["true"], cwd=Path("."), label="x", timeout=42)
    assert captured["timeout"] == 42


def test_run_maps_timeout_expired_to_query_timeout_error(monkeypatch: pytest.MonkeyPatch):
    def fake_subprocess_run(cmd, **kwargs):
        raise subprocess.TimeoutExpired(
            cmd=cmd,
            timeout=42,
            output="prefix-STDOUT-TAIL",
            stderr="stderr-tail",
        )

    monkeypatch.setattr(subprocess, "run", fake_subprocess_run)
    with pytest.raises(run_hmmer_cli.QueryTimeoutError) as excinfo:
        run_hmmer_cli._run(["sleep", "100"], cwd=Path("."), label="jackhmmer", timeout=42)
    assert excinfo.value.timeout_s == 42
    assert excinfo.value.label == "jackhmmer"
    assert "STDOUT-TAIL" in excinfo.value.stdout_tail
    assert excinfo.value.stderr_tail == "stderr-tail"
    assert isinstance(excinfo.value.__cause__, subprocess.TimeoutExpired)
    assert isinstance(excinfo.value, run_hmmer_cli.EvaluationInterrupted)


def test_run_maps_signal_termination_to_evaluation_interrupted(monkeypatch: pytest.MonkeyPatch):
    def fake_subprocess_run(cmd, **kwargs):
        return subprocess.CompletedProcess(cmd, -15, "", "terminated")

    monkeypatch.setattr(subprocess, "run", fake_subprocess_run)
    with pytest.raises(run_hmmer_cli.EvaluationInterrupted, match="signal 15") as excinfo:
        run_hmmer_cli._run(["hmmer"], cwd=Path("."), label="jackhmmer")

    assert excinfo.value.label == "jackhmmer"
    assert excinfo.value.returncode == -15


def test_timed_run_preserves_evaluation_interrupted(monkeypatch: pytest.MonkeyPatch):
    interrupted = run_hmmer_cli.EvaluationInterrupted(
        "received signal 15", label="jackhmmer", returncode=-15
    )

    def fake_run(cmd, **kwargs):
        raise interrupted

    monkeypatch.setattr(run_hmmer_cli, "_run", fake_run)
    with pytest.raises(run_hmmer_cli.EvaluationInterrupted) as excinfo:
        run_hmmer_cli._timed_run(["hmmer"], cwd=Path("."), label="jackhmmer")

    assert excinfo.value is interrupted
    assert not isinstance(excinfo.value, run_hmmer_cli.CommandProfileError)


def test_timed_run_re_raises_query_timeout_with_wall_time(monkeypatch: pytest.MonkeyPatch):
    import time

    def fake_run(cmd, **kwargs):
        time.sleep(0.01)  # ensure the re-measured wall time is measurably > 0
        raise run_hmmer_cli.QueryTimeoutError(label="jackhmmer", timeout_s=5, wall_s=0.0)

    monkeypatch.setattr(run_hmmer_cli, "_run", fake_run)
    with pytest.raises(run_hmmer_cli.QueryTimeoutError) as excinfo:
        run_hmmer_cli._timed_run(["x"], cwd=Path("."), label="jackhmmer", timeout=5)
    # must NOT be swallowed into a CommandProfileError
    assert not isinstance(excinfo.value, run_hmmer_cli.CommandProfileError)
    assert excinfo.value.timeout_s == 5
    assert excinfo.value.wall_s > 0


def test_run_all_queries_tags_timeout_with_query_idx_and_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from seq_align.assembler import AssembledConfig

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "query1.fasta").write_text(">d1abcA_\nMKV\n", encoding="utf-8")
    (data_dir / "query2.fasta").write_text(">d2xyzA_\nMKV\n", encoding="utf-8")
    astral = tmp_path / "astral.fasta"
    astral.write_text(">d1abcA_\nMKV\n", encoding="utf-8")
    uniref = tmp_path / "uniref.fasta"
    uniref.write_text(">u1\nMKV\n", encoding="utf-8")
    scop = tmp_path / "scop.txt"
    scop.write_text("d1abcA_\ta.1.1.1\n", encoding="utf-8")
    (tmp_path / "hmmer-src").mkdir()

    monkeypatch.setattr(run_hmmer_cli, "_copy_source", lambda src, dst: dst.mkdir(parents=True, exist_ok=True))
    monkeypatch.setattr(run_hmmer_cli, "_apply_patches", lambda root, cfg: None)
    monkeypatch.setattr(run_hmmer_cli, "_build_hmmer", lambda root, cpus, **_kwargs: (Path("/bin/true"), Path("/bin/true")))
    monkeypatch.setattr(run_hmmer_cli, "_hash_file", lambda path: "fake-sha")

    def exploding_run_one(query, idx, **kwargs):
        raise run_hmmer_cli.QueryTimeoutError(label="jackhmmer", timeout_s=2.0, wall_s=2.0)

    monkeypatch.setattr(run_hmmer_cli, "_run_one_query", exploding_run_one)

    with pytest.raises(run_hmmer_cli.QueryTimeoutError) as excinfo:
        run_hmmer_cli.run_all_queries(
            AssembledConfig(),
            data_dir=data_dir,
            stage="smoke",
            max_iter=1,
            astral_db=astral,
            uniref_db=uniref,
            scop_txt=scop,
            hmmer_src_root=tmp_path / "hmmer-src",
            work_root=tmp_path / "work",
        )
    assert excinfo.value.query_idx == 1
    assert excinfo.value.query_path == str((data_dir / "query1.fasta").resolve())


def test_run_all_queries_defaults_query_timeout_to_none():
    assert inspect.signature(run_hmmer_cli.run_all_queries).parameters["query_timeout"].default is None
    assert inspect.signature(run_hmmer_cli._run).parameters["timeout"].default is None


def test_query_timeout_maps_to_interrupted_attempt_with_timed_out_query(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    runner = _checkpoint_runner(tmp_path, query_timeout=3)
    patchset = runner.build_stock_patchset(stage="full")

    def timeout_backend(cfg, **kwargs):
        kwargs["resume_provider"]({"jackhmmer_sha256": "jack-sha", "hmmsearch_sha256": "search-sha"})
        kwargs["on_query_complete"](
            1,
            str(Path(kwargs["data_dir"]) / "query1.fasta"),
            Path(kwargs["work_root"]) / "queries" / "query_0001",
            _query_result(1),
        )
        err = run_hmmer_cli.QueryTimeoutError(label="jackhmmer", timeout_s=3.0, wall_s=3.1)
        err.query_idx = 2
        err.query_path = str(Path(kwargs["data_dir"]) / "query2.fasta")
        raise err

    monkeypatch.setattr(run_hmmer_cli, "run_all_queries", timeout_backend)

    with pytest.raises(run_hmmer_cli.QueryTimeoutError):
        runner.evaluate(patchset, object(), stage="full", baseline=True)

    attempts = runner.store.iter_evaluation_attempts()
    interrupted = [item for item in attempts if item["status"] == "interrupted"]
    assert len(interrupted) == 1
    assert interrupted[0]["completed_queries"] == 1
    timed_out = interrupted[0]["interruption"]["timed_out_query"]
    assert timed_out["query_idx"] == 2
    assert timed_out["label"] == "jackhmmer"
    assert timed_out["timeout_s"] == 3.0
    assert "query2.fasta" in timed_out["query_path"]
    # the timed-out query committed no completion, so only query 1 is durable
    assert len(runner.store.iter_query_completions()) == 1
    # no EvaluationRecord is produced when the attempt is interrupted
    assert runner.store.iter_evaluations() == []


def test_controller_passes_query_timeout_cli_flag(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    captured = {}

    def fake_run_loop(**kwargs):
        captured.update(kwargs)
        return []

    monkeypatch.setattr(controller, "run_loop", fake_run_loop)
    monkeypatch.setattr(controller, "make_proposer", lambda args: object())

    assert controller.main([
        "--run-dir", str(tmp_path / "run"),
        "--query-timeout", "2.5",
    ]) == 0
    assert captured["query_timeout"] == 2.5


def test_configuration_payload_omits_query_timeout_when_unset(tmp_path: Path):
    class Runner:
        invocation_id = None

        def reuse_fingerprint(self, *, stage: str) -> str:
            return f"fingerprint:{stage}"

    runner = Runner()
    payload = controller._configuration_payload(
        {"rounds": 1, "inner_rounds": 1, "stage": "full", "query_timeout": None},
        runner,
    )
    assert "query_timeout" not in payload["options"]

    payload_set = controller._configuration_payload(
        {"rounds": 1, "inner_rounds": 1, "stage": "full", "query_timeout": 5.0},
        runner,
    )
    assert payload_set["options"]["query_timeout"] == 5.0


# ---------------------------------------------------------------------------
# P4: _open_invocation pick-order fix
# ---------------------------------------------------------------------------

def test_open_invocation_picks_latest_created_at_then_invocation_id(tmp_path: Path):
    class Runner:
        invocation_id = None

        def reuse_fingerprint(self, *, stage: str) -> str:
            return f"fingerprint:{stage}"

    store = LoopStore(tmp_path / "store")
    runner = Runner()
    arguments = {"rounds": 1, "inner_rounds": 1, "stage": "full"}
    config = controller._configuration_payload(arguments, runner)
    config_id = sha256_text(canonical_json(config))

    def put_interrupted(invocation_id: str, created_at: str) -> None:
        store.put_invocation(InvocationRecord(
            invocation_id=invocation_id,
            configuration_id=config_id,
            configuration=config,
            requested_outer=1,
            inner_budget=1,
            created_at=created_at,
        ))
        summary = InvocationSummaryRecord(
            invocation_id=invocation_id,
            configuration_id=config_id,
            requested_outer=1,
            inner_budget=1,
            terminal_state="interrupted",
            high_water_mark="0.0",
            created_at=created_at,
        )
        summary.invocation_summary_id = record_hash(summary, id_field="invocation_summary_id")
        store.put_invocation_summary(summary)

    # created_at must dominate: the newer record has a lexicographically
    # SMALLER invocation_id, so a naive file-name order would pick the wrong one.
    put_interrupted("zzz-older", "2026-08-01T00:00:00Z")
    put_interrupted("aaa-newer", "2026-08-02T00:00:00Z")
    picked = controller._open_invocation(store, runner, arguments)
    assert picked.invocation_id == "aaa-newer"

    # tie-break: identical created_at falls back to invocation_id order
    store2 = LoopStore(tmp_path / "store2")
    for inv_id, ts in (("bbb-tie", "2026-08-03T00:00:00Z"), ("ccc-tie", "2026-08-03T00:00:00Z")):
        store2.put_invocation(InvocationRecord(
            invocation_id=inv_id,
            configuration_id=config_id,
            configuration=config,
            requested_outer=1,
            inner_budget=1,
            created_at=ts,
        ))
        summary = InvocationSummaryRecord(
            invocation_id=inv_id,
            configuration_id=config_id,
            requested_outer=1,
            inner_budget=1,
            terminal_state="interrupted",
            high_water_mark="0.0",
            created_at=ts,
        )
        summary.invocation_summary_id = record_hash(summary, id_field="invocation_summary_id")
        store2.put_invocation_summary(summary)
    picked2 = controller._open_invocation(store2, runner, arguments)
    assert picked2.invocation_id == "ccc-tie"


# ---------------------------------------------------------------------------
# Resume replay: interrupted inner slot replays its patchset without reproposal
# ---------------------------------------------------------------------------

def _resume_replay_records(patchset_id: str, stage: str, decision: str) -> EvaluationRecord:
    return EvaluationRecord(
        eval_id=f"eval_{stage}_{patchset_id[:8]}",
        patchset_id=patchset_id,
        round_idx=1,
        stage=stage,
        status="ok",
        decision=decision,
        eval_result={"stage": stage, "status": "ok", "tp": 10, "fp": 0, "cost_s": 5.0, "meta": {"comparability": _comparable(stage=stage)}},
        created_at=utc_now(),
    )


@pytest.mark.requires_hmmer_source
def test_resume_replays_interrupted_inner_slot_without_reproposal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    run_dir = tmp_path / "run"
    baseline = EvalSummary(
        eval_id="baseline",
        patchset_id=None,
        stage="full",
        status="ok",
        tp=10,
        fp=0,
        tp_fp_ratio=10.0,
        cost_s=100.0,
        comparability=_comparable(stage="full"),
    )
    monkeypatch.setattr(
        controller,
        "_baseline_summary",
        lambda store, runner=None, stage="full": baseline,
    )
    monkeypatch.setattr(
        controller,
        "_current_best_summary",
        lambda store, runner=None, stage="full", accept_min_tp_fraction=1.0: None,
    )

    def fake_static_verify(self, patchset):
        record = EvaluationRecord(
            eval_id=f"static_{patchset.patchset_id[:8]}",
            patchset_id=patchset.patchset_id,
            round_idx=1,
            stage="static_verify",
            status="ok",
            decision="needs_smoke",
            created_at=utc_now(),
        )
        self.store.put_evaluation(record)
        return record, object()

    monkeypatch.setattr(controller.PatchRunner, "static_verify", fake_static_verify)

    def flaky_evaluate(
        self,
        patchset,
        cfg,
        *,
        stage,
        current_best=None,
        quality_reference=None,
        keep_work=False,
        baseline=False,
        attempt_kind="primary",
        # The controller now always passes these two: ``commit_indexes`` is a safety
        # flag (an inner trial must be able to refuse writing the authoritative
        # indexes) and ``paired_reference`` binds the paired confirmation to this
        # round's parent. The stub mirrors the real signature instead of swallowing
        # them in ``**kwargs`` so a future drift still fails loudly.
        commit_indexes=True,
        paired_reference=None,
        write_legacy_result=True,
        query_timeout_reference=None,
    ):
        if stage == "smoke":
            return _resume_replay_records(patchset.patchset_id, "smoke", "accepted_smoke")
        raise run_hmmer_cli.EvaluationInterrupted("stop after smoke")

    monkeypatch.setattr(controller.PatchRunner, "evaluate", flaky_evaluate)

    # Phase 1: real fixture proposal/build; full evaluation interrupted.
    with pytest.raises(run_hmmer_cli.EvaluationInterrupted):
        controller.run_hierarchical_loop(
            run_dir=run_dir,
            outer_rounds=1,
            inner_trials=1,
            proposer=FixtureProposer(),
            cpus=1,
            max_iter=1,
            full_max_iter=1,
        )

    store = LoopStore(run_dir)
    invocation = InvocationRecord.model_validate(store.iter_invocations()[0])
    heads = store.slot_heads(invocation_id=invocation.invocation_id)
    inner_heads = [head for head in heads.values() if head.inner_idx is not None]
    assert len(inner_heads) == 1
    assert inner_heads[0].state == "interrupted"
    resumed_patchset_id = (inner_heads[0].result or {}).get("patchset_id")
    assert resumed_patchset_id

    # Phase 2: identical restart must replay the stored patchset, never
    # re-propose (a deterministic proposer would regenerate the same patch plan
    # and be rejected by novelty enforcement — the pre-fix deadlock).
    class NoopProposer(BaseProposer):
        def propose(self, pack):
            raise AssertionError("patch proposal should not run during resume replay")

        def propose_algorithm_plan(self, pack, *, feedback=None, attempt_idx=None):
            raise AssertionError("algorithm plan proposal should not run during resume replay")

        def propose_plan(self, pack, **kwargs):
            raise AssertionError("patch plan proposal should not run during resume replay")

    def resume_evaluate(self, patchset, cfg, **kwargs):
        stage = kwargs["stage"]
        decision = "accepted_full" if stage == "full" else "accepted_smoke"
        return _resume_replay_records(patchset.patchset_id, stage, decision)

    monkeypatch.setattr(controller.PatchRunner, "evaluate", resume_evaluate)

    results = controller.run_hierarchical_loop(
        run_dir=run_dir,
        outer_rounds=1,
        inner_trials=1,
        proposer=NoopProposer(),
        cpus=1,
        max_iter=1,
        full_max_iter=1,
    )

    resumed = [item for item in results if item.get("resumed")]
    assert len(resumed) == 1
    assert resumed[0]["resumed_patchset_id"] == resumed_patchset_id
    assert resumed[0]["status"] == "ok"
    assert resumed[0]["decision"] == "accepted_full"
    # same invocation resumed; no second invocation was created
    assert len(list(store.iter_invocations())) == 1


def test_main_maps_query_timeout_to_exit_2(monkeypatch: pytest.MonkeyPatch):
    """A per-query timeout reaching main() must exit cleanly with code 2
    (same contract as EvaluationInterrupted), not leak a raw traceback."""

    def fake_run_loop(**kwargs):
        raise run_hmmer_cli.QueryTimeoutError(
            label="jackhmmer", timeout_s=2.0, wall_s=2.0
        )

    monkeypatch.setattr(controller, "run_loop", fake_run_loop)
    with pytest.raises(SystemExit) as excinfo:
        controller.main(["--run-dir", "runs/__test_query_timeout_exit2__"])
    assert excinfo.value.code == 2


# ---------------------------------------------------------------------------
# Post-hoc profiling pass: both outcomes must leave audit evidence
# ---------------------------------------------------------------------------

def _profile_patchset(store, tmp_path: Path) -> PatchsetRecord:
    """A minimal patchset whose only job here is to identify the profiling pass."""
    return PatchsetRecord(
        patchset_id="patchset-profile-test",
        round_idx=1,
        base_hmmer_source=str(tmp_path),
        patches=[
            PatchRecord(
                component_name="ssv_filter_neon",
                c_target_file="impl_neon/ssvfilter.c",
                c_target_functions=["p7_SSVFilter"],
                unified_diff="--- a/impl_neon/ssvfilter.c\n+++ b/impl_neon/ssvfilter.c\n",
            )
        ],
    )


def _profiling_eval_record(*, available: bool, diagnostic: dict | None = None) -> EvaluationRecord:
    stage_timing: dict = {
        "time_source": "instrumented_build" if available else "output_parse",
        "stage_time_available": available,
        "stage_time_s": {"dp_kernel": 12.5} if available else {},
        "stage_time_fraction": {"dp_kernel": 1.0} if available else {},
        "measured_stage_total_s": 12.5 if available else 0.0,
    }
    if diagnostic is not None:
        stage_timing["instrumented_profile"] = diagnostic
    profile = {
        "schema_version": "hmmer_cli_eval_profile.v1",
        "dominant_command": "jackhmmer",
        "command_wall_s": {"jackhmmer": 30.0},
        "command_wall_fraction": {"jackhmmer": 1.0},
        "query_cost_stats": {"count": 5, "max": 8.0},
        "stage_timing": stage_timing,
        "top_slowest_queries": [],
    }
    return EvaluationRecord.model_construct(
        status="ok",
        stage="full",
        decision="accepted_full",
        patchset_id="patchset-profile-test",
        eval_id="eval-profile-test",
        eval_result={"stage": "full", "status": "ok", "tp": 33, "fp": 0, "cost_s": 30.0, "meta": {"profile": profile}},
        metric_delta={"stage": "full", "decision": "accepted_full", "profile": profile, "cost_s": 30.0},
        build={},
    )


def test_posthoc_profile_success_writes_instrumented_profile_evidence(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """A working profiling pass must show up as time_source=instrumented_build."""
    store = LoopStore(tmp_path / "run")
    runner = PatchRunner(store, posthoc_profile="local")
    patchset = _profile_patchset(store, tmp_path)

    monkeypatch.setattr(
        runner, "_run_profiling_eval", lambda ps, **_kwargs: _profiling_eval_record(available=True)
    )
    record = runner.run_posthoc_profile(patchset)

    assert record is not None
    evidence = store.read_index("experience_profile_evidence") or []
    instrumented = [
        item
        for item in evidence
        if ((item.get("profile") or {}).get("stage_timing") or {}).get("time_source") == "instrumented_build"
    ]
    assert instrumented, "no experience record carries time_source=instrumented_build"
    assert instrumented[-1]["profile"]["stage_timing"]["stage_time_s"] == {"dp_kernel": 12.5}


def test_posthoc_profile_failure_records_the_reason(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """2026-10-08: the failed pass must persist *why*, not just "no stage times"."""
    store = LoopStore(tmp_path / "run")
    runner = PatchRunner(store, posthoc_profile="local")
    patchset = _profile_patchset(store, tmp_path)
    diagnostic = {"reason": "p7_evo_prof.txt not present", "path": "/q/p7_evo_prof.txt"}

    monkeypatch.setattr(
        runner,
        "_run_profiling_eval",
        lambda ps, **_kwargs: _profiling_eval_record(available=False, diagnostic=diagnostic),
    )
    record = runner.run_posthoc_profile(patchset)

    assert record is None
    failures = store.read_index("experience_workflow_failures") or []
    ours = [item for item in failures if item.get("outcome") == "posthoc_profile_failed"]
    assert ours, "a failed profiling pass left no experience record"
    assert "p7_evo_prof.txt not present" in (ours[-1].get("lesson") or "")
    assert ours[-1]["metric_summary"]["failure_stage"] == "posthoc_profile"
