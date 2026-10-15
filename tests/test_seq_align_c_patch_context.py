import pytest
import json

from seq_align.c_patch_loop.context import (
    ALGORITHM_PLAN_OUTPUT_SCHEMA,
    PLAN_OUTPUT_SCHEMA,
    _bounded_items,
    algorithm_plan_prompt,
    build_context_pack,
    context_prompt,
    inner_implementation_memory_view,
    outer_algorithm_memory_view,
    patch_repair_memory_view,
    plan_prompt,
    repair_prompt,
)
from seq_align.c_patch_loop.experience import (
    append_and_compress,
    experience_from_evaluation,
    workflow_experience,
)
from seq_align.c_patch_loop.models import (
    AlgorithmPlanProposal,
    EvalSummary,
    EvaluationRecord,
    PatchPlanProposal,
    PatchProposal,
    PatchRecord,
    PatchsetRecord,
)
from seq_align.c_patch_loop.store import LoopStore, utc_now


@pytest.mark.requires_hmmer_source
def test_context_pack_uses_c_contracts_and_has_no_unresolved_placeholders(tmp_path):
    store = LoopStore(tmp_path)
    pack = build_context_pack(store, round_idx=3)
    assert pack.context_pack_id
    names = {c.name for c in pack.allowed_c_contracts}
    assert names == {
        "pipeline_flow",
        "jackhmmer_loop",
        "dp_kernel_neon",
        "ssv_filter_neon",
        "null2_correction_neon",
        "sequence_io",
    }
    assert pack.source_excerpts
    assert all(excerpt.start_line > 0 for excerpt in pack.source_excerpts)
    assert all(excerpt.function != "<missing>" for excerpt in pack.source_excerpts)
    prompt = context_prompt(pack)
    assert "{run_dir}" not in prompt
    assert "{round_idx}" not in prompt
    assert "python_payload" in prompt  # appears only as a hard prohibition
    assert "Do not propose python_payload" in prompt
    assert "Do not submit parameter-only edits" in prompt
    assert "concrete C algorithm update" in prompt
    assert "x <= 127" in prompt
    assert "MAX_BANDS > 14" in prompt
    assert "PATCH APPLICATION POLICY" in prompt
    assert "INNER_IMPLEMENTATION_PAYLOAD_JSON" in prompt
    assert "CONTEXT_PACK_JSON" not in prompt
    assert "COMPRESSED EXPERIENCE SUMMARY" not in prompt
    assert "L1 RAW WORKING MEMORY" not in prompt
    assert "explicit-parent HMMER source excerpts" in prompt
    assert "pipeline_flow" in PLAN_OUTPUT_SCHEMA["properties"]["component_name"]["enum"]
    assert "dp_kernel_neon" in pack.required_output_schema["properties"]["component_name"]["enum"]
    assert "dp_kernel" not in pack.required_output_schema["properties"]["component_name"]["enum"]
    assert "jackhmmer_search_loop_strategy" in ALGORITHM_PLAN_OUTPUT_SCHEMA["properties"]["algorithm_family"]["enum"]
    assert "pipeline_flow_strategy" in ALGORITHM_PLAN_OUTPUT_SCHEMA["properties"]["algorithm_family"]["enum"]
    assert "hit_ranking_alignment_strategy" in ALGORITHM_PLAN_OUTPUT_SCHEMA["properties"]["algorithm_family"]["enum"]
    assert "sequence_streaming_strategy" in ALGORITHM_PLAN_OUTPUT_SCHEMA["properties"]["algorithm_family"]["enum"]


@pytest.mark.requires_hmmer_source
def test_context_pack_includes_measured_profile_evidence(tmp_path):
    store = LoopStore(tmp_path)
    profile = {
        "schema_version": "hmmer_cli_eval_profile.v1",
        "dominant_command": "jackhmmer",
        "command_wall_s": {"jackhmmer": 9.0, "hmmsearch": 1.0},
        "command_wall_fraction": {"jackhmmer": 0.9, "hmmsearch": 0.1},
        "query_cost_stats": {"count": 1, "max": 10.0},
        "top_slowest_queries": [{"query_id": "q1", "cost_s": 10.0, "dominant_command": "jackhmmer"}],
    }
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
            "meta": {"profile": profile},
        },
        created_at=utc_now(),
    )
    store.put_evaluation(record)
    summary = EvalSummary(eval_id="eval_profile", patchset_id="stock_patchset", stage="full", status="ok", tp=10, fp=1, cost_s=10.0)

    pack = build_context_pack(store, round_idx=1, baseline=summary, current_best=summary)
    patch_prompt = context_prompt(pack)
    first_prompt = plan_prompt(pack)
    outer_prompt = algorithm_plan_prompt(pack)

    assert "profile_evidence_citation" in PLAN_OUTPUT_SCHEMA["required"]
    assert "profile_evidence_citation" in ALGORITHM_PLAN_OUTPUT_SCHEMA["required"]
    assert pack.profile_evidence["has_profile"] is True
    assert pack.profile_evidence["baseline"]["profile"]["dominant_command"] == "jackhmmer"
    assert pack.profile_evidence["recommended_contracts"][0]["component_name"] == "pipeline_flow"
    assert pack.profile_evidence["recommended_contracts"][1]["component_name"] == "jackhmmer_loop"
    assert pack.profile_evidence["function_hotspots"][0]["command"] == "jackhmmer"
    assert "hmmsearch-only" in " ".join(pack.profile_evidence["avoid_targets"])
    names = {c.name for c in pack.allowed_c_contracts}
    assert {"dp_kernel_neon", "ssv_filter_neon", "sequence_io", "null2_correction_neon"}.issubset(names)
    assert {"jackhmmer_loop", "pipeline_flow"}.issubset(names)
    assert any(e.component_name == "ssv_filter_neon" and e.function == "calc_band_8" and e.start_line > 0 for e in pack.source_excerpts)
    assert any(e.component_name == "sequence_io" and e.function == "addbuf" and e.start_line > 0 for e in pack.source_excerpts)
    assert any(e.component_name == "jackhmmer_loop" and e.function == "p7_evo_iter_BeginRound" and e.start_line > 0 for e in pack.source_excerpts)
    assert any(e.component_name == "pipeline_flow" and e.function == "p7_evo_align_Run" and e.start_line > 0 for e in pack.source_excerpts)
    assert "INNER_IMPLEMENTATION_PAYLOAD_JSON" in patch_prompt
    assert "INNER_IMPLEMENTATION_MEMORY_VIEW_JSON" in first_prompt
    assert "PATCH APPLICATION POLICY" in first_prompt
    assert "CONTEXT_PACK_JSON" not in first_prompt
    assert first_prompt.count("INNER_IMPLEMENTATION_MEMORY_VIEW_JSON") == 1
    assert "L3 IMPLEMENTATION WISDOM" not in first_prompt
    assert "ALGORITHM_PLAN_OUTPUT_SCHEMA" in outer_prompt
    assert "algorithm-level speed optimization" in outer_prompt
    assert "decision surface" in outer_prompt
    assert "filter cascade" in outer_prompt
    assert "exact fallback" in outer_prompt
    assert "Input parsing and data movement are valid only under sequence_streaming_strategy" in outer_prompt
    assert "parser/streaming algorithms" in outer_prompt
    assert "profile_evidence_citation" in first_prompt
    assert "dominant_command" in patch_prompt
    assert "jackhmmer" in first_prompt


def test_context_pack_includes_pyhmmer_phase_profile_evidence(tmp_path):
    store = LoopStore(tmp_path)
    profile = {
        "schema_version": "pyhmmer_eval_profile.v1",
        "backend": "pyhmmer",
        "dominant_phase": "jackhmmer_iterations",
        "phase_wall_s": {"jackhmmer_iterations": 9.0, "final_hmmsearch": 1.0},
        "phase_wall_fraction": {"jackhmmer_iterations": 0.9, "final_hmmsearch": 0.1},
        "dominant_algorithm_phase": "jackhmmer_iterations",
        "algorithm_phase_wall_s": {"jackhmmer_iterations": 9.0, "final_hmmsearch": 1.0},
        "algorithm_phase_wall_fraction": {"jackhmmer_iterations": 0.9, "final_hmmsearch": 0.1},
        "query_cost_stats": {"count": 1, "max": 10.0},
        "iteration_stats": {"count": 1, "max": 3},
        "iteration_hit_stats": {"included": {"count": 1, "max": 8}},
        "top_slowest_queries": [
            {
                "query_id": "q1",
                "cost_s": 10.0,
                "dominant_phase": "jackhmmer_iterations",
                "phase_wall_fraction": {"jackhmmer_iterations": 0.9},
                "iteration_results": [{"iteration": 1, "hits": {"total": 20, "included": 8}}],
            }
        ],
    }
    record = EvaluationRecord(
        eval_id="eval_pyhmmer_profile",
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
            "meta": {"profile": profile},
        },
        created_at=utc_now(),
    )
    store.put_evaluation(record)
    summary = EvalSummary(eval_id="eval_pyhmmer_profile", patchset_id="stock_patchset", stage="full", status="ok", tp=10, fp=1, cost_s=10.0)

    pack = build_context_pack(store, round_idx=1, baseline=summary, current_best=summary)
    first_prompt = plan_prompt(pack)
    outer_prompt = algorithm_plan_prompt(pack)

    assert pack.profile_evidence["has_profile"] is True
    assert pack.profile_evidence["baseline"]["profile"]["dominant_phase"] == "jackhmmer_iterations"
    assert pack.profile_evidence["baseline"]["profile"]["dominant_algorithm_phase"] == "jackhmmer_iterations"
    assert pack.profile_evidence["baseline"]["profile"]["phase_wall_fraction"]["jackhmmer_iterations"] == 0.9
    assert pack.profile_evidence["recommended_contracts"][0]["component_name"] == "jackhmmer_loop"
    recommended = {item["component_name"] for item in pack.profile_evidence["recommended_contracts"]}
    assert {"jackhmmer_loop", "pipeline_flow"}.issubset(recommended)
    assert {"dp_kernel_neon", "ssv_filter_neon", "sequence_io"}.issubset(recommended)
    assert "dominant_phase=jackhmmer_iterations" in " ".join(pack.profile_evidence["observations"])
    assert "dominant_phase" in first_prompt
    assert "phase_wall_fraction" in first_prompt
    assert "dominant_phase" in outer_prompt
    assert "profile_evidence_citation" in outer_prompt


def test_context_pack_load_phase_profile_recommends_sequence_io_outer_target(tmp_path):
    store = LoopStore(tmp_path)
    profile = {
        "schema_version": "pyhmmer_eval_profile.v1",
        "backend": "pyhmmer",
        "dominant_phase": "load_targets",
        "phase_wall_s": {"load_targets": 9.0, "jackhmmer_iterations": 1.0},
        "phase_wall_fraction": {"load_targets": 0.9, "jackhmmer_iterations": 0.1},
        "dominant_algorithm_phase": "load_targets",
        "algorithm_phase_wall_s": {"load_targets": 9.0, "jackhmmer_iterations": 1.0},
        "algorithm_phase_wall_fraction": {"load_targets": 0.9, "jackhmmer_iterations": 0.1},
        "top_slowest_queries": [{"query_id": "q1", "cost_s": 10.0, "dominant_phase": "load_targets"}],
    }
    record = EvaluationRecord(
        eval_id="eval_pyhmmer_load_profile",
        patchset_id="stock_patchset",
        round_idx=0,
        stage="full",
        status="ok",
        decision="accepted_baseline",
        eval_result={"stage": "full", "status": "ok", "tp": 10, "fp": 1, "tp_fp_ratio": 10 / 11, "cost_s": 10.0, "meta": {"profile": profile}},
        created_at=utc_now(),
    )
    store.put_evaluation(record)
    summary = EvalSummary(eval_id="eval_pyhmmer_load_profile", patchset_id="stock_patchset", stage="full", status="ok", tp=10, fp=1, cost_s=10.0)

    pack = build_context_pack(store, round_idx=1, baseline=summary, current_best=summary)

    assert pack.profile_evidence["recommended_contracts"][0]["component_name"] == "sequence_io"
    assert any("Do not skip, reorder, truncate" in item for item in pack.profile_evidence["avoid_targets"])


def test_context_pack_maps_gprof_hotspots_to_new_contracts(tmp_path):
    store = LoopStore(tmp_path)
    profile = {
        "schema_version": "hmmer_cli_eval_profile.v1",
        "dominant_command": "jackhmmer",
        "command_wall_s": {"jackhmmer": 30.0, "hmmsearch": 1.0},
        "command_wall_fraction": {"jackhmmer": 0.967742, "hmmsearch": 0.032258},
        "query_cost_stats": {"count": 1, "max": 31.0},
        "top_slowest_queries": [{"query_id": "q1", "cost_s": 31.0, "dominant_command": "jackhmmer"}],
        "external_profiler": {"tool": "gprof", "status": "ok", "excluded_from_cost_s": True},
        "symbol_hotspots": [
            {
                "rank": 1,
                "command": "jackhmmer",
                "symbol": "p7_Pipeline",
                "dso": "gprof",
                "approx_command_wall_s": 9.1,
                "max_self_pct": 30.3,
                "profile_source": "gprof_flat_profile",
            },
            {
                "rank": 2,
                "command": "jackhmmer",
                "symbol": "addbuf",
                "dso": "gprof",
                "approx_command_wall_s": 8.8,
                "max_self_pct": 29.3,
                "profile_source": "gprof_flat_profile",
            },
            {
                "rank": 3,
                "command": "jackhmmer",
                "symbol": "calc_band_8",
                "dso": "gprof",
                "approx_command_wall_s": 5.3,
                "max_self_pct": 17.75,
                "profile_source": "gprof_flat_profile",
            },
        ],
    }
    record = EvaluationRecord(
        eval_id="eval_gprof_hotspots",
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
            "cost_s": 31.0,
            "meta": {"profile": profile},
        },
        created_at=utc_now(),
    )
    store.put_evaluation(record)
    summary = EvalSummary(
        eval_id="eval_gprof_hotspots",
        patchset_id="stock_patchset",
        stage="full",
        status="ok",
        tp=10,
        fp=1,
        cost_s=31.0,
    )

    pack = build_context_pack(store, round_idx=1, baseline=summary, current_best=summary)
    hotspots = pack.profile_evidence["fine_grained_hotspots"]

    assert hotspots[0]["component_name"] == "pipeline_flow"
    assert hotspots[1]["component_name"] == "sequence_io"
    assert hotspots[2]["component_name"] == "ssv_filter_neon"
    assert pack.profile_evidence["function_hotspots"][0]["function"] == "p7_Pipeline"
    assert pack.profile_evidence["contract_hotspots"][0]["component_name"] == "pipeline_flow"
    assert pack.profile_evidence["recommended_contracts"][0]["component_name"] == "pipeline_flow"
    algorithm_plan = AlgorithmPlanProposal(
        algorithm_family="ssv_filter_strategy",
        baseline_algorithm_summary="Stock SSV filtering computes band scores before downstream DP.",
        proposed_algorithm_change="Preserve SSV score semantics while reducing redundant band-scoring work.",
        speed_mechanism="Reduce repeated calc_band_8 arithmetic in the measured hotspot.",
        quality_preservation_argument="TP and FP remain unchanged because thresholds and reported hits are unchanged.",
        target_contracts=["ssv_filter_neon"],
        target_functions=["calc_band_8"],
        profile_evidence_citation="eval_gprof_hotspots reports dominant_command=jackhmmer and calc_band_8 hotspot.",
    )

    assert "calc_band_8" in plan_prompt(pack)
    assert "sequence_io" in context_prompt(pack)
    assert "APPROVED_ALGORITHM_PLAN_JSON" in plan_prompt(pack, algorithm_plan=algorithm_plan)
    assert "ssv_filter_strategy" in plan_prompt(pack, algorithm_plan=algorithm_plan)



def test_context_pack_includes_symbol_profile_and_recent_negatives(tmp_path):
    store = LoopStore(tmp_path)
    profile = {
        "schema_version": "hmmer_cli_eval_profile.v1",
        "dominant_command": "jackhmmer",
        "command_wall_s": {"jackhmmer": 20.0, "hmmsearch": 1.0},
        "command_wall_fraction": {"jackhmmer": 0.952381, "hmmsearch": 0.047619},
        "query_cost_stats": {"count": 1, "max": 21.0},
        "top_slowest_queries": [{"query_id": "q1", "cost_s": 21.0, "dominant_command": "jackhmmer"}],
        "external_profiler": {
            "tool": "perf",
            "status": "ok",
            "excluded_from_cost_s": True,
            "strategy": "post_eval_top_slowest_queries",
        },
        "symbol_hotspots": [
            {
                "rank": 1,
                "command": "jackhmmer",
                "symbol": "forward_engine",
                "dso": "jackhmmer",
                "approx_command_wall_s": 7.5,
                "max_self_pct": 37.5,
                "profile_source": "perf_report_stdio",
            }
        ],
    }
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
            "cost_s": 21.0,
            "meta": {"profile": profile},
        },
        created_at=utc_now(),
    )
    store.put_evaluation(record)
    append_and_compress(store, workflow_experience(
        outcome="plan_validation_failure",
        component_name="dp_kernel",
        lesson="dp_kernel forward_engine plan repeated a slow pattern without symbol evidence.",
        metric_summary={"round_idx": 2, "decision": "rejected_not_faster", "cost_s": 22.0},
        patch_signature={"component_name": "dp_kernel", "target_functions": ["forward_engine"], "round_idx": 2},
        negative_patterns=["quality preserved but speedup threshold not met"],
    ))
    summary = EvalSummary(
        eval_id="eval_symbol_profile",
        patchset_id="stock_patchset",
        stage="full",
        status="ok",
        tp=10,
        fp=1,
        cost_s=21.0,
    )

    pack = build_context_pack(store, round_idx=3, baseline=summary, current_best=summary)
    first_prompt = plan_prompt(pack)
    patch_prompt = context_prompt(pack)

    assert pack.profile_evidence["has_symbol_profile"] is True
    assert pack.profile_evidence["fine_grained_hotspots"][0]["symbol"] == "forward_engine"
    assert pack.profile_evidence["function_hotspots"][0]["function"] == "forward_engine"
    assert pack.profile_evidence["contract_hotspots"][0]["component_name"] == "dp_kernel_neon"
    assert pack.profile_evidence["recommended_contracts"][0]["priority"] == "measured-symbol-hotspot"
    assert pack.recent_negative_experience[0]["outcome"] == "plan_validation_failure"
    assert "INNER_IMPLEMENTATION_MEMORY_VIEW_JSON" in first_prompt
    assert "fine_grained_hotspots" in first_prompt
    assert "implementation_experience" in first_prompt
    assert "forward_engine plan repeated a slow pattern" not in first_prompt
    assert "forward_engine" in first_prompt
    assert "recent_negative_experience" in patch_prompt
    assert "forward_engine plan repeated a slow pattern" in patch_prompt


def test_memory_budget_keeps_complete_semantically_unique_items():
    duplicate_old = {"semantic_id": "same", "lesson": "older hotspot lesson"}
    duplicate_new = {"semantic_id": "same", "lesson": "newer hotspot lesson"}
    oversized = {"semantic_id": "large", "lesson": "hotspot " + ("x" * 500)}
    fitting = {"semantic_id": "fit", "lesson": "hotspot fitting lesson"}

    selected = _bounded_items(
        [duplicate_old, fitting, oversized, duplicate_new],
        terms={"hotspot"},
        quota=4,
        budget=180,
    )

    assert selected == [duplicate_new, fitting]
    encoded = json.dumps(selected, ensure_ascii=False, sort_keys=True)
    assert len(encoded) <= 180
    assert "x" * 100 not in encoded


def _prompt_json(prompt: str, marker: str) -> dict:
    payload = prompt.split(marker, 1)[1].lstrip()
    return json.JSONDecoder().raw_decode(payload)[0]


def test_inner_memory_includes_approved_outer_plan(tmp_path):
    pack = build_context_pack(LoopStore(tmp_path), round_idx=1)
    algorithm_plan = AlgorithmPlanProposal(
        algorithm_family="pipeline_flow_strategy",
        baseline_algorithm_summary="Stock pipeline evaluates the complete filter cascade.",
        proposed_algorithm_change="Add a semantics-preserving exact fallback decision surface.",
        speed_mechanism="Avoid redundant cascade work before exact fallback.",
        quality_preservation_argument="Exact fallback preserves all reported and included hits.",
        target_contracts=["pipeline_flow"],
        target_functions=["p7_evo_align_Run"],
        profile_evidence_citation="No measured profile is available.",
    )

    view = inner_implementation_memory_view(pack, algorithm_plan)

    assert view.approved_algorithm_plan is not None
    assert view.approved_algorithm_plan["algorithm_family"] == "pipeline_flow_strategy"
    assert view.approved_algorithm_plan["target_functions"] == ["p7_evo_align_Run"]
    assert "pipeline_flow_strategy" in plan_prompt(pack, algorithm_plan=algorithm_plan)


def test_final_and_repair_payloads_preserve_approved_outer_plan(tmp_path):
    pack = build_context_pack(LoopStore(tmp_path), round_idx=2)
    algorithm_plan = AlgorithmPlanProposal(
        algorithm_family="pipeline_flow_strategy",
        baseline_algorithm_summary="Stock pipeline evaluates the complete filter cascade.",
        proposed_algorithm_change="Add a semantics-preserving exact fallback decision surface.",
        speed_mechanism="Avoid redundant cascade work before exact fallback.",
        quality_preservation_argument="Exact fallback preserves all reported and included hits.",
        target_contracts=["pipeline_flow"],
        target_functions=["p7_evo_align_Run"],
        profile_evidence_citation="No measured profile is available.",
    )
    patch_plan = PatchPlanProposal(
        component_name="pipeline_flow",
        target_functions=["p7_evo_align_Run"],
        algorithm_hypothesis="The approved decision surface removes redundant work.",
        intended_code_change="Cache the exact fallback predicate inside p7_evo_align_Run.",
        runtime_effective_argument="The predicate executes in the rebuilt jackhmmer hot path.",
    )

    final_payload = _prompt_json(
        context_prompt(pack, plan=patch_plan, algorithm_plan=algorithm_plan),
        "INNER_IMPLEMENTATION_PAYLOAD_JSON:\n",
    )
    approved = final_payload["memory_view"]["approved_algorithm_plan"]
    assert approved["algorithm_family"] == "pipeline_flow_strategy"
    assert approved["proposed_algorithm_change"] == algorithm_plan.proposed_algorithm_change
    assert approved["speed_mechanism"] == algorithm_plan.speed_mechanism
    assert approved["quality_preservation_argument"] == algorithm_plan.quality_preservation_argument
    assert approved["target_contracts"] == ["pipeline_flow"]
    assert approved["target_functions"] == ["p7_evo_align_Run"]

    repair_payload = _prompt_json(
        repair_prompt(
            pack,
            previous=PatchProposal(component_name="pipeline_flow", rationale="previous patch"),
            feedback=json.dumps({"failure_kind": "plan_diff_binding"}),
            attempt_idx=1,
            plan=patch_plan,
            algorithm_plan=algorithm_plan,
        ),
        "INNER_IMPLEMENTATION_PAYLOAD_JSON:\n",
    )
    assert repair_payload["memory_view"]["approved_algorithm_plan"] == approved


def test_repair_memory_prioritizes_current_failure_and_deltas(tmp_path):
    store = LoopStore(tmp_path)
    pack = build_context_pack(store, round_idx=2)
    workflow_items = [
        {
            "experience_id": "repair-match",
            "semantic_id": "binding-repair",
            "scope": "workflow",
            "lesson": "Repair a plan_diff_binding containment failure.",
        },
        {
            "experience_id": "implementation-only",
            "semantic_id": "implementation-only",
            "scope": "inner_implementation",
            "lesson": "Do not expose this item to repair memory.",
        },
    ]
    (store.experience_dir / "experience.jsonl").write_text(
        "\n".join(json.dumps(item) for item in workflow_items) + "\n",
        encoding="utf-8",
    )
    feedback = json.dumps({
        "validation_records": [
            {"failure_kind": "old_failure", "violations": ["old"]},
            {
                "failure_kind": "plan_diff_binding",
                "violations": ["actual function escaped approved plan"],
                "plan_diff_binding": {"binding_status": "rejected", "containment": {"patch_plan": False}},
                "build_comparisons": {"jackhmmer_sha_changed": False},
            },
        ]
    })

    view = patch_repair_memory_view(pack, feedback)

    assert view.current_validation_failure["failure_kind"] == "plan_diff_binding"
    assert view.binding_delta["binding_status"] == "rejected"
    assert view.build_provenance_delta["jackhmmer_sha_changed"] is False
    assert [item["experience_id"] for item in view.repair_experience] == ["repair-match"]
    prompt = repair_prompt(
        pack,
        previous=PatchProposal(component_name="pipeline_flow", rationale="previous patch"),
        feedback=feedback,
        attempt_idx=1,
    )
    assert prompt.count("PATCH_REPAIR_MEMORY_VIEW_JSON") == 1
    assert "VALIDATOR_FEEDBACK" not in prompt
    assert "old_failure" not in prompt


def test_evaluation_workflow_failures_only_enter_repair_memory(tmp_path):
    store = LoopStore(tmp_path)
    patchset = PatchsetRecord(
        patchset_id="patchset_workflow_failure",
        round_idx=1,
        base_hmmer_source="/stock/hmmer",
        patches=[
            PatchRecord(
                component_name="pipeline_flow",
                c_target_file="src/p7_pipeline.c",
                c_target_functions=["p7_Pipeline"],
                unified_diff=(
                    "--- a/src/p7_pipeline.c\n"
                    "+++ b/src/p7_pipeline.c\n"
                    "@@ -1 +1 @@\n-old\n+new\n"
                ),
                diff_sha256="workflow-failure-diff",
                rationale="Implement approved pipeline decision change.",
            )
        ],
    )
    failures = [
        ("compile_error", "rejected_compile_error", ["compiler failed"]),
        (
            "run_error",
            "rejected_run_error",
            ["git apply --check failed: patch does not apply"],
        ),
        ("run_error", "rejected_run_error", ["jackhmmer exited 1"]),
        (
            "ok",
            "rejected_not_runtime_effective",
            ["rebuilt binary hash did not change"],
        ),
    ]
    records = []
    for index, (status, decision, errors) in enumerate(failures):
        evaluation = EvaluationRecord(
            eval_id=f"workflow_eval_{index}",
            patchset_id=patchset.patchset_id or "",
            round_idx=1,
            stage="smoke",
            status=status,
            decision=decision,
            eval_result={
                "stage": "smoke",
                "status": status,
                "errors": errors,
            },
            errors=errors,
            created_at=utc_now(),
        )
        record = experience_from_evaluation(patchset, evaluation)
        append_and_compress(store, record)
        records.append(record)

    static_evaluation = EvaluationRecord(
        eval_id="workflow_eval_scope",
        patchset_id=patchset.patchset_id or "",
        round_idx=1,
        stage="static_verify",
        status="static_rejected",
        decision="rejected_static_verify",
        eval_result={
            "stage": "static_verify",
            "status": "static_rejected",
            "errors": ["actual function escaped approved plan"],
        },
        errors=["actual function escaped approved plan"],
        created_at=utc_now(),
    )
    static_record = experience_from_evaluation(
        patchset,
        static_evaluation,
    )
    append_and_compress(store, static_record)
    records.append(static_record)

    pack = build_context_pack(store, round_idx=2)
    repair = patch_repair_memory_view(
        pack,
        json.dumps({
            "failure_kind": "patch_apply_failure",
            "violations": ["patch does not apply"],
        }),
    )
    outer = outer_algorithm_memory_view(pack)
    inner = inner_implementation_memory_view(pack)

    expected_ids = {record.experience_id for record in records}
    for record in records:
        targeted_repair = patch_repair_memory_view(
            pack,
            json.dumps({
                "failure_kind": record.outcome,
                "errors": record.negative_patterns,
            }),
        )
        assert record.experience_id in {
            item["experience_id"]
            for item in targeted_repair.repair_experience
        }
    assert records[1].experience_id in {
        item["experience_id"] for item in repair.repair_experience
    }
    assert {record.scope for record in records} == {"workflow"}
    assert {
        record.mechanism_projection["track"] for record in records
    } == {"workflow"}
    assert not expected_ids.intersection(
        item["experience_id"] for item in outer.algorithm_experience
    )
    assert not expected_ids.intersection(
        item["experience_id"]
        for item in inner.implementation_experience
    )
