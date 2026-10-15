from pathlib import Path

from seq_align.c_patch_loop import wisdom
from seq_align.c_patch_loop.context import build_context_pack, context_prompt, plan_prompt, algorithm_plan_prompt
from seq_align.c_patch_loop.experience import (
    algorithm_mechanism_projection,
    implementation_mechanism_projection,
    mechanism_fingerprint,
)
from seq_align.c_patch_loop.models import AlgorithmPlanRecord, DurableWisdomRecord, EvalSummary, EvaluationRecord, PatchPlanRecord, PatchRecord, PatchsetRecord
from seq_align.c_patch_loop.store import LoopStore, record_hash, sha256_text, utc_now


def test_seed_wisdom_import_is_idempotent(tmp_path: Path):
    wisdom_dir = tmp_path / "wisdom"

    first = wisdom.import_seed_wisdom(wisdom_dir)
    second = wisdom.import_seed_wisdom(wisdom_dir)
    loaded = wisdom.load_durable_wisdom(wisdom_dir)

    assert len(first) == 1
    assert len(second) == 1
    assert len(loaded) == 1
    record = loaded[0]
    assert record.confidence == "confirmed"
    assert record.wisdom_kind == "implementation_wisdom"
    assert record.component_name == "sequence_io"
    assert record.target_functions == ["addbuf"]
    assert "cccb255b9c0d2ec126e3b913ecf22531de2270bff4ede48dda5c8bef5e420a0f" in record.source_eval_ids
    assert "87f40e4cf3282cf0fb4b88afbcf8118672dcad02035ae201ed6590ab985bfe67" in record.source_eval_ids
    assert record.source_patchset_ids == ["3f0141f583e29f8531c2ba20eb6bad291f587b5d1edd8afcab7f7f8cbd2f22c2"]
    assert record.metric_evidence["median_speedup_fraction"] == 0.22439045235190677
    assert record.metric_evidence["stock"]["tp"] == 14
    assert record.metric_evidence["stock"]["fp"] == 3
    assert record.metric_evidence["patch"]["tp"] == 14
    assert record.metric_evidence["patch"]["fp"] == 3


def test_context_pack_includes_hcc_layers_and_seeded_wisdom(tmp_path: Path):
    store = LoopStore(tmp_path / "run")
    wisdom_dir = tmp_path / "wisdom"
    wisdom.import_seed_wisdom(wisdom_dir)
    profile = {
        "schema_version": "hmmer_cli_eval_profile.v1",
        "dominant_command": "jackhmmer",
        "command_wall_s": {"jackhmmer": 20.0, "hmmsearch": 1.0},
        "command_wall_fraction": {"jackhmmer": 20.0 / 21.0, "hmmsearch": 1.0 / 21.0},
        "top_slowest_queries": [{"query_id": "q1", "cost_s": 21.0, "dominant_command": "jackhmmer"}],
        "external_profiler": {"tool": "gprof", "status": "ok", "excluded_from_cost_s": True},
        "symbol_hotspots": [
            {
                "rank": 1,
                "command": "jackhmmer",
                "symbol": "addbuf",
                "dso": "gprof",
                "approx_command_wall_s": 8.0,
                "max_self_pct": 40.0,
                "profile_source": "gprof_flat_profile",
            }
        ],
    }
    ev = EvaluationRecord(
        eval_id="eval_addbuf_profile",
        patchset_id="stock_patchset",
        round_idx=0,
        stage="full",
        status="ok",
        decision="accepted_baseline",
        eval_result={"stage": "full", "status": "ok", "tp": 14, "fp": 3, "tp_fp_ratio": 14 / 17, "cost_s": 21.0, "meta": {"profile": profile}},
        created_at=utc_now(),
    )
    store.put_evaluation(ev)
    summary = EvalSummary(eval_id="eval_addbuf_profile", patchset_id="stock_patchset", stage="full", status="ok", tp=14, fp=3, cost_s=21.0)

    pack = build_context_pack(
        store,
        round_idx=1,
        parent_patchset_id=summary.patchset_id,
        current_best_patchset_id=summary.patchset_id,
        baseline=summary,
        current_best=summary,
        wisdom_dir=wisdom_dir,
    )
    first_prompt = plan_prompt(pack)
    patch_prompt = context_prompt(pack)

    assert pack.l1_raw_working_memory
    assert isinstance(pack.l2_phase_summaries, list)
    assert pack.l3_durable_wisdom
    assert pack.l3_durable_wisdom[0].component_name == "sequence_io"
    assert pack.l3_durable_wisdom[0].target_functions == ["addbuf"]
    assert "explicit parent_version_id" in pack.application_base_policy
    assert "materializes that immutable parent" in pack.application_base_policy
    assert "PATCH APPLICATION POLICY" in first_prompt
    assert "INNER_IMPLEMENTATION_MEMORY_VIEW_JSON" in first_prompt
    assert "INNER_IMPLEMENTATION_PAYLOAD_JSON" in patch_prompt
    assert "CONTEXT_PACK_JSON" not in first_prompt
    assert "COMPRESSED EXPERIENCE SUMMARY" not in first_prompt
    assert "explicit-parent HMMER source excerpts" in patch_prompt


def test_outer_algorithm_prompt_filters_addbuf_implementation_wisdom_but_allows_streaming_family(tmp_path: Path):
    store = LoopStore(tmp_path / "run")
    wisdom_dir = tmp_path / "wisdom"
    wisdom.import_seed_wisdom(wisdom_dir)
    pack = build_context_pack(store, round_idx=1, wisdom_dir=wisdom_dir)
    outer_prompt = algorithm_plan_prompt(pack)
    inner_prompt = plan_prompt(pack)

    assert pack.l3_durable_wisdom[0].wisdom_kind == "implementation_wisdom"
    assert pack.l3_durable_wisdom[0].component_name == "sequence_io"
    assert "Input parsing and data movement are valid only under sequence_streaming_strategy" in outer_prompt
    for forbidden in ("addbuf", "cache", "SIMD"):
        assert forbidden.lower() not in outer_prompt.lower()
    assert "INNER_IMPLEMENTATION_MEMORY_VIEW_JSON" in inner_prompt
    assert "addbuf" in inner_prompt


def test_wisdom_retrieval_ranks_profile_matched_records(tmp_path: Path):
    wisdom_dir = tmp_path / "wisdom"
    seed = wisdom.import_seed_wisdom(wisdom_dir)[0]
    unrelated = DurableWisdomRecord(
        source="test",
        confidence="confirmed",
        component_name="domain_def",
        target_functions=["p7_domaindef_ByPosteriorHeuristics"],
        target_symbols=["p7_domaindef_ByPosteriorHeuristics"],
        lesson="Unrelated domain definition lesson.",
        retrieval_tags=["domain_def"],
        created_at=utc_now(),
    )
    wisdom.upsert_durable_wisdom(unrelated, wisdom_dir)
    profile_evidence = {
        "contract_hotspots": [{"component_name": "sequence_io", "functions": ["addbuf"]}],
        "function_hotspots": [{"component_name": "sequence_io", "function": "addbuf", "symbols": ["addbuf"]}],
        "fine_grained_hotspots": [{"component_name": "sequence_io", "function": "addbuf", "symbol": "addbuf"}],
        "recommended_contracts": [{"component_name": "sequence_io", "target_functions": ["addbuf"], "priority": "measured-symbol-hotspot"}],
    }

    ranked = wisdom.retrieve_wisdom_for_context(profile_evidence, [], wisdom_dir=wisdom_dir)

    assert ranked[0].wisdom_id == seed.wisdom_id
    assert ranked[0].component_name == "sequence_io"


def test_promote_wisdom_from_run_requires_full_accepted_speedup(tmp_path: Path):
    store = LoopStore(tmp_path / "run")
    wisdom_dir = tmp_path / "wisdom"
    patch = PatchRecord(
        component_name="sequence_io",
        c_target_file="easel/esl_sqio_ascii.c",
        c_target_functions=["addbuf"],
        unified_diff="--- a/easel/esl_sqio_ascii.c\n+++ b/easel/esl_sqio_ascii.c\n@@ -1 +1 @@\n-old\n+new\n",
        diff_sha256=sha256_text("sequence_io-addbuf-diff"),
        rationale="Cache hot addbuf loads.",
    )
    patch.patch_id = record_hash(patch, id_field="patch_id")
    patchset = PatchsetRecord(
        round_idx=1,
        patch_plan_id="plan_full",
        base_hmmer_source="/stock/hmmer",
        source_hashes={"easel/esl_sqio_ascii.c": "stockhash"},
        patches=[patch],
        created_at=utc_now(),
    )
    patchset.patchset_id = record_hash(patchset, id_field="patchset_id")
    store.put_patchset(patchset)
    smoke_ev = EvaluationRecord(
        eval_id="eval_smoke_speedup",
        patchset_id=patchset.patchset_id or "",
        round_idx=1,
        stage="smoke",
        status="ok",
        decision="accepted_speedup",
        eval_result={"stage": "smoke", "status": "ok", "tp": 14, "fp": 3, "cost_s": 9.0},
        metric_delta={"tp_delta": 0, "fp_delta": 0, "speedup_fraction": 0.1},
        created_at=utc_now(),
    )
    full_ev = EvaluationRecord(
        eval_id="eval_full_speedup",
        patchset_id=patchset.patchset_id or "",
        round_idx=1,
        stage="full",
        status="ok",
        decision="accepted_speedup",
        eval_result={"stage": "full", "status": "ok", "tp": 14, "fp": 3, "cost_s": 8.0, "meta": {"timing": {"eval_repeats": 3}}},
        metric_delta={"tp_delta": 0, "fp_delta": 0, "speedup_fraction": 0.2},
        created_at=utc_now(),
    )
    store.put_evaluation(smoke_ev)
    store.put_evaluation(full_ev)

    promoted = wisdom.promote_wisdom_from_run(store, wisdom_dir=wisdom_dir, require_full=True)

    assert len(promoted) == 1
    assert promoted[0].confidence == "confirmed"
    assert promoted[0].source_eval_ids == ["eval_full_speedup"]
    assert promoted[0].source_patchset_ids == [patchset.patchset_id]
    assert promoted[0].source_patch_plan_ids == ["plan_full"]
    assert promoted[0].metric_evidence["metric_delta"]["speedup_fraction"] == 0.2


def test_promote_wisdom_from_run_creates_dual_track_for_canonical_algorithm_plan(tmp_path: Path):
    store = LoopStore(tmp_path / "run")
    wisdom_dir = tmp_path / "wisdom"
    algorithm_plan = AlgorithmPlanRecord(
        algorithm_plan_id="algorithm_plan_full",
        round_idx=1,
        algorithm_family="pipeline_flow_strategy",
        baseline_algorithm_summary="Run every filter stage.",
        proposed_algorithm_change="Short-circuit candidates before expensive DP.",
        speed_mechanism="Reduce expensive DP work after the early filter.",
        quality_preservation_argument="Only candidates proven irrelevant are skipped.",
        target_contracts=["pipeline"],
        target_functions=["p7_Pipeline"],
    )
    store.put_algorithm_plan(algorithm_plan)
    patch_plan = PatchPlanRecord(
        patch_plan_id="patch_plan_full",
        round_idx=1,
        algorithm_plan_id=algorithm_plan.algorithm_plan_id,
        component_name="pipeline",
        c_target_file="src/p7_pipeline.c",
        c_target_functions=["p7_Pipeline"],
        target_symbols=["p7_Pipeline"],
        algorithm_hypothesis="The approved early filter reduces downstream DP work.",
        intended_code_change="Implement the approved early filter in p7_Pipeline.",
        runtime_effective_argument="The rebuilt binary skips downstream DP for proven irrelevant candidates.",
        anti_degenerate_check="Borderline candidates use the exact stock path.",
    )
    store.put_patch_plan(patch_plan)
    patch = PatchRecord(
        component_name="pipeline",
        c_target_file="src/p7_pipeline.c",
        c_target_functions=["p7_Pipeline"],
        unified_diff="--- a/src/p7_pipeline.c\n+++ b/src/p7_pipeline.c\n@@ -1 +1 @@\n-old\n+new\n",
        diff_sha256=sha256_text("pipeline-diff"),
        rationale="Implement the approved early filter.",
    )
    patchset = PatchsetRecord(
        round_idx=1,
        algorithm_plan_id=algorithm_plan.algorithm_plan_id,
        patch_plan_id="patch_plan_full",
        base_hmmer_source="/stock/hmmer",
        source_hashes={"src/p7_pipeline.c": "stockhash"},
        patches=[patch],
    )
    patchset.patchset_id = record_hash(patchset, id_field="patchset_id")
    store.put_patchset(patchset)
    ev = EvaluationRecord(
        eval_id="eval_dual_full",
        patchset_id=patchset.patchset_id or "",
        round_idx=1,
        stage="full",
        status="ok",
        decision="accepted_speedup",
        eval_result={"tp": 14, "fp": 3, "cost_s": 8.0, "meta": {"timing": {"eval_repeats": 3}}},
        metric_delta={"tp_delta": 0, "fp_delta": 0, "speedup_fraction": 0.2},
    )
    store.put_evaluation(ev)

    promoted = wisdom.promote_wisdom_from_run(store, wisdom_dir=wisdom_dir)

    assert [record.wisdom_kind for record in promoted] == ["implementation_wisdom", "algorithm_wisdom"]
    implementation, algorithm = promoted
    assert implementation.source_patch_plan_ids == ["patch_plan_full"]
    assert implementation.diff_sha256 == patch.diff_sha256
    assert implementation.source_algorithm_plan_ids == []
    expected_implementation_projection = implementation_mechanism_projection(patch_plan)
    assert implementation.mechanism_projection == expected_implementation_projection
    assert implementation.mechanism_fingerprint == mechanism_fingerprint(expected_implementation_projection)
    assert algorithm.algorithm_family == "pipeline_flow_strategy"
    assert algorithm.component_name == "pipeline"
    assert algorithm.mechanism == "Reduce expensive DP work after the early filter."
    assert algorithm.target_functions == ["p7_Pipeline"]
    assert algorithm.source_algorithm_plan_ids == ["algorithm_plan_full"]
    assert algorithm.source_patch_plan_ids == []
    assert algorithm.diff_sha256 is None
    expected_algorithm_projection = algorithm_mechanism_projection(algorithm_plan)
    assert algorithm.mechanism_projection == expected_algorithm_projection
    assert algorithm.mechanism_fingerprint == mechanism_fingerprint(expected_algorithm_projection)
    for field in ("source_eval_ids", "source_patchset_ids", "metric_evidence"):
        assert getattr(algorithm, field) == getattr(implementation, field)
    summaries = (wisdom_dir / "indexes" / "by_component.json").read_text(encoding="utf-8")
    assert '"source_algorithm_plan_ids"' in summaries
    assert '"source_patch_plan_ids"' in summaries


def test_legacy_wisdom_defaults_algorithm_plan_provenance(tmp_path: Path):
    wisdom_dir = tmp_path / "wisdom"
    wisdom_dir.mkdir()
    (wisdom_dir / "wisdom.jsonl").write_text(
        '{"wisdom_id":"legacy","lesson":"legacy implementation","source_patch_plan_ids":["patch-plan"]}\n',
        encoding="utf-8",
    )

    loaded = wisdom.load_durable_wisdom(wisdom_dir)

    assert loaded[0].wisdom_kind == "implementation_wisdom"
    assert loaded[0].source_algorithm_plan_ids == []
    assert loaded[0].source_patch_plan_ids == ["patch-plan"]


def test_promote_single_full_speedup_as_candidate_until_confirmed(tmp_path: Path):
    store = LoopStore(tmp_path / "run")
    wisdom_dir = tmp_path / "wisdom"
    patch = PatchRecord(
        component_name="sequence_io",
        c_target_file="easel/esl_sqio_ascii.c",
        c_target_functions=["seebuf"],
        unified_diff="--- a/easel/esl_sqio_ascii.c\n+++ b/easel/esl_sqio_ascii.c\n@@ -1 +1 @@\n-old\n+new\n",
        diff_sha256=sha256_text("sequence_io-seebuf-diff"),
        rationale="Cache seebuf hot-loop loads.",
    )
    patch.patch_id = record_hash(patch, id_field="patch_id")
    patchset = PatchsetRecord(
        round_idx=1,
        patch_plan_id="plan_single_full",
        base_hmmer_source="/stock/hmmer",
        source_hashes={"easel/esl_sqio_ascii.c": "stockhash"},
        patches=[patch],
        created_at=utc_now(),
    )
    patchset.patchset_id = record_hash(patchset, id_field="patchset_id")
    store.put_patchset(patchset)
    ev = EvaluationRecord(
        eval_id="eval_single_full_speedup",
        patchset_id=patchset.patchset_id or "",
        round_idx=1,
        stage="full",
        status="ok",
        decision="accepted_speedup",
        eval_result={"stage": "full", "status": "ok", "tp": 14, "fp": 3, "cost_s": 8.0, "meta": {"timing": {"eval_repeats": 1}}},
        metric_delta={"tp_delta": 0, "fp_delta": 0, "speedup_fraction": 0.05},
        created_at=utc_now(),
    )
    store.put_evaluation(ev)

    promoted = wisdom.promote_wisdom_from_run(store, wisdom_dir=wisdom_dir, require_full=True)

    assert len(promoted) == 1
    assert promoted[0].confidence == "candidate"
    assert promoted[0].metric_evidence["promotion_confirmation"]["reason"] == "single_full_eval_only"
    assert promoted[0].metric_evidence["promotion_confirmation"]["eval_repeats"] == 1



def test_context_pack_binds_inner_prompts_to_explicit_accepted_parent(tmp_path: Path):
    store = LoopStore(tmp_path / "run")
    accepted_parent = "accepted-lineage-patchset"
    pack = build_context_pack(
        store,
        round_idx=2,
        parent_patchset_id=accepted_parent,
        current_best_patchset_id=accepted_parent,
    )

    inner_prompt = plan_prompt(pack)
    patch_prompt = context_prompt(pack)

    assert pack.parent_patchset_id == accepted_parent
    assert pack.current_best_patchset_id == accepted_parent
    assert f'"parent_version_id": "{accepted_parent}"' in inner_prompt
    assert "explicit parent_version_id" in inner_prompt
    assert "explicit-parent HMMER source excerpts" in patch_prompt
