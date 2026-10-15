from pathlib import Path
import json

import pytest

from seq_align.c_patch_loop import wisdom
from seq_align.c_patch_loop.models import (
    ActualDiffFootprint,
    AlgorithmPlanRecord,
    DiffHunkFootprint,
    DurableWisdomRecord,
    EvaluationAttemptRecord,
    EvaluationRecord,
    ExperienceRecord,
    InvocationRecord,
    InvocationSummaryRecord,
    PatchRecord,
    PatchsetRecord,
    PlanDiffBindingRecord,
    QueryCompletionRecord,
    RunMigrationRecord,
    SlotEventRecord,
)
from seq_align.c_patch_loop.store import LoopStore, hash_file, record_hash, utc_now


def _patchset(created_at: str) -> PatchsetRecord:
    patch = PatchRecord(
        component_name="null2_correction",
        c_target_file="generic_null2.c",
        c_target_functions=["p7_GNull2_ByExpectation"],
        unified_diff="--- a/src/generic_null2.c\n+++ b/src/generic_null2.c\n",
    )
    patch.patch_id = record_hash(patch, id_field="patch_id")
    return PatchsetRecord(
        round_idx=1,
        base_hmmer_source="/tmp/hmmer",
        patches=[patch],
        created_at=created_at,
    )


def test_actual_footprint_and_binding_are_versioned_structured_records():
    footprint = ActualDiffFootprint(
        extraction_status="ok",
        target_file="src/example.c",
        hunks=[DiffHunkFootprint(
            old_start=10,
            old_count=2,
            new_start=10,
            new_count=3,
            changed_old_lines=[11],
            changed_new_lines=[11, 12],
            pre_functions=["hot_path"],
            post_functions=["hot_path"],
        )],
        pre_functions=["hot_path"],
        post_functions=["hot_path"],
        actual_functions=["hot_path"],
    )
    binding = PlanDiffBindingRecord(
        binding_status="ok",
        final_component_name="example",
        final_allowed_file="src/example.c",
        final_allowed_functions=["hot_path"],
        actual_footprint=footprint,
        containment={
            "actual_within_final_proposal": True,
            "final_and_actual_within_inner_plan": True,
            "final_and_actual_within_outer_plan": True,
        },
    )

    data = binding.model_dump(mode="json")
    assert data["schema_version"] == "1.0"
    assert data["actual_footprint"]["schema_version"] == "1.0"
    assert data["actual_footprint"]["hunks"][0]["changed_new_lines"] == [11, 12]
    assert data["containment"]["actual_within_final_proposal"] is True


    a = _patchset("2026-01-01T00:00:00Z")
    b = _patchset("2026-01-02T00:00:00Z")
    assert record_hash(a, id_field="patchset_id") == record_hash(b, id_field="patchset_id")


def _stored_patchset(store: LoopStore, *, round_idx: int = 1) -> PatchsetRecord:
    record = _patchset("2026-01-01T00:00:00Z").model_copy(
        update={"round_idx": round_idx}
    )
    record.patchset_id = record_hash(record, id_field="patchset_id")
    store.put_patchset(record)
    return record


def _stored_evaluation(
    store: LoopStore,
    patchset: PatchsetRecord,
) -> EvaluationRecord:
    record = EvaluationRecord(
        patchset_id=patchset.patchset_id or "",
        round_idx=patchset.round_idx,
        stage="full",
        status="ok",
        decision="accepted_speedup",
        eval_result={"status": "ok", "tp": 2, "fp": 0, "cost_s": 1.0},
    )
    record.eval_id = record_hash(record, id_field="eval_id")
    store.put_evaluation(record)
    return record


def test_import_verified_evidence_is_immutable_idempotent_and_provenanced(
    tmp_path: Path,
):
    source = LoopStore(tmp_path / "source")
    target = LoopStore(tmp_path / "target")
    patchset = _stored_patchset(source, round_idx=23)
    evaluation = _stored_evaluation(source, patchset)
    paths = {
        f"patchsets/{patchset.patchset_id}.json": (
            source.patchsets_dir / f"{patchset.patchset_id}.json"
        ),
        f"evaluations/{evaluation.eval_id}.json": (
            source.evaluations_dir / f"{evaluation.eval_id}.json"
        ),
    }
    manifest = {name: hash_file(path) for name, path in paths.items()}

    migration = target.import_verified_evidence(
        source,
        source_run_id="frozen-run",
        target_run_id="new-run",
        migration_reason="Reconfirm historical candidate",
        ruleset_version="c-patch-loop.v2",
        patchset_ids=[patchset.patchset_id or ""],
        evaluation_ids=[evaluation.eval_id or ""],
        source_artifact_hashes=manifest,
        source_round_lineage={patchset.patchset_id or "": 23},
    )
    again = target.import_verified_evidence(
        source,
        source_run_id="frozen-run",
        target_run_id="new-run",
        migration_reason="Reconfirm historical candidate",
        ruleset_version="c-patch-loop.v2",
        patchset_ids=[patchset.patchset_id or ""],
        evaluation_ids=[evaluation.eval_id or ""],
        source_artifact_hashes=manifest,
        source_round_lineage={patchset.patchset_id or "": 23},
    )

    assert migration.migration_id == again.migration_id
    assert migration.status == "imported"
    assert migration.source_round_lineage == {patchset.patchset_id: 23}
    assert len(target.iter_migrations()) == 1
    assert target.get_patchset(patchset.patchset_id or "") == patchset
    assert target.get_evaluation(evaluation.eval_id or "") == evaluation
    assert {name: hash_file(path) for name, path in paths.items()} == manifest
    assert (
        target.patchsets_dir / f"{patchset.patchset_id}.json"
    ).read_bytes() == paths[f"patchsets/{patchset.patchset_id}.json"].read_bytes()
    assert (
        target.evaluations_dir / f"{evaluation.eval_id}.json"
    ).read_bytes() == paths[f"evaluations/{evaluation.eval_id}.json"].read_bytes()


def test_import_verified_evidence_preserves_legacy_json_bytes(tmp_path: Path):
    source = LoopStore(tmp_path / "source")
    target = LoopStore(tmp_path / "target")
    patchset = _stored_patchset(source)
    evaluation = _stored_evaluation(source, patchset)
    evaluation_path = source.evaluations_dir / f"{evaluation.eval_id}.json"
    raw = evaluation.model_dump(mode="json")
    for field in (
        "attempt_ids",
        "primary_attempt_id",
        "evaluation_fingerprint",
        "query_completion_ids",
        "requested_queries",
        "completed_queries",
        "resumed_queries",
        "interruption",
    ):
        raw.pop(field)
    raw["schema_version"] = "1.0"
    raw["eval_id"] = None
    legacy_id = record_hash(raw, id_field="eval_id")
    raw["eval_id"] = legacy_id
    evaluation_path.unlink()
    evaluation_path = source.evaluations_dir / f"{legacy_id}.json"
    evaluation_path.write_text(
        json.dumps(raw, ensure_ascii=False, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    patchset_path = source.patchsets_dir / f"{patchset.patchset_id}.json"
    manifest = {
        f"patchsets/{patchset.patchset_id}.json": hash_file(patchset_path),
        f"evaluations/{legacy_id}.json": hash_file(evaluation_path),
    }
    source_bytes = evaluation_path.read_bytes()

    target.import_verified_evidence(
        source,
        source_run_id="frozen-run",
        target_run_id="new-run",
        migration_reason="Preserve legacy evidence",
        ruleset_version="c-patch-loop.v2",
        patchset_ids=[patchset.patchset_id or ""],
        evaluation_ids=[legacy_id],
        source_artifact_hashes=manifest,
    )

    target_path = target.evaluations_dir / f"{legacy_id}.json"
    assert target_path.read_bytes() == source_bytes
    assert hash_file(target_path) == manifest[f"evaluations/{legacy_id}.json"]
    assert json.loads(target_path.read_text(encoding="utf-8"))["schema_version"] == "1.0"
    assert target.get_evaluation(legacy_id).schema_version == "1.0"


def test_import_verified_evidence_preflights_all_target_conflicts(tmp_path: Path):
    source = LoopStore(tmp_path / "source")
    target = LoopStore(tmp_path / "target")
    patchset = _stored_patchset(source)
    evaluation = _stored_evaluation(source, patchset)
    patchset_path = source.patchsets_dir / f"{patchset.patchset_id}.json"
    evaluation_path = source.evaluations_dir / f"{evaluation.eval_id}.json"
    manifest = {
        f"patchsets/{patchset.patchset_id}.json": hash_file(patchset_path),
        f"evaluations/{evaluation.eval_id}.json": hash_file(evaluation_path),
    }
    conflicting_target = target.evaluations_dir / f"{evaluation.eval_id}.json"
    conflicting_target.write_text("{}\n", encoding="utf-8")

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        target.import_verified_evidence(
            source,
            source_run_id="frozen-run",
            target_run_id="new-run",
            migration_reason="test",
            ruleset_version="c-patch-loop.v2",
            patchset_ids=[patchset.patchset_id or ""],
            evaluation_ids=[evaluation.eval_id or ""],
            source_artifact_hashes=manifest,
        )

    assert not (target.patchsets_dir / f"{patchset.patchset_id}.json").exists()
    assert target.iter_migrations() == []


def test_import_verified_evidence_rejects_unsupported_schema_without_writes(
    tmp_path: Path,
):
    source = LoopStore(tmp_path / "source")
    target = LoopStore(tmp_path / "target")
    patchset = _stored_patchset(source)
    path = source.patchsets_dir / f"{patchset.patchset_id}.json"
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["schema_version"] = "9.0"
    raw["patchset_id"] = None
    unsupported_id = record_hash(raw, id_field="patchset_id")
    raw["patchset_id"] = unsupported_id
    path.unlink()
    path = source.patchsets_dir / f"{unsupported_id}.json"
    path.write_text(json.dumps(raw, sort_keys=True) + "\n", encoding="utf-8")
    relpath = f"patchsets/{unsupported_id}.json"

    with pytest.raises(ValueError, match="unsupported patchset schema"):
        target.import_verified_evidence(
            source,
            source_run_id="frozen-run",
            target_run_id="new-run",
            migration_reason="test",
            ruleset_version="c-patch-loop.v2",
            patchset_ids=[unsupported_id],
            evaluation_ids=[],
            source_artifact_hashes={relpath: hash_file(path)},
        )

    assert target.iter_patchsets() == []
    assert target.iter_migrations() == []


def test_import_verified_evidence_rejects_hash_mismatch_without_writes(
    tmp_path: Path,
):
    source = LoopStore(tmp_path / "source")
    target = LoopStore(tmp_path / "target")
    patchset = _stored_patchset(source)
    relpath = f"patchsets/{patchset.patchset_id}.json"

    with pytest.raises(ValueError, match="hash mismatch"):
        target.import_verified_evidence(
            source,
            source_run_id="frozen-run",
            target_run_id="new-run",
            migration_reason="test",
            ruleset_version="c-patch-loop.v2",
            patchset_ids=[patchset.patchset_id or ""],
            evaluation_ids=[],
            source_artifact_hashes={relpath: "0" * 64},
        )

    assert target.iter_patchsets() == []
    assert target.iter_migrations() == []


def test_run_migration_hash_ignores_created_at():
    record = RunMigrationRecord(
        source_run_id="old",
        source_run_path="/old",
        target_run_id="new",
        target_run_path="/new",
        migration_reason="test",
        ruleset_version="v2",
        status="imported",
        created_at="2026-01-01T00:00:00Z",
    )
    changed = record.model_copy(update={"created_at": "2026-01-02T00:00:00Z"})
    assert record_hash(record, id_field="migration_id") == record_hash(
        changed,
        id_field="migration_id",
    )


def test_store_refuses_different_content_for_same_id(tmp_path: Path):
    store = LoopStore(tmp_path)
    a = _patchset(utc_now())
    a.patchset_id = "same-id"
    b = _patchset(utc_now())
    b.patchset_id = "same-id"
    b.patches[0].unified_diff += "@@ -1 +1 @@\n-a\n+b\n"
    store.put_patchset(a)
    with pytest.raises(FileExistsError):
        store.put_patchset(b)


def test_algorithm_plan_store_writes_content_addressed_record(tmp_path: Path):
    store = LoopStore(tmp_path)
    record = AlgorithmPlanRecord(
        round_idx=1,
        context_pack_id="ctx",
        comparison_patchset_id="stock",
        algorithm_family="pipeline_flow_strategy",
        baseline_algorithm_summary="Stock p7_Pipeline applies the full filter cascade before reporting hits.",
        proposed_algorithm_change="Use a candidate-pruning decision surface with exact fallback for borderline targets.",
        speed_mechanism="Reduce downstream search work for confidently rejected candidates while exact fallback preserves borderline scoring.",
        quality_preservation_argument="TP and FP are unchanged because reportable/includable candidates use the exact stock path.",
        target_contracts=["pipeline_flow"],
        target_functions=["p7_Pipeline"],
        profile_evidence_citation="No measured profile evidence is available in this context pack.",
        created_at="2026-01-01T00:00:00Z",
    )
    record.algorithm_plan_id = record_hash(record, id_field="algorithm_plan_id")

    store.put_algorithm_plan(record)
    loaded = store.iter_algorithm_plans()

    assert loaded[0]["algorithm_plan_id"] == record.algorithm_plan_id
    assert loaded[0]["algorithm_family"] == "pipeline_flow_strategy"
    assert record_hash(record.model_copy(update={"created_at": "2026-01-02T00:00:00Z"}), id_field="algorithm_plan_id") == record.algorithm_plan_id



def test_wisdom_record_hash_ignores_created_at():
    a = DurableWisdomRecord(
        confidence="confirmed",
        component_name="sequence_io",
        target_functions=["addbuf"],
        lesson="Cache addbuf hot loop loads.",
        created_at="2026-01-01T00:00:00Z",
    )
    b = a.model_copy(update={"created_at": "2026-01-02T00:00:00Z"})

    assert record_hash(a, id_field="wisdom_id") == record_hash(b, id_field="wisdom_id")


def test_wisdom_upsert_is_idempotent_and_rejects_conflict(tmp_path: Path):
    record = DurableWisdomRecord(
        wisdom_id="same-wisdom",
        confidence="candidate",
        component_name="dp_kernel",
        target_functions=["forward_engine"],
        lesson="Candidate lesson.",
        created_at=utc_now(),
    )
    returned = wisdom.upsert_durable_wisdom(record, tmp_path / "wisdom")
    again = wisdom.upsert_durable_wisdom(record.model_copy(update={"created_at": utc_now()}), tmp_path / "wisdom")
    conflict = record.model_copy(update={"lesson": "Different lesson."})

    assert returned.wisdom_id == "same-wisdom"
    assert again.wisdom_id == "same-wisdom"
    assert len(wisdom.load_durable_wisdom(tmp_path / "wisdom")) == 1
    with pytest.raises(FileExistsError):
        wisdom.upsert_durable_wisdom(conflict, tmp_path / "wisdom")


def test_experience_append_is_idempotent_and_rejects_conflict(tmp_path: Path):
    store = LoopStore(tmp_path)
    record = ExperienceRecord(
        experience_id="same-experience",
        scope="workflow",
        polarity="negative",
        failure_stage="proposal",
        failure_class="proposal_error",
        outcome="patch_proposal_error",
        lesson_kind="workflow",
        lesson="Proposal failed.",
        created_at="2026-01-01T00:00:00Z",
    )

    store.append_experience(record)
    store.append_experience(record.model_copy(update={"created_at": "2026-01-02T00:00:00Z"}))
    lines = (store.experience_dir / "experience.jsonl").read_text(encoding="utf-8").splitlines()

    assert len(lines) == 1
    conflict = record.model_copy(update={"lesson": "Different failure."})
    with pytest.raises(FileExistsError):
        store.append_experience(conflict)


def test_iter_experience_uses_canonical_idempotent_records(tmp_path: Path):
    store = LoopStore(tmp_path)
    path = store.experience_dir / "experience.jsonl"
    first = {
        "experience_id": "stable",
        "outcome": "validated_unevaluated",
        "lesson": "Validated only.",
        "created_at": "2026-01-01T00:00:00Z",
    }
    repeated = {**first, "created_at": "2026-01-02T00:00:00Z"}
    without_id = {"outcome": "neutral", "lesson": "Legacy evidence."}
    path.write_text(
        "\n".join([
            json.dumps(first),
            "{malformed",
            json.dumps(["not", "a", "record"]),
            json.dumps(repeated),
            json.dumps(without_id),
        ]) + "\n",
        encoding="utf-8",
    )

    assert store.iter_experience() == [first, without_id]


def test_iter_experience_rejects_conflicting_duplicate_id(tmp_path: Path):
    store = LoopStore(tmp_path)
    path = store.experience_dir / "experience.jsonl"
    path.write_text(
        "\n".join([
            json.dumps({"experience_id": "conflict", "outcome": "neutral", "lesson": "First."}),
            json.dumps({"experience_id": "conflict", "outcome": "neutral", "lesson": "Second."}),
        ]) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(FileExistsError, match="conflicting experience_id: conflict"):
        store.iter_experience()


def test_evaluation_checkpoint_records_are_content_addressed_and_strict(
    tmp_path: Path,
):
    store = LoopStore(tmp_path)
    attempt = EvaluationAttemptRecord(
        invocation_id="invocation-1",
        patchset_id="patchset-1",
        round_idx=3,
        stage="full",
        evaluation_fingerprint="eval-fingerprint",
        status="started",
        requested_queries=2,
        created_at="2026-01-01T00:00:00Z",
    )
    attempt.attempt_id = record_hash(attempt, id_field="attempt_id")
    completion = QueryCompletionRecord(
        attempt_id=attempt.attempt_id,
        invocation_id=attempt.invocation_id,
        evaluation_fingerprint=attempt.evaluation_fingerprint,
        query_fingerprint="query-fingerprint",
        patchset_id=attempt.patchset_id,
        stage="full",
        attempt_kind="primary",
        query_idx=1,
        query_id="query-1",
        query_path="queries/query_0001",
        query_sha256="query-sha",
        binary_sha256={"jackhmmer": "binary-sha"},
        input_sha256={"uniref": "dataset-sha"},
        status="ok",
        tp=4,
        fp=0,
        cost_s=1.5,
        artifacts={
            "hmmsearch_tbl": {
                "path": "hmmsearch.tbl",
                "bytes": 12,
                "sha256": "artifact-sha",
            }
        },
        created_at="2026-01-01T00:00:01Z",
    )
    completion.query_completion_id = record_hash(
        completion,
        id_field="query_completion_id",
    )

    store.put_evaluation_attempt(attempt)
    store.put_evaluation_attempt(
        attempt.model_copy(update={"created_at": "2026-01-02T00:00:00Z"})
    )
    store.put_query_completion(completion)

    assert store.get_evaluation_attempt(attempt.attempt_id) == attempt
    assert store.get_query_completion(
        completion.query_completion_id
    ) == completion
    assert len(store.iter_evaluation_attempts()) == 1
    assert len(store.iter_query_completions()) == 1
    assert store.iter_evaluations() == []

    path = store.query_completions_dir / (
        f"{completion.query_completion_id}.json"
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["tp"] = 99
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="content hash mismatch"):
        store.get_query_completion(completion.query_completion_id)


def test_slot_ledger_tracks_parented_heads_and_high_water(tmp_path: Path):
    store = LoopStore(tmp_path)
    invocation = InvocationRecord(
        invocation_id="invocation",
        configuration_id="config",
        configuration={"stage": "full"},
        requested_outer=2,
        inner_budget=3,
    )
    store.put_invocation(invocation)
    pending = SlotEventRecord(
        slot_id="config:1:1",
        invocation_id="invocation",
        configuration_id="config",
        outer_idx=1,
        inner_idx=1,
        state="pending",
    )
    pending.slot_event_id = record_hash(pending, id_field="slot_event_id")
    store.put_slot_event(pending)
    first_high_water = store.canonical_high_water()
    running = SlotEventRecord(
        slot_id=pending.slot_id,
        invocation_id="invocation",
        configuration_id="config",
        outer_idx=1,
        inner_idx=1,
        parent_slot_event_id=pending.slot_event_id,
        state="running",
    )
    running.slot_event_id = record_hash(running, id_field="slot_event_id")
    store.put_slot_event(running)

    assert store.slot_heads(invocation_id="invocation")[pending.slot_id] == running
    assert store.canonical_high_water() != first_high_water


def test_single_writer_excludes_second_store(tmp_path: Path):
    first = LoopStore(tmp_path)
    second = LoopStore(tmp_path)

    with first.single_writer():
        with pytest.raises(RuntimeError, match="another controller"):
            with second.single_writer():
                pass


def test_slot_heads_isolate_same_logical_slot_across_invocations(tmp_path: Path):
    store = LoopStore(tmp_path)
    for invocation_id in ("invocation-1", "invocation-2"):
        event = SlotEventRecord(
            slot_id="config:1:1",
            invocation_id=invocation_id,
            configuration_id="config",
            outer_idx=1,
            inner_idx=1,
            state="pending",
        )
        event.slot_event_id = record_hash(event, id_field="slot_event_id")
        store.put_slot_event(event)

    global_heads = store.slot_heads()

    assert len(global_heads) == 2
    assert {
        head.invocation_id for head in global_heads.values()
    } == {"invocation-1", "invocation-2"}
    assert list(store.slot_heads(invocation_id="invocation-1")) == [
        "config:1:1"
    ]


def test_invocation_summary_does_not_advance_canonical_high_water(
    tmp_path: Path,
):
    store = LoopStore(tmp_path)
    invocation = InvocationRecord(
        invocation_id="invocation",
        configuration_id="config",
        configuration={},
        requested_outer=1,
        inner_budget=1,
    )
    store.put_invocation(invocation)
    high_water = store.canonical_high_water()
    summary = InvocationSummaryRecord(
        invocation_id=invocation.invocation_id,
        configuration_id=invocation.configuration_id,
        requested_outer=1,
        inner_budget=1,
        terminal_state="complete",
        high_water_mark=high_water,
    )
    summary.invocation_summary_id = record_hash(
        summary,
        id_field="invocation_summary_id",
    )

    store.put_invocation_summary(summary)

    assert store.canonical_high_water() == high_water


def test_strict_invocation_loader_rejects_embedded_id_mismatch(
    tmp_path: Path,
):
    store = LoopStore(tmp_path)
    invocation = InvocationRecord(
        invocation_id="invocation",
        configuration_id="config",
        configuration={},
        requested_outer=1,
        inner_budget=1,
    )
    store.put_invocation(invocation)
    path = store.invocations_dir / "invocation.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["invocation_id"] = "different"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="embedded ID mismatch"):
        store.get_invocation("invocation")


def test_invocation_summary_is_content_addressed(tmp_path: Path):
    store = LoopStore(tmp_path)
    summary = InvocationSummaryRecord(
        invocation_id="invocation",
        configuration_id="config",
        requested_outer=1,
        inner_budget=2,
        blocked_slots=2,
        terminal_state="partial",
        high_water_mark=store.canonical_high_water(),
    )
    summary.invocation_summary_id = record_hash(
        summary,
        id_field="invocation_summary_id",
    )

    store.put_invocation_summary(summary)
    assert store.iter_invocation_summaries()[0]["blocked_slots"] == 2


def test_attempt_identity_distinguishes_independent_invocations():
    common = {
        "patchset_id": "patchset",
        "round_idx": 1,
        "stage": "full",
        "evaluation_fingerprint": "same-inputs",
        "status": "started",
    }
    first = EvaluationAttemptRecord(invocation_id="invocation-1", **common)
    second = EvaluationAttemptRecord(invocation_id="invocation-2", **common)

    assert record_hash(first, id_field="attempt_id") != record_hash(
        second,
        id_field="attempt_id",
    )


def test_strict_patchset_loader_rejects_embedded_id_mismatch(tmp_path: Path):
    store = LoopStore(tmp_path)
    patchset = _patchset("2026-01-01T00:00:00Z")
    patchset.patchset_id = record_hash(patchset, id_field="patchset_id")
    store.put_patchset(patchset)
    path = store.patchsets_dir / f"{patchset.patchset_id}.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["patchset_id"] = "different"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="embedded ID mismatch"):
        store.get_patchset(patchset.patchset_id)


def test_legacy_evaluation_record_loads_with_empty_checkpoint_lineage():
    record = EvaluationRecord.model_validate({
        "schema_version": "1.0",
        "eval_id": "legacy",
        "patchset_id": "patchset",
        "round_idx": 1,
        "stage": "full",
        "status": "ok",
    })

    assert record.attempt_ids == []
    assert record.primary_attempt_id is None
    assert record.query_completion_ids == []
    assert record.completed_queries is None


def test_legacy_experience_fields_remain_compatible():
    record = ExperienceRecord(outcome="rejected_not_faster", lesson="Legacy lesson.")

    assert record.scope == "inner_implementation"
    assert record.polarity == "neutral"
    assert record.lesson_kind == "algorithm"
