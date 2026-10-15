from __future__ import annotations

from pathlib import Path

import pytest

from seq_align.assembler import AssembledConfig
from seq_align.c_patch_loop.models import EvalSummary, EvaluationRecord, PatchsetRecord
from seq_align.c_patch_loop.runner import PatchRunner
from seq_align.c_patch_loop.store import LoopStore
from seq_align.eval_schema import EvalResult
from seq_align.slurm_eval import _full_query_timeout_policy


def _queries(tmp_path: Path) -> Path:
    data_dir = tmp_path / "queries"
    data_dir.mkdir()
    for index, query_id in enumerate(("q0", "q1")):
        (data_dir / f"all_samples.part_{index}.fasta").write_text(
            f">{query_id}\nMKT\n", encoding="utf-8"
        )
    return data_dir


def _eval_result(costs: tuple[float, float]) -> EvalResult:
    return EvalResult(
        stage="full",
        status="ok",
        tp=2,
        fp=0,
        cost_s=sum(costs),
        n_seqs=2,
        meta={
            "query_results": [
                {
                    "query_index": index,
                    "query_id": f"q{index}",
                    "status": "ok",
                    "cost_s": cost,
                    "tp": 1,
                    "fp": 0,
                    "tp_fp_ratio": 1.0,
                }
                for index, cost in enumerate(costs)
            ]
        },
    )


def _write_evaluation_record(path: Path, costs: tuple[float, float]) -> EvaluationRecord:
    result = _eval_result(costs)
    record = EvaluationRecord(
        eval_id="outer-parent-eval",
        patchset_id="outer-parent-patchset",
        round_idx=4,
        stage="full",
        status="ok",
        decision="accepted_full",
        eval_result=result.model_dump(mode="json"),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(record.model_dump_json(indent=2), encoding="utf-8")
    return record


def test_canonical_evaluation_record_is_a_query_timeout_reference(tmp_path: Path) -> None:
    data_dir = _queries(tmp_path)
    reference_path = tmp_path / "c_patch_loop" / "evaluations" / "outer.json"
    _write_evaluation_record(reference_path, (10.0, 20.0))

    policy = _full_query_timeout_policy(
        stage="full",
        run_dir=tmp_path,
        round_idx=5,
        data_dir=data_dir,
        reference_result_path=reference_path,
    )

    assert policy is not None
    assert policy["reference_result_path"] == str(reference_path)
    assert [row["reference_cost_s"] for row in policy["queries"]] == [10.0, 20.0]
    assert [row["time_limit_s"] for row in policy["queries"]] == [12.0, 24.0]


def test_explicit_outer_parent_overrides_mutating_accepted_eval_for_every_candidate(
    tmp_path: Path,
) -> None:
    data_dir = _queries(tmp_path)
    parent_path = tmp_path / "c_patch_loop" / "evaluations" / "outer-parent.json"
    _write_evaluation_record(parent_path, (10.0, 20.0))
    accepted_path = tmp_path / "accepted_eval.json"

    observed_limits = []
    for round_idx, accepted_costs in ((5, (100.0, 200.0)), (6, (1.0, 2.0))):
        accepted_path.write_text(
            _eval_result(accepted_costs).model_dump_json(indent=2), encoding="utf-8"
        )
        policy = _full_query_timeout_policy(
            stage="full",
            run_dir=tmp_path,
            round_idx=round_idx,
            data_dir=data_dir,
            reference_result_path=parent_path,
        )
        assert policy is not None
        observed_limits.append([row["time_limit_s"] for row in policy["queries"]])

    assert observed_limits == [[12.0, 24.0], [12.0, 24.0]]
    fallback = _full_query_timeout_policy(
        stage="full", run_dir=tmp_path, round_idx=7, data_dir=data_dir
    )
    assert fallback is not None
    assert [row["time_limit_s"] for row in fallback["queries"]] == [1.2, 2.4]


def test_explicit_outer_parent_reference_must_be_complete(tmp_path: Path) -> None:
    data_dir = _queries(tmp_path)
    invalid_path = tmp_path / "c_patch_loop" / "evaluations" / "invalid.json"
    invalid_path.parent.mkdir(parents=True)
    invalid_path.write_text(
        EvaluationRecord(
            eval_id="invalid",
            patchset_id="parent",
            round_idx=4,
            stage="full",
            status="ok",
            eval_result=None,
        ).model_dump_json(indent=2),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="explicit query timeout reference"):
        _full_query_timeout_policy(
            stage="full",
            run_dir=tmp_path,
            round_idx=5,
            data_dir=data_dir,
            reference_result_path=invalid_path,
        )


def test_patch_runner_evaluate_forwards_explicit_reference_to_slurm(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_dir = _queries(tmp_path)
    astral = tmp_path / "astral.fa"
    uniref = tmp_path / "uniref.fa"
    scop = tmp_path / "scop.txt"
    astral.write_text(">q0\nMKT\n", encoding="utf-8")
    uniref.write_text(">u0\nMKT\n", encoding="utf-8")
    scop.write_text("q0\ta.1.1.1\n", encoding="utf-8")
    store = LoopStore(tmp_path / "run")
    runner = PatchRunner(
        store,
        data_dir=data_dir,
        astral_db=astral,
        uniref_db=uniref,
        scop_txt=scop,
        eval_backend="slurm",
    )
    patchset = PatchsetRecord(
        patchset_id="candidate-patchset",
        round_idx=5,
        base_hmmer_source=str(runner.hmmer_src_root),
        patches=[],
    )
    store.put_patchset(patchset)
    reference_path = tmp_path / "outer-parent.json"
    _write_evaluation_record(reference_path, (10.0, 20.0))
    calls: list[dict] = []

    def fake_submit_and_wait(**kwargs):
        calls.append(kwargs)
        return _eval_result((8.0, 16.0))

    monkeypatch.setattr("seq_align.slurm_eval.submit_and_wait", fake_submit_and_wait)
    current_best = EvalSummary(
        eval_id="outer-parent-eval",
        patchset_id="outer-parent-patchset",
        stage="full",
        status="ok",
        tp=2,
        fp=0,
        cost_s=30.0,
    )

    record = runner.evaluate(
        patchset,
        AssembledConfig(),
        stage="full",
        current_best=current_best,
        quality_reference=current_best,
        query_timeout_reference=reference_path,
        commit_indexes=False,
        write_legacy_result=False,
    )

    assert record.status == "ok"
    assert len(calls) == 1
    assert calls[0]["round_idx"] == 5
    assert calls[0]["query_timeout_reference"] == str(reference_path.resolve())
