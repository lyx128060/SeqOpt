"""Tests for chunked Slurm submission.

A run of N queries is submitted as ``ceil(N / queries_per_job)`` independent
Slurm arrays instead of one N-element array. The property that keeps this cheap
to implement is that every worker still writes ``result_<global index>.json``: a
chunk exports its offset and the worker adds its local array id, so the
aggregation path stays chunk-unaware. These tests pin exactly that.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import pytest

from seq_align import slurm_eval
from seq_align.eval_schema import EvalResult


def _fake_sbatch(record: list[list[str]], base_id: int = 1000):
    """Stand in for subprocess.run: capture sbatch argv, never touch a cluster."""

    class _Result:
        def __init__(self, code: int, out: str = "", err: str = "") -> None:
            self.returncode = code
            self.stdout = out
            self.stderr = err

    def run(command, **_kwargs):
        if isinstance(command, list) and command and command[0] == "sbatch":
            record.append(list(command))
            return _Result(0, f"{base_id + len(record)};testcluster\n")
        return _Result(0)

    return run


def test_fifty_queries_become_ten_jobs_of_five(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """The whole point: 50 queries -> 10 job IDs, 5 sequences each."""
    monkeypatch.delenv("SEQ_ALIGN_SLURM_FULL_QUERIES_PER_JOB", raising=False)
    monkeypatch.setenv("SEQ_ALIGN_SLURM_FULL_MAX_PARALLEL", "10")
    calls: list[list[str]] = []
    monkeypatch.setattr(slurm_eval.subprocess, "run", _fake_sbatch(calls))
    # The worker command is not what is under test, and it legitimately requires a
    # cluster-side interpreter that this checkout does not have.
    monkeypatch.setattr(slurm_eval, "_host_worker_command", lambda *a, **k: "WORKER")

    waited: dict = {}
    monkeypatch.setattr(
        slurm_eval, "_wait_for_job", lambda **kw: waited.update(kw) or "WAITED"
    )

    request = {
        "stage": "full",
        "runtime": "host",
        "query_paths": [f"query_{index:03d}.fasta" for index in range(50)],
    }
    outcome = slurm_eval._submit_query_jobs(
        stage="full",
        request_path=tmp_path / "request.json",
        request=request,
        job_dir=tmp_path,
        resources=slurm_eval.resources_for("full"),
        submitted_epoch=1.0,
        build_submission=None,
    )
    assert outcome == "WAITED"

    # 10 submissions, each a 5-element array.
    assert len(calls) == 10
    assert [call[call.index("--array") + 1] for call in calls] == ["0-4%10"] * 10

    # Each carries its own global offset, so the workers cover 0..49 exactly once.
    offsets = [
        int(call[call.index("--export") + 1].split("SEQALIGN_QUERY_OFFSET=")[1])
        for call in calls
    ]
    assert offsets == [0, 5, 10, 15, 20, 25, 30, 35, 40, 45]

    # Waiting covers every chunk, and still asks for all 50 results.
    assert waited["expected"] == 50
    assert waited["job_ids"] == [str(1000 + index) for index in range(1, 11)]

    submission = json.loads((tmp_path / "submission.json").read_text(encoding="utf-8"))
    assert submission["query_count"] == 50
    assert submission["queries_per_job"] == 5
    assert len(submission["job_ids"]) == 10
    assert submission["chunk_offsets"] == offsets
    # `job_id` is retained so a pre-chunking run still resumes and single-job
    # callers keep working.
    assert submission["job_id"] == submission["job_ids"][0]
    assert submission["build_job_id"] is None


def test_settled_shape_is_pinned_at_the_dispatched_argv_level(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """The settled shape, read back from the dispatched sbatch argv.

    * ``--cpus-per-task 45`` -- the job allocation,
    * ``--array 0-4%5`` -- 5 elements, at most 5 running,
    * 10 submissions for 50 queries,
    * ``query_cpus_for("full") == 8`` -- the literal ``--cpu`` per query.
    """
    monkeypatch.setenv("SEQ_ALIGN_SLURM_FULL_CPUS", "45")
    monkeypatch.setenv("SEQ_ALIGN_SLURM_FULL_QUERY_CPUS", "8")
    monkeypatch.setenv("SEQ_ALIGN_SLURM_FULL_QUERIES_PER_JOB", "5")
    monkeypatch.setenv("SEQ_ALIGN_SLURM_FULL_MAX_PARALLEL", "5")

    calls: list[list[str]] = []
    monkeypatch.setattr(slurm_eval.subprocess, "run", _fake_sbatch(calls))
    monkeypatch.setattr(slurm_eval, "_host_worker_command", lambda *a, **k: "WORKER")
    monkeypatch.setattr(slurm_eval, "_wait_for_job", lambda **kw: "WAITED")

    request = {
        "stage": "full",
        "runtime": "host",
        "query_paths": [f"query_{index:03d}.fasta" for index in range(50)],
    }
    slurm_eval._submit_query_jobs(
        stage="full",
        request_path=tmp_path / "request.json",
        request=request,
        job_dir=tmp_path,
        resources=slurm_eval.resources_for("full"),
        submitted_epoch=1.0,
        build_submission=None,
    )

    assert len(calls) == 10, "50 queries in 5-query jobs is 10 submissions"
    for call in calls:
        assert call[call.index("--cpus-per-task") + 1] == "45", call
        assert call[call.index("--array") + 1] == "0-4%5", call
        assert call[call.index("--partition") + 1] == "compute", call

    submission = json.loads((tmp_path / "submission.json").read_text(encoding="utf-8"))
    assert submission["queries_per_job"] == 5
    assert submission["resources"]["cpus"] == 45
    assert submission["max_parallel"] == 5
    assert submission["query_count"] == 50
    assert len(submission["job_ids"]) == 10
    # The other knob: cores ONE query may use, i.e. the literal HMMER --cpu.
    assert slurm_eval.query_cpus_for("full") == 8
    assert slurm_eval.min_job_cpus_for("full") == 45  # 5 * (8 + 1)


def test_slurm_aggregate_carries_the_workers_stage_timings(tmp_path: Path):
    """An instrumented Slurm pass must be readable back as stage shares.

    Each worker profiles only its own query, and the session profile is normally
    assembled inside ``run_all_queries`` -- which on Slurm runs inside the worker, for
    a single query. Without the controller-side assembly the record carries no profile
    at all, ``run_posthoc_profile`` reports "no stage times", and the stage-share
    comparison cannot be made on this backend.
    """
    from seq_align import slurm_eval

    def worker(index: int, ssv_s: float, dp_s: float):
        result = EvalResult(stage="full", status="ok", tp=1, fp=0, cost_s=ssv_s + dp_s)
        result.meta = {
            "query_index": index,
            "profile": {
                "schema_version": "hmmer_cli_eval_profile.v1",
                "stage": "full",
                "status": "ok",
                "cpus": 8,
                "max_iter": 1,
                "query_id": f"q{index}",
                "cost_s": ssv_s + dp_s,
                "stage_timing": {
                    "time_source": "instrumented_build",
                    "stage_time_available": True,
                    # The worker reports its own aggregate: per-stage seconds flat.
                    # (The nested query-level shape uses contract_stages entries.)
                    "stage_time_s": {"ssv_filter": ssv_s, "dp_kernel": dp_s},
                    "jackhmmer_summary": {"elapsed_total_s": ssv_s + dp_s},
                },
            },
        }
        return result

    aggregate = slurm_eval._aggregate_results(
        "full",
        [worker(0, 8.0, 2.0), worker(1, 6.0, 4.0)],
        job_id="1",
        resources=slurm_eval.resources_for("full"),
        max_parallel=5,
        job_dir=tmp_path,
        batch_wall_s=1.0,
        query_cpus=8,
    )

    timing = aggregate.meta["profile"]["stage_timing"]
    assert timing["stage_time_available"] is True
    assert timing["time_source"] == "instrumented_build"
    assert timing["stage_time_fraction"] == {"dp_kernel": 0.3, "ssv_filter": 0.7}


def test_chunk_size_is_configurable_and_ragged_tail_is_handled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """7 queries at 3 per job -> arrays of 3, 3, 1 (not a 0-element tail)."""
    monkeypatch.setenv("SEQ_ALIGN_SLURM_FULL_QUERIES_PER_JOB", "3")
    monkeypatch.setenv("SEQ_ALIGN_SLURM_FULL_MAX_PARALLEL", "2")
    calls: list[list[str]] = []
    monkeypatch.setattr(slurm_eval.subprocess, "run", _fake_sbatch(calls))
    monkeypatch.setattr(slurm_eval, "_host_worker_command", lambda *a, **k: "WORKER")
    monkeypatch.setattr(slurm_eval, "_wait_for_job", lambda **kw: "WAITED")

    request = {"stage": "full", "runtime": "host", "query_paths": [f"q{i}" for i in range(7)]}
    slurm_eval._submit_query_jobs(
        stage="full",
        request_path=tmp_path / "request.json",
        request=request,
        job_dir=tmp_path,
        resources=slurm_eval.resources_for("full"),
        submitted_epoch=1.0,
        build_submission=None,
    )
    assert [call[call.index("--array") + 1] for call in calls] == ["0-2%2", "0-2%2", "0-0%2"]
    offsets = [
        int(call[call.index("--export") + 1].split("SEQALIGN_QUERY_OFFSET=")[1])
        for call in calls
    ]
    assert offsets == [0, 3, 6]


def test_failed_chunk_is_recorded_so_a_retry_can_fill_the_gap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """A mid-loop sbatch failure must not lose the jobs already in the queue."""
    monkeypatch.delenv("SEQ_ALIGN_SLURM_FULL_QUERIES_PER_JOB", raising=False)
    calls: list[list[str]] = []
    good = _fake_sbatch(calls)

    class _Result:
        returncode = 1
        stdout = ""
        stderr = "sbatch: error: invalid partition"

    def run(command, **kwargs):
        # Fail on the *third* submission, so exactly two chunks got into the queue.
        if isinstance(command, list) and command and command[0] == "sbatch" and len(calls) == 2:
            calls.append(list(command))
            return _Result()
        return good(command, **kwargs)

    monkeypatch.setattr(slurm_eval.subprocess, "run", run)
    monkeypatch.setattr(slurm_eval, "_host_worker_command", lambda *a, **k: "WORKER")
    monkeypatch.setattr(slurm_eval, "_wait_for_job", lambda **kw: "SHOULD NOT WAIT")

    request = {"stage": "full", "runtime": "host", "query_paths": [f"q{i}" for i in range(50)]}
    outcome = slurm_eval._submit_query_jobs(
        stage="full",
        request_path=tmp_path / "request.json",
        request=request,
        job_dir=tmp_path,
        resources=slurm_eval.resources_for("full"),
        submitted_epoch=1.0,
        build_submission=None,
    )
    # A submission failure yields run_error via `_error`, matching the pre-chunking
    # path; a *build* failure is the one that reports compile_error.
    assert outcome.status == "run_error"
    assert "invalid partition" in " ".join(outcome.errors)

    # 3 job ids survived: the partial record is what a retry resumes from.
    submission = json.loads((tmp_path / "submission.json").read_text(encoding="utf-8"))
    assert len(submission["job_ids"]) == 2
    assert submission["chunk_offsets"] == [0, 5]


@pytest.mark.parametrize(
    ("offset", "array_id", "explicit", "expected"),
    [
        ("10", "3", None, 13),      # chunked: global = offset + local array id
        (None, "7", None, 7),       # legacy single array still works
        ("40", "4", None, 44),      # last chunk of a 50-query run
        (None, None, "9", 9),       # explicit override wins
    ],
)
def test_worker_resolves_the_global_query_index(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    offset: str | None,
    array_id: str | None,
    explicit: str | None,
    expected: int,
):
    """The worker must map (chunk offset, local array id) onto one global query."""
    seen: dict = {}
    monkeypatch.setattr(
        slurm_eval,
        "run_request",
        lambda request_path, query_index: (
            seen.update(index=query_index) or EvalResult(stage="full", status="ok")
        ),
    )
    request_path = tmp_path / "request.json"
    request_path.write_text(
        json.dumps({"stage": "full", "result_dir": str(tmp_path / "results")}),
        encoding="utf-8",
    )

    for name, value in (
        ("SEQALIGN_QUERY_OFFSET", offset),
        ("SLURM_ARRAY_TASK_ID", array_id),
        ("SEQALIGN_QUERY_INDEX", explicit),
    ):
        if value is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, value)
    monkeypatch.setattr("sys.argv", ["slurm_eval", "--request", str(request_path)])

    assert slurm_eval.main() == 0
    assert seen["index"] == expected
    assert (tmp_path / "results" / f"result_{expected:03d}.json").is_file()


def test_patchless_baseline_still_compiles_once_and_points_all_jobs_at_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """主任务流程：先编译一次，然后所有测序 job 都指向本轮编译出的那一个产物。

    A patch-less candidate (the stock baseline) must take the same compile-once
    path. Before this change it fell through to ``hmmer_bundle_dir=None``, so each
    of the query workers compiled HMMER for itself -- 50 workers, 50 compiles, and
    no single "the code we actually measured".
    """
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    for index in range(2):
        (data_dir / f"all_samples.part_{index + 1:03d}.fasta").write_text(
            f">q{index}\nMKV\n", encoding="utf-8"
        )

    build_calls: list[dict] = []

    def fake_ensure_bundle(**kwargs):
        build_calls.append(kwargs)
        return {"status": "ok", "bundle_reused": False}, {"job_id": "build-42"}

    monkeypatch.setattr(slurm_eval, "_ensure_candidate_bundle", fake_ensure_bundle)

    # The real identity hashes the HMMER source tree, which this checkout
    # deliberately does not ship. What is under test is the *flow* -- one compile
    # for a patch-less submission, and every chunk pointed at that one artifact --
    # so a deterministic identity stands in for the hash.
    from seq_align import run_hmmer_cli

    monkeypatch.setattr(
        run_hmmer_cli,
        "binary_bundle_identity",
        lambda cfg, **kwargs: {
            "bundle_id": "b" * 64,
            "patch_set_sha256": "p" * 64,
            "build_context": kwargs.get("build_context", "host"),
        },
    )

    captured: dict = {}

    def fake_submit_jobs(**kwargs):
        captured.update(kwargs)
        return EvalResult(stage="full", status="ok")

    monkeypatch.setattr(slurm_eval, "_submit_query_jobs", fake_submit_jobs)

    # Pin the production core shape explicitly instead of inheriting whatever the
    # ambient environment happens to set. submit_and_wait now refuses to dispatch a
    # job allocation that cannot fit queries_per_job queries at query_cpus each.
    monkeypatch.setenv("SEQ_ALIGN_SLURM_FULL_CPUS", str(slurm_eval.min_job_cpus_for("full")))
    monkeypatch.delenv("SEQ_ALIGN_SLURM_FULL_QUERY_CPUS", raising=False)

    outcome = slurm_eval.submit_and_wait(
        stage="full",
        mods=[],  # <- no patch at all: this is the stock baseline
        run_dir=tmp_path / "run",
        round_idx=0,
        data_dir=data_dir,
        max_iter=1,
        astral_db=tmp_path / "astral.fa",
        uniref_db=tmp_path / "uniref.fa",
        scop_txt=tmp_path / "scop.txt",
    )
    assert outcome.status == "ok", outcome.errors

    # Exactly one compile happened, for a submission that carried no patches.
    assert len(build_calls) == 1

    request = captured["request"]
    assert request["hmmer_bundle_id"], "worker request has no bundle identity"
    assert request["hmmer_bundle_dir"], "workers were not pointed at the compiled artifact"
    # One identity for the whole submission, so every chunk resolves to the same
    # artifact, and each worker re-verifies it against the bundle manifest.
    assert request["hmmer_bundle_identity"]["bundle_id"] == request["hmmer_bundle_id"]
    assert str(request["hmmer_bundle_dir"]).endswith(str(request["hmmer_bundle_id"]))


def test_build_request_accepts_a_patchless_baseline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """The build worker must accept mods=[] now that every eval compiles once."""
    from seq_align import run_hmmer_cli

    seen: dict = {}
    monkeypatch.setattr(
        run_hmmer_cli,
        "build_binary_bundle",
        lambda cfg, bundle_dir, **kwargs: (
            seen.update(bundle_dir=str(bundle_dir), **kwargs) or {"bundle_id": "abc"}
        ),
    )
    request_path = tmp_path / "request.json"
    request_path.write_text(
        json.dumps(
            {
                "mods": [],
                "hmmer_bundle_dir": str(tmp_path / "bundles" / "abc"),
                "hmmer_bundle_id": "abc",
                "hmmer_build_context": "host-cluster",
                "build_resources": {"cpus": 2},
            }
        ),
        encoding="utf-8",
    )
    result = slurm_eval.run_build_request(request_path)
    assert result["status"] == "ok", result.get("errors")
    assert seen["bundle_dir"] == str(tmp_path / "bundles" / "abc")
    assert seen["expected_bundle_id"] == "abc"


def test_resume_accepts_both_chunked_and_legacy_submission_shapes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """An interrupted run must resume whether it was submitted chunked or not."""
    round_path = tmp_path / "round_000"
    job_dir = round_path / "slurm" / "full_123"
    results_dir = job_dir / "results"
    results_dir.mkdir(parents=True)
    request = {
        "stage": "full",
        "mods": [],
        "max_iter": 1,
        "source_paths": {"data_dir": "d", "astral_db": "a", "uniref_db": "u", "scop_txt": "s"},
        "query_timeout_policy": None,
        "optimization_objective": slurm_eval.objective_metadata(),
        "runtime": "host",
        "image_ref": None,
        "hmmer_bundle_id": None,
        "query_paths": ["q0", "q1"],
        # The core/concurrency shape this request was measured under. Resume now
        # requires it to match the live configuration, so a fixture that omits it
        # describes a measurement whose shape is unknown.
        "resources": asdict(slurm_eval.resources_for("full")),
        "query_cpus": slurm_eval.query_cpus_for("full"),
        "queries_per_job": slurm_eval.queries_per_job("full"),
    }
    (job_dir / "request.json").write_text(json.dumps(request), encoding="utf-8")
    for index in range(2):
        (results_dir / f"result_{index:03d}.json").write_text(
            EvalResult(stage="full", status="ok").model_dump_json(), encoding="utf-8"
        )

    for submission in (
        {"job_id": "55", "job_ids": ["55", "56"], "query_count": 2, "runtime": "host", "image_ref": None},
        {"job_id": "55", "query_count": 2, "runtime": "host", "image_ref": None},  # legacy shape
    ):
        (job_dir / "submission.json").write_text(json.dumps(submission), encoding="utf-8")
        found = slurm_eval._find_resumable_job(
            round_path=round_path,
            stage="full",
            mods=[],
            max_iter=1,
            source_paths=request["source_paths"],
            query_timeout_policy=None,
            optimization_objective=slurm_eval.objective_metadata(),
            bundle_id=None,
            runtime="host",
            image_ref=None,
        )
        assert found is not None, f"failed to resume submission shape: {sorted(submission)}"

    # The same job directory must NOT be re-attached when the recorded core shape
    # differs from the live one: aggregating leftover results from a different
    # allocation into a record that claims the new shape is exactly the silent
    # cross-configuration comparison this guard exists to prevent.
    for field, value in (
        ("query_cpus", slurm_eval.query_cpus_for("full") + 1),
        ("queries_per_job", slurm_eval.queries_per_job("full") + 1),
    ):
        mismatched = dict(request)
        mismatched[field] = value
        (job_dir / "request.json").write_text(json.dumps(mismatched), encoding="utf-8")
        (job_dir / "submission.json").write_text(
            json.dumps({"job_id": "55", "query_count": 2, "runtime": "host", "image_ref": None}),
            encoding="utf-8",
        )
        assert slurm_eval._find_resumable_job(
            round_path=round_path,
            stage="full",
            mods=[],
            max_iter=1,
            source_paths=request["source_paths"],
            query_timeout_policy=None,
            optimization_objective=slurm_eval.objective_metadata(),
            bundle_id=None,
            runtime="host",
            image_ref=None,
        ) is None, f"resumed across a changed {field}"
