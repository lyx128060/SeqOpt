from __future__ import annotations

import json
import shlex
import shutil
from dataclasses import asdict
from pathlib import Path

import pytest

from seq_align.assembler import AssembledConfig
from seq_align.components.base import ComponentKind, ComponentSpec
from seq_align.eval_schema import EvalResult
from seq_align import run_hmmer_cli, slurm_eval

IMAGE_A = "registry/image@sha256:" + "a" * 64
IMAGE_B = "registry/image@sha256:" + "b" * 64


def _patched_config() -> AssembledConfig:
    cfg = AssembledConfig()
    cfg.c_patches = [
        ComponentSpec(
            contract_name="test_kernel",
            kind=ComponentKind.C_PATCH,
            c_patch=(
                "--- a/src/kernel.c\n"
                "+++ b/src/kernel.c\n"
                "@@ -1 +1 @@\n"
                "-int value = 1;\n"
                "+int value = 2;\n"
            ),
            c_target_file="kernel.c",
            c_target_functions=["kernel"],
        )
    ]
    cfg.needs_recompile = True
    return cfg


def test_source_tree_provenance_tracks_architecture_specific_kernel_archives(
    tmp_path: Path,
) -> None:
    src = tmp_path / "src"
    (src / "impl_neon").mkdir(parents=True)
    (src / "impl_sse").mkdir()
    (src / "libhmmer.a").write_bytes(b"generic")
    (src / "impl_neon" / "libhmmerimpl.a").write_bytes(b"neon")
    (src / "impl_sse" / "libhmmerimpl.a").write_bytes(b"sse")

    provenance = run_hmmer_cli._source_tree_provenance(
        tmp_path,
        _patched_config(),
    )

    libraries = provenance["libraries"]
    assert libraries["libhmmer.a"]["exists"] is True
    assert libraries["impl_neon/libhmmerimpl.a"]["exists"] is True
    assert libraries["impl_sse/libhmmerimpl.a"]["exists"] is True
    assert libraries["impl_neon/libhmmerimpl.a"]["sha256"] != libraries["impl_sse/libhmmerimpl.a"]["sha256"]


def test_build_job_defaults_and_docker_command_use_node_local_staging(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in (
        "SEQ_ALIGN_SLURM_BUILD_CPUS",
        "SEQ_ALIGN_SLURM_BUILD_MEMORY",
        "SEQ_ALIGN_SLURM_BUILD_TIME",
    ):
        monkeypatch.delenv(name, raising=False)
    data_root = tmp_path / "benchmark-data"
    data_root.mkdir()
    monkeypatch.setenv("SEQ_ALIGN_SLURM_DOCKER_IMAGE", IMAGE_A)
    monkeypatch.setenv("SEQ_ALIGN_SLURM_DATA_ROOT", str(data_root))

    resources = slurm_eval.build_resources_for()
    command = slurm_eval._docker_worker_command(
        slurm_eval.PROJECT_ROOT / "request.json",
        image_ref=IMAGE_A,
        mode="build",
    )

    assert (resources.cpus, resources.memory, resources.walltime) == (4, "8G", "00:30:00")
    assert "SEQ_ALIGN_HMMER_BUILD_WORK_ROOT=/local/job/seqalign-build" in command
    assert "SLURM_ARRAY_TASK_ID" not in command
    assert "--mode build" in command


def test_default_shared_personal_mount_maps_to_workspace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = Path("/shared/alice/agent_workspace/EvoMaster-Jackhmmer")
    monkeypatch.setattr(slurm_eval, "PROJECT_ROOT", project)
    monkeypatch.delenv("SEQ_ALIGN_SLURM_HOST_PROJECT_ROOT", raising=False)
    monkeypatch.delenv("SEQ_ALIGN_SLURM_CONTAINER_PROJECT_ROOT", raising=False)

    mapped = slurm_eval._container_project_path(project / "runs" / "request.json")

    assert slurm_eval._host_project_mount_root() == Path("/shared/alice")
    assert mapped == Path("/workspace/agent_workspace/EvoMaster-Jackhmmer/runs/request.json")


def test_nested_shared_directory_does_not_trigger_cluster_layout_detection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = tmp_path / "shared" / "repo"
    project.mkdir(parents=True)
    monkeypatch.setattr(slurm_eval, "PROJECT_ROOT", project)
    monkeypatch.delenv("SEQ_ALIGN_SLURM_HOST_PROJECT_ROOT", raising=False)
    monkeypatch.delenv("SEQ_ALIGN_SLURM_CONTAINER_PROJECT_ROOT", raising=False)

    assert slurm_eval._host_project_mount_root() == project.resolve()
    assert slurm_eval._container_project_path(project / "request.json") == Path(
        "/workspace/request.json"
    )


def test_docker_worker_quotes_custom_mount_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    host_project_root = tmp_path / "owner root"
    project = host_project_root / "repo"
    request = project / "runs" / "request one.json"
    request.parent.mkdir(parents=True)
    data_root = tmp_path / "benchmark data"
    data_root.mkdir()
    monkeypatch.setattr(slurm_eval, "PROJECT_ROOT", project)
    monkeypatch.setenv("SEQ_ALIGN_SLURM_HOST_PROJECT_ROOT", str(host_project_root))
    monkeypatch.setenv("SEQ_ALIGN_SLURM_CONTAINER_PROJECT_ROOT", "/workspace mount")
    monkeypatch.setenv("SEQ_ALIGN_SLURM_DATA_ROOT", str(data_root))
    monkeypatch.setenv("SEQ_ALIGN_SLURM_CONTAINER_DATA_ROOT", "/benchmark data")
    monkeypatch.setenv("SEQ_ALIGN_SLURM_DOCKER_IMAGE", IMAGE_B)

    tokens = shlex.split(slurm_eval._docker_worker_command(request, image_ref=IMAGE_A))

    assert tokens[tokens.index("--mount") + 1] == f"{data_root}:/benchmark data:ro"
    assert tokens[tokens.index("--workdir") + 1] == "/workspace mount/repo"
    assert tokens[tokens.index("--request") + 1] == (
        "/workspace mount/repo/runs/request one.json"
    )
    assert "PYTHONPATH=/workspace mount/repo/workspace" in tokens
    assert IMAGE_A in tokens
    assert IMAGE_B not in tokens


def test_submit_fails_before_job_creation_when_data_mount_is_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = tmp_path / "runs"
    monkeypatch.setenv("SEQ_ALIGN_SLURM_RUNTIME", "docker")
    monkeypatch.setenv("SEQ_ALIGN_SLURM_DOCKER_IMAGE", IMAGE_A)
    monkeypatch.setenv("SEQ_ALIGN_SLURM_DATA_ROOT", str(tmp_path / "missing-data"))
    monkeypatch.setattr(
        slurm_eval.subprocess,
        "run",
        lambda *args, **kwargs: pytest.fail("subprocess must not run after mount preflight failure"),
    )

    result = slurm_eval.submit_and_wait(
        stage="smoke",
        mods=[],
        run_dir=run_dir,
        round_idx=1,
        data_dir=tmp_path / "queries",
        max_iter=1,
        astral_db=tmp_path / "astral.fa",
        uniref_db=tmp_path / "uniref.fa",
        scop_txt=tmp_path / "scop.txt",
    )

    assert result.status == "run_error"
    assert "SEQ_ALIGN_SLURM_DATA_ROOT is not a directory" in result.errors[0]
    assert not run_dir.exists()


def test_submit_persists_captured_image_when_environment_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = tmp_path / "project"
    query_dir = project / "queries"
    query_dir.mkdir(parents=True)
    (query_dir / "all_samples.part_001.fasta").write_text(
        ">query\nMKT\n", encoding="utf-8"
    )
    data_root = tmp_path / "benchmark-data"
    data_root.mkdir()
    astral = data_root / "astral.fa"
    uniref = data_root / "uniref.fa"
    scop = data_root / "scop.txt"
    for path in (astral, uniref, scop):
        path.write_text("fixture\n", encoding="utf-8")

    monkeypatch.setattr(slurm_eval, "PROJECT_ROOT", project)
    monkeypatch.setenv("SEQ_ALIGN_SLURM_RUNTIME", "docker")
    monkeypatch.setenv("SEQ_ALIGN_SLURM_DOCKER_IMAGE", IMAGE_A)
    monkeypatch.setenv("SEQ_ALIGN_SLURM_DATA_ROOT", str(data_root))
    monkeypatch.setattr(slurm_eval, "assemble", lambda mods: _patched_config())
    monkeypatch.setattr(
        run_hmmer_cli,
        "binary_bundle_identity",
        lambda cfg, *, hmmer_src_root, build_context, **_kwargs: {
            "schema_version": 1,
            "source_tree_sha256": "source-sha",
            "patch_set_sha256": "patch-sha",
            "build_context": build_context,
            "configure_policy": "test",
            "bundle_id": "bundle-sha",
        },
    )
    original_snapshot = slurm_eval._snapshot_queries

    def snapshot_and_change_image(data_dir: Path, destination: Path) -> list[Path]:
        paths = original_snapshot(data_dir, destination)
        monkeypatch.setenv("SEQ_ALIGN_SLURM_DOCKER_IMAGE", IMAGE_B)
        return paths

    monkeypatch.setattr(slurm_eval, "_snapshot_queries", snapshot_and_change_image)
    build_calls: list[dict] = []

    def fake_build(**kwargs):
        build_calls.append(kwargs)
        return (
            {"status": "ok"},
            {"job_id": "build-1", "runtime": "docker", "image_ref": IMAGE_A},
        )

    monkeypatch.setattr(slurm_eval, "_ensure_candidate_bundle", fake_build)
    commands: list[list[str]] = []

    def fake_run(command, **kwargs):
        commands.append(command)
        return type("Submitted", (), {"returncode": 0, "stdout": "123\n", "stderr": ""})()

    monkeypatch.setattr(slurm_eval.subprocess, "run", fake_run)
    monkeypatch.setattr(
        slurm_eval,
        "_wait_for_job",
        lambda **kwargs: EvalResult(stage=kwargs["stage"], status="ok"),
    )

    result = slurm_eval.submit_and_wait(
        stage="smoke",
        mods=[],
        run_dir=project / "runs",
        round_idx=0,
        data_dir=query_dir,
        max_iter=1,
        astral_db=astral,
        uniref_db=uniref,
        scop_txt=scop,
    )

    assert result.status == "ok"
    job_dir = next((project / "runs" / "round_000" / "slurm").iterdir())
    request = json.loads((job_dir / "request.json").read_text(encoding="utf-8"))
    submission = json.loads((job_dir / "submission.json").read_text(encoding="utf-8"))
    worker = commands[0][commands[0].index("--wrap") + 1]
    assert request["runtime"] == submission["runtime"] == "docker"
    assert request["image_ref"] == submission["image_ref"] == IMAGE_A
    assert request["hmmer_bundle_identity"]["image_ref"] == IMAGE_A
    assert request["hmmer_build_context"] == f"docker:{IMAGE_A}"
    assert build_calls[0]["image_ref"] == IMAGE_A
    assert IMAGE_A in shlex.split(worker)
    assert IMAGE_B not in shlex.split(worker)


def test_build_submission_uses_request_image_not_current_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    data_root = tmp_path / "benchmark-data"
    data_root.mkdir()
    request_path = project / "request.json"
    request_path.write_text("{}", encoding="utf-8")
    job_dir = project / "job"
    job_dir.mkdir()
    monkeypatch.setattr(slurm_eval, "PROJECT_ROOT", project)
    monkeypatch.setenv("SEQ_ALIGN_SLURM_DATA_ROOT", str(data_root))
    monkeypatch.setenv("SEQ_ALIGN_SLURM_DOCKER_IMAGE", IMAGE_B)
    commands: list[list[str]] = []

    def fake_run(command, **kwargs):
        commands.append(command)
        return type("Submitted", (), {"returncode": 0, "stdout": "build-7\n", "stderr": ""})()

    monkeypatch.setattr(slurm_eval.subprocess, "run", fake_run)
    monkeypatch.setattr(
        slurm_eval,
        "_wait_for_build",
        lambda **kwargs: {"status": "ok"},
    )

    _, submission = slurm_eval._ensure_candidate_bundle(
        request_path=request_path,
        job_dir=job_dir,
        host_bundle_dir=job_dir / "missing-bundle",
        bundle_id="bundle-sha",
        patch_set_sha256="patch-sha",
        runtime="docker",
        image_ref=IMAGE_A,
        submitted_epoch=1.0,
    )

    persisted = json.loads(
        (job_dir / "build_submission.json").read_text(encoding="utf-8")
    )
    worker = commands[0][commands[0].index("--wrap") + 1]
    assert submission == persisted
    assert persisted["runtime"] == "docker"
    assert persisted["image_ref"] == IMAGE_A
    assert IMAGE_A in shlex.split(worker)
    assert IMAGE_B not in shlex.split(worker)


def test_docker_job_is_not_resumed_after_pinned_image_changes(
    tmp_path: Path,
) -> None:
    round_path = tmp_path / "round_001"
    job_dir = round_path / "slurm" / "full_100"
    result_dir = job_dir / "results"
    result_dir.mkdir(parents=True)
    source_paths = {
        "data_dir": "/data/queries",
        "astral_db": "/data/astral.fa",
        "uniref_db": "/data/uniref.fa",
        "scop_txt": "/data/scop.txt",
    }
    objective = {"objective_id": "test"}
    request = {
        "stage": "full",
        "mods": [],
        "max_iter": 5,
        "source_paths": source_paths,
        "query_paths": ["query0"],
        "optimization_objective": objective,
        "runtime": "docker",
        "image_ref": IMAGE_A,
        # The core/concurrency shape this request was measured under. Resume now
        # requires it to match the live configuration, so a fixture that omits it
        # describes a measurement whose shape is unknown.
        "resources": asdict(slurm_eval.resources_for("full")),
        "query_cpus": slurm_eval.query_cpus_for("full"),
        "queries_per_job": slurm_eval.queries_per_job("full"),
    }
    submission = {
        "job_id": "987",
        "submitted_epoch": 1.0,
        "resources": {
            "partition": "compute",
            "cpus": 1,
            "memory": "8G",
            "walltime": "12:00:00",
        },
        "max_parallel": 1,
        "query_count": 1,
        "runtime": "docker",
        "image_ref": IMAGE_A,
    }
    (job_dir / "request.json").write_text(json.dumps(request), encoding="utf-8")
    (job_dir / "submission.json").write_text(json.dumps(submission), encoding="utf-8")
    (result_dir / "result_000.json").write_text(
        EvalResult(stage="full", status="ok").model_dump_json(), encoding="utf-8"
    )

    changed = slurm_eval._find_resumable_job(
        round_path=round_path,
        stage="full",
        mods=[],
        max_iter=5,
        source_paths=source_paths,
        optimization_objective=objective,
        runtime="docker",
        image_ref=IMAGE_B,
    )
    unchanged = slurm_eval._find_resumable_job(
        round_path=round_path,
        stage="full",
        mods=[],
        max_iter=5,
        source_paths=source_paths,
        optimization_objective=objective,
        runtime="docker",
        image_ref=IMAGE_A,
    )

    assert changed is None
    assert unchanged is not None


def test_candidate_bundle_is_built_once_reused_and_hash_checked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source"
    (source / "src").mkdir(parents=True)
    (source / "src" / "kernel.c").write_text("int value = 1;\n", encoding="utf-8")
    cfg = _patched_config()
    builds: list[str] = []

    def fake_copy(source_root: Path, build_root: Path) -> None:
        shutil.copytree(source_root, build_root)

    def fake_apply(build_root: Path, candidate) -> None:
        (build_root / "src" / "kernel.c").write_text("int value = 2;\n", encoding="utf-8")

    def fake_build(build_root: Path, *, cpus: int, **_kwargs):
        payload = (build_root / "src" / "kernel.c").read_bytes()
        builds.append(payload.decode())
        for name in ("jackhmmer", "hmmsearch"):
            path = build_root / "src" / name
            path.write_bytes(name.encode() + b":" + payload)
            path.chmod(0o755)
        return build_root / "src" / "jackhmmer", build_root / "src" / "hmmsearch"

    monkeypatch.setattr(run_hmmer_cli, "_copy_source", fake_copy)
    monkeypatch.setattr(run_hmmer_cli, "_apply_patches", fake_apply)
    monkeypatch.setattr(run_hmmer_cli, "_build_hmmer", fake_build)

    identity = run_hmmer_cli.binary_bundle_identity(
        cfg, hmmer_src_root=source, build_context="test-image"
    )
    bundle_dir = tmp_path / "bundles" / str(identity["bundle_id"])
    first = run_hmmer_cli.build_binary_bundle(
        cfg,
        bundle_dir,
        hmmer_src_root=source,
        build_context="test-image",
        expected_bundle_id=str(identity["bundle_id"]),
    )
    second = run_hmmer_cli.build_binary_bundle(
        cfg,
        bundle_dir,
        hmmer_src_root=source,
        build_context="test-image",
        expected_bundle_id=str(identity["bundle_id"]),
    )

    assert first["bundle_id"] == second["bundle_id"] == identity["bundle_id"]
    assert first["source_tree_sha256"] == identity["source_tree_sha256"]
    assert first["excluded_from_cost_s"] is True
    assert len(builds) == 2  # stock provenance + one incremental candidate build

    binary = bundle_dir / "bin" / "jackhmmer"
    binary.chmod(0o755)
    binary.write_bytes(b"tampered")
    with pytest.raises(RuntimeError, match="binary hash mismatch"):
        run_hmmer_cli.load_binary_bundle(
            bundle_dir, expected_bundle_id=str(identity["bundle_id"])
        )


def test_array_query_workers_receive_the_same_prebuilt_bundle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = _patched_config()
    monkeypatch.setattr(slurm_eval, "assemble", lambda mods: cfg)
    calls: list[dict] = []

    def fake_run_all_queries(candidate, **kwargs):
        calls.append(kwargs)
        return EvalResult(stage=kwargs["stage"], status="ok", tp=1, cost_s=3.0, n_seqs=1)

    monkeypatch.setattr(run_hmmer_cli, "run_all_queries", fake_run_all_queries)
    queries = []
    for index in range(2):
        path = tmp_path / f"query_{index}.fasta"
        path.write_text(f">q{index}\nMKT\n", encoding="utf-8")
        queries.append(str(path))
    request = {
        "stage": "full",
        "mods": [],
        "query_paths": queries,
        "max_iter": 3,
        "astral_db": str(tmp_path / "astral.fa"),
        "uniref_db": str(tmp_path / "uniref.fa"),
        "scop_txt": str(tmp_path / "scop.txt"),
        "work_root": str(tmp_path / "work"),
        "resources": {"partition": "compute", "cpus": 1, "memory": "16G", "walltime": "12:00:00"},
        # Cores ONE query may use (the literal HMMER --cpu). It is deliberately
        # distinct from the job allocation recorded in ``resources`` above, so the
        # worker must read this key and not ``resources.cpus``.
        "query_cpus": 1,
        "hmmer_bundle_dir": str(tmp_path / "bundle"),
        "hmmer_bundle_id": "candidate-sha",
    }
    request_path = tmp_path / "request.json"
    request_path.write_text(json.dumps(request), encoding="utf-8")

    assert slurm_eval.run_request(request_path, 0).status == "ok"
    assert slurm_eval.run_request(request_path, 1).status == "ok"
    assert [call["hmmer_bundle_id"] for call in calls] == ["candidate-sha", "candidate-sha"]
    assert len({call["hmmer_bundle_dir"] for call in calls}) == 1
    assert all(call["cpus"] == 1 for call in calls)


def test_cli_query_uses_bundle_without_copying_or_compiling_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = _patched_config()
    bundle_dir = tmp_path / "bundle"
    bin_dir = bundle_dir / "bin"
    bin_dir.mkdir(parents=True)
    artifacts = {}
    for name in ("jackhmmer", "hmmsearch"):
        path = bin_dir / name
        shutil.copy2("/bin/true", path)
        path.chmod(0o555)
        artifacts[name] = {
            "path": f"bin/{name}",
            "size": path.stat().st_size,
            "sha256": run_hmmer_cli._hash_file(path),
        }
    manifest = {
        "schema_version": run_hmmer_cli.BUNDLE_SCHEMA_VERSION,
        "bundle_id": "candidate-sha",
        "source_tree_sha256": "source-sha",
        "patch_set_sha256": run_hmmer_cli._patch_set_digest(cfg),
        "binaries": artifacts,
        "build": {"mode": "content_addressed_bundle", "runtime_effective": True},
        "runtime_effective": True,
        "excluded_from_cost_s": True,
    }
    (bundle_dir / run_hmmer_cli.BUNDLE_MANIFEST_NAME).write_text(
        json.dumps(manifest), encoding="utf-8"
    )

    data_dir = tmp_path / "queries"
    data_dir.mkdir()
    (data_dir / "all_samples.part_0.fasta").write_text(">q0\nMKT\n", encoding="utf-8")
    astral = tmp_path / "astral.fa"
    uniref = tmp_path / "uniref.fa"
    scop = tmp_path / "scop.txt"
    astral.write_text(">q0\nMKT\n", encoding="utf-8")
    uniref.write_text(">u0\nMKT\n", encoding="utf-8")
    scop.write_text("", encoding="utf-8")

    monkeypatch.setattr(
        run_hmmer_cli,
        "_build_hmmer",
        lambda *args, **kwargs: pytest.fail("query worker must not compile HMMER"),
    )
    monkeypatch.setattr(
        run_hmmer_cli,
        "_copy_source",
        lambda *args, **kwargs: pytest.fail("query worker must not copy HMMER source"),
    )
    monkeypatch.setattr(run_hmmer_cli.run_jack, "_load_scop_lineages", lambda path: {})
    monkeypatch.setattr(
        run_hmmer_cli,
        "_run_one_query",
        lambda *args, **kwargs: EvalResult(
            stage="full", status="ok", tp=2, fp=0, cost_s=7.0, n_seqs=1,
            meta={"query_id": "q0"},
        ),
    )

    result = run_hmmer_cli.run_all_queries(
        cfg,
        data_dir=data_dir,
        stage="full",
        astral_db=astral,
        uniref_db=uniref,
        scop_txt=scop,
        hmmer_src_root=tmp_path / "source-does-not-need-to-exist",
        work_root=tmp_path / "query-work",
        hmmer_bundle_dir=bundle_dir,
        hmmer_bundle_id="candidate-sha",
    )

    assert result.status == "ok"
    assert result.cost_s == 7.0
    assert result.meta["hmmer_bundle"]["source_tree_sha256"] == "source-sha"
    assert result.meta["jackhmmer_sha256"] == artifacts["jackhmmer"]["sha256"]


def test_aggregate_records_bundle_hashes_but_excludes_build_from_cost(
    tmp_path: Path,
) -> None:
    bundle = {
        "bundle_id": "candidate-sha",
        "source_tree_sha256": "source-sha",
        "patch_set_sha256": "patch-sha",
        "binaries": {
            "jackhmmer": {"sha256": "jack-sha"},
            "hmmsearch": {"sha256": "search-sha"},
        },
        "build_wall_s": 50.0,
        "excluded_from_cost_s": True,
    }
    slurm_eval._write_json_atomic(
        tmp_path / "build_result.json", {"status": "ok", "bundle": bundle}
    )
    slurm_eval._write_json_atomic(
        tmp_path / "build_submission.json", {"job_id": "build-7"}
    )
    results = [
        EvalResult(stage="full", status="ok", tp=1, cost_s=11.0, n_seqs=1),
        EvalResult(stage="full", status="ok", tp=2, cost_s=13.0, n_seqs=1),
    ]

    aggregate = slurm_eval._aggregate_results(
        "full",
        results,
        job_id="array-8",
        resources=slurm_eval.SlurmResources("compute", 1, "16G", "12:00:00"),
        max_parallel=10,
        job_dir=tmp_path,
        batch_wall_s=70.0,
    )

    assert aggregate.cost_s == 24.0
    assert aggregate.meta["batch_wall_s"] == 70.0
    assert aggregate.meta["build_wall_s"] == 50.0
    assert aggregate.meta["build_job_id"] == "build-7"
    assert aggregate.meta["cost_s_excludes_build"] is True
    assert aggregate.meta["hmmer_bundle"]["source_tree_sha256"] == "source-sha"
    assert aggregate.meta["hmmer_bundle"]["binaries"]["jackhmmer"]["sha256"] == "jack-sha"
