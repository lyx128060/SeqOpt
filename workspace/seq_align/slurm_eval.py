"""Submit isolated SeqOpt evaluations to Slurm and collect their result."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .assembler import assemble
from .components.base import ComponentModification
from .eval_schema import EvalResult
from .optimization_objective import objective_metadata

PROJECT_ROOT = Path(__file__).resolve().parents[2]
POLL_INTERVAL_S = 5


@dataclass(frozen=True)
class SlurmResources:
    partition: str
    cpus: int
    memory: str
    walltime: str


def resources_for(stage: str) -> SlurmResources:
    """Read fixed per-stage Slurm requests from the environment.

    ``cpus`` is the *job* allocation (``sbatch --cpus-per-task``). It is NOT the
    number of cores a query gets: a job runs ``queries_per_job(stage)`` queries
    concurrently, each owning ``query_cpus_for(stage)`` HMMER worker threads plus
    one coordinator thread. The default allocation is therefore derived, never a
    bare ``1``: the old default of 1 core per job meant ``jackhmmer --cpu 0``
    (HMMER's "detect the machine" sentinel) multiplied by five concurrent queries
    inside a one-core cgroup -- recorded, meanwhile, as ``cpus: 8`` in the
    evaluation config. Two disagreeing core numbers and one of them silently
    self-detecting is exactly the class of bug the comparability gate refuses.

    The full-stage wall clock is deliberately generous (24h). A full comparison
    round on the production query set runs two repeats plus the interleaved paired
    re-measurements, and every one of those is a complete 50-query evaluation
    across 10 array jobs; a 12h request can expire mid-round and turn a measurable
    iteration into a ``repeatedly_timed_out`` slot. The partition limit is 30 days,
    so 24h is a safe request.
    """
    prefix = "SEQ_ALIGN_SLURM_SMOKE" if stage == "smoke" else "SEQ_ALIGN_SLURM_FULL"
    derived_cpus = min_job_cpus_for(stage)
    defaults = (derived_cpus, "4G", "00:30:00") if stage == "smoke" else (derived_cpus, "16G", "24:00:00")
    return SlurmResources(
        partition=os.getenv("SEQ_ALIGN_SLURM_PARTITION", "compute"),
        cpus=int(os.getenv(f"{prefix}_CPUS", str(defaults[0]))),
        memory=os.getenv(f"{prefix}_MEMORY", defaults[1]),
        walltime=os.getenv(f"{prefix}_TIME", defaults[2]),
    )


def build_resources_for() -> SlurmResources:
    """Return the isolated resources used once per candidate build."""
    return SlurmResources(
        partition=os.getenv("SEQ_ALIGN_SLURM_PARTITION", "compute"),
        cpus=int(os.getenv("SEQ_ALIGN_SLURM_BUILD_CPUS", "4")),
        memory=os.getenv("SEQ_ALIGN_SLURM_BUILD_MEMORY", "8G"),
        walltime=os.getenv("SEQ_ALIGN_SLURM_BUILD_TIME", "00:30:00"),
    )


def max_parallel_for(stage: str) -> int:
    """Return the maximum number of simultaneously running query tasks in one job.

    Defaults to ``queries_per_job``: the job allocation is sized for exactly that
    many concurrent queries, so letting the array run more elements at once than
    the allocation covers would reintroduce oversubscription. ``%N`` and the job
    allocation are two views of the same concurrency number.
    """
    prefix = "SEQ_ALIGN_SLURM_SMOKE" if stage == "smoke" else "SEQ_ALIGN_SLURM_FULL"
    return max(1, int(os.getenv(f"{prefix}_MAX_PARALLEL", str(queries_per_job(stage)))))


def queries_per_job(stage: str) -> int:
    """Return how many queries one Slurm submission covers.

    An N-query run is split into ``ceil(N / chunk)`` submissions rather than one
    N-element array. Each submission still schedules its own queries independently
    and keeps its own ``%N`` throttle, so concurrency behaviour is preserved; what
    changes is that the job count stays bounded and no single job owns the queue.

    Chunks are independent, which is what keeps the rest of the pipeline untouched:
    every worker writes ``result_<global index>.json``, so ``_read_results``,
    ``_aggregate_results`` and the resume path need no notion of chunks at all.
    """
    prefix = "SEQ_ALIGN_SLURM_SMOKE" if stage == "smoke" else "SEQ_ALIGN_SLURM_FULL"
    return max(1, int(os.getenv(f"{prefix}_QUERIES_PER_JOB", "5")))


# ── The two orthogonal core knobs ────────────────────────────────────────────
# "Cores" is two independent quantities and they must never be conflated:
#
#   query_cpus  -- cores ONE query may use. This is the value HMMER receives as
#                  ``--cpu``. It is what a per-query cost number is measured at.
#   queries_per_job -- how many queries share one job. This is the concurrency
#                  (and the ``%N`` throttle) inside a single allocation.
#
# The job allocation is then a *derived* quantity that must cover both:
#
#   job_cpus = queries_per_job * (query_cpus + COORDINATOR_CPUS_PER_QUERY)
#
# The ``+1`` per query is the coordinator/main thread that HMMER adds on top of
# its ``--cpu`` worker threads; it is the "机动" cores. With the production shape
# (5 queries, 8 threads each) that is 5 * 9 = 45 cores per job, and a 50-query run
# is 10 jobs running 10 x 5 = 50 queries in parallel.
DEFAULT_QUERY_CPUS = 8
COORDINATOR_CPUS_PER_QUERY = 1


def query_cpus_for(stage: str) -> int:
    """Return the number of cores one query may use (the literal HMMER ``--cpu``).

    This is deliberately independent of ``resources_for().cpus``: a job-level
    allocation of 45 cores must not turn into ``jackhmmer --cpu 45``.
    """
    prefix = "SEQ_ALIGN_SLURM_SMOKE" if stage == "smoke" else "SEQ_ALIGN_SLURM_FULL"
    return max(1, int(os.getenv(f"{prefix}_QUERY_CPUS", str(DEFAULT_QUERY_CPUS))))


def min_job_cpus_for(stage: str) -> int:
    """Smallest allocation that can run ``queries_per_job`` queries at ``query_cpus``."""
    return queries_per_job(stage) * (query_cpus_for(stage) + COORDINATOR_CPUS_PER_QUERY)


def evaluation_runtime() -> str:
    """Return the isolated execution runtime for Slurm query workers."""
    runtime = os.getenv("SEQ_ALIGN_SLURM_RUNTIME", "host").lower()
    if runtime not in {"host", "docker"}:
        raise ValueError("SEQ_ALIGN_SLURM_RUNTIME must be 'host' or 'docker'")
    return runtime


def _validated_docker_image_ref(value: object) -> str:
    """Return an exact immutable Docker reference or fail closed."""
    image_ref = value.strip() if isinstance(value, str) else ""
    name, separator, digest = image_ref.rpartition("@sha256:")
    if (
        not name
        or not separator
        or any(character.isspace() for character in image_ref)
        or len(digest) != 64
        or any(character not in "0123456789abcdefABCDEF" for character in digest)
    ):
        raise ValueError(
            "SEQ_ALIGN_SLURM_DOCKER_IMAGE must be pinned by an "
            "@sha256:<64 hex characters> digest"
        )
    return image_ref


def _docker_image_ref() -> str:
    """Capture and validate the immutable image used by this submission."""
    return _validated_docker_image_ref(os.getenv("SEQ_ALIGN_SLURM_DOCKER_IMAGE", ""))


def _host_project_mount_root() -> Path:
    """Return the host directory mapped to the container project root.

    ``slurm-docker-run`` commonly maps ``/shared/<user>`` to ``/workspace``.
    Sites with a different policy can set ``SEQ_ALIGN_SLURM_HOST_PROJECT_ROOT``.
    Outside that layout (for example, a test checkout), map the repository root.
    """
    configured = os.getenv("SEQ_ALIGN_SLURM_HOST_PROJECT_ROOT", "").strip()
    if configured:
        root = Path(configured).expanduser().resolve()
    else:
        parts = PROJECT_ROOT.parts
        if len(parts) >= 3 and parts[0] == os.sep and parts[1] == "shared":
            root = Path("/shared") / parts[2]
        else:
            root = PROJECT_ROOT.resolve()
    try:
        PROJECT_ROOT.resolve().relative_to(root)
    except ValueError as exc:
        raise ValueError(f"project {PROJECT_ROOT.resolve()} is outside configured mount root {root}") from exc
    return root


def _container_project_mount_root() -> Path:
    return _container_mount_root("SEQ_ALIGN_SLURM_CONTAINER_PROJECT_ROOT", "/workspace")


def _host_data_mount_root() -> Path:
    configured = os.getenv("SEQ_ALIGN_SLURM_DATA_ROOT", "").strip()
    if not configured:
        raise ValueError(
            "SEQ_ALIGN_SLURM_DATA_ROOT is required for Docker evaluation; "
            "set it to the common host directory containing benchmark files"
        )
    root = Path(configured).expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"SEQ_ALIGN_SLURM_DATA_ROOT is not a directory: {root}")
    if ":" in str(root):
        raise ValueError("SEQ_ALIGN_SLURM_DATA_ROOT must not contain ':'")
    return root


def _container_data_mount_root() -> Path:
    return _container_mount_root("SEQ_ALIGN_SLURM_CONTAINER_DATA_ROOT", "/data")


def _container_mount_root(name: str, default: str) -> Path:
    raw = os.getenv(name, "").strip() or default
    root = Path(raw)
    if not root.is_absolute():
        raise ValueError(f"{name} must be an absolute container path: {raw!r}")
    if ":" in raw:
        raise ValueError(f"{name} must not contain ':'")
    return root


def _container_project_path(path: Path) -> Path:
    """Translate a project path through the configured project mount."""
    resolved = path.resolve()
    try:
        relative = resolved.relative_to(_host_project_mount_root())
    except ValueError as exc:
        raise ValueError(f"project path is outside the configured project mount: {resolved}") from exc
    return _container_project_mount_root() / relative


def _container_data_path(path: Path) -> Path:
    """Translate benchmark data through the configured read-only mount."""
    source_root = _host_data_mount_root()
    try:
        return _container_data_mount_root() / path.resolve().relative_to(source_root)
    except ValueError as exc:
        raise ValueError(f"benchmark data must live below {source_root}: {path}") from exc


def _container_visible_path(path: Path) -> Path:
    """Map a host path that is visible through one of the Docker worker mounts."""
    resolved = path.resolve()
    try:
        return _container_project_path(resolved)
    except ValueError:
        return _container_data_path(resolved)


def _docker_worker_command(
    request_path: Path,
    *,
    image_ref: str,
    mode: str = "query",
    query_offset: int = 0,
) -> str:
    """Build a worker command bound to the image captured in its request."""
    image_ref = _validated_docker_image_ref(image_ref)
    container_root = _container_project_path(PROJECT_ROOT)
    container_request = _container_project_path(request_path)
    host_data_root = _host_data_mount_root()
    container_data_root = _container_data_mount_root()
    return " ".join(
        (
            # Some Slurm configurations expose the physical-core allocation as
            # SLURM_CPUS_PER_TASK=2 while retaining the requested
            # SLURM_TRES_PER_TASK=cpu=1. Nested srun rejects those conflicting
            # values, so keep the requested TRES as the single source of truth.
            "/usr/bin/env -u SLURM_CPUS_PER_TASK srun --cpu-bind=cores /usr/bin/sudo -n slurm-docker-run",
            "--pull missing",
            "--mount", shlex.quote(f"{host_data_root}:{container_data_root}:ro"),
            "--env", shlex.quote(f"PYTHONPATH={container_root / 'workspace'}"),
            "--env", "SEQ_ALIGN_HMMER_BIN_DIR=/opt/hmmer-src/src",
            *(('--env', 'SEQ_ALIGN_HMMER_BUILD_WORK_ROOT=/local/job/seqalign-build') if mode == "build" else ()),
            # slurm-docker-run forwards only selected Slurm variables; retain the
            # array index explicitly so each query container owns one query, and
            # inject this chunk's global offset so the container still maps to the
            # right query of the whole run.
            *(('--env', 'SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID') if mode == "query" else ()),
            *(('--env', f'SEQALIGN_QUERY_OFFSET={int(query_offset)}') if mode == "query" else ()),
            "--workdir", shlex.quote(str(container_root)),
            shlex.quote(image_ref),
            "/opt/evomaster/.venv/bin/python",
            "-m seq_align.slurm_eval",
            "--request", shlex.quote(str(container_request)),
            "--mode", shlex.quote(mode),
        )
    )


def _host_worker_command(request_path: Path, *, mode: str) -> str:
    python = PROJECT_ROOT / ".venv" / "bin" / "python"
    if not python.is_file():
        raise ValueError(f"Slurm worker Python not found: {python}")
    return " ".join(
        (
            "/usr/bin/env -u SLURM_CPUS_PER_TASK srun --cpu-bind=cores /usr/bin/env",
            f"PYTHONPATH={shlex.quote(str(PROJECT_ROOT / 'workspace'))}",
            shlex.quote(str(python)),
            "-m seq_align.slurm_eval",
            "--request", shlex.quote(str(request_path)),
            "--mode", shlex.quote(mode),
        )
    )


def submit_and_wait(
    *,
    stage: str,
    mods: list[dict[str, Any]],
    run_dir: str | Path,
    round_idx: int,
    data_dir: str | Path,
    max_iter: int,
    astral_db: str | Path,
    uniref_db: str | Path,
    scop_txt: str | Path,
    query_timeout_reference: str | Path | None = None,
    profiling: bool = False,
    expected_query_cpus: int | None = None,
) -> EvalResult:
    """Snapshot an evaluation, submit it, and return the worker's result.

    ``profiling`` switches the whole job to the instrumented profiling build: the
    bundle is built with the p7_evo_prof stage timers compiled in and every query
    worker points $P7_EVO_PROF_OUT at its own work directory. It is a build-level
    variant, so it goes through the same bundle/resume/aggregation machinery as any
    other candidate -- only the bundle identity differs.

    ``expected_query_cpus`` is the per-query core count the caller believes it is
    measuring at (``PatchRunner.cpus``). It is checked against the value the Slurm
    submission will actually dispatch and a disagreement is a hard error: the old
    code let the recorded ``eval_config.cpus`` and the real ``--cpus-per-task``
    drift apart silently, so a record could claim 8 while the job got 1.
    """
    batch_started_epoch = time.time()
    resources = resources_for(stage)
    query_cpus = query_cpus_for(stage)
    min_job_cpus = min_job_cpus_for(stage)
    # Fail closed on the two silent-divergence modes this configuration used to have.
    if expected_query_cpus is not None and int(expected_query_cpus) != query_cpus:
        return _error(
            stage,
            "per-query core count mismatch: the runner is configured for "
            f"{int(expected_query_cpus)} core(s) per query but this submission would dispatch "
            f"{query_cpus}. Set SEQ_ALIGN_SLURM_{'SMOKE' if stage == 'smoke' else 'FULL'}_QUERY_CPUS="
            f"{int(expected_query_cpus)} (or fix --cpus); a cost measured at one core count must "
            "never be recorded as another.",
        )
    if resources.cpus < min_job_cpus:
        return _error(
            stage,
            f"job allocation too small: --cpus-per-task={resources.cpus} cannot run "
            f"{queries_per_job(stage)} concurrent queries at {query_cpus} cores each plus one "
            f"coordinator; need at least {min_job_cpus}. Raise "
            f"SEQ_ALIGN_SLURM_{'SMOKE' if stage == 'smoke' else 'FULL'}_CPUS or lower "
            f"SEQ_ALIGN_SLURM_{'SMOKE' if stage == 'smoke' else 'FULL'}_QUERIES_PER_JOB.",
        )
    runtime = evaluation_runtime()
    run_path = Path(run_dir).resolve()
    image_ref: str | None = None
    if runtime == "docker":
        try:
            image_ref = _docker_image_ref()
        except ValueError as exc:
            return _error(stage, str(exc))
        try:
            # Resolve every mount and translated input before creating a job
            # directory or invoking sbatch. This catches missing/mis-scoped
            # benchmark roots on the controller host.
            _host_data_mount_root()
            _container_project_mount_root()
            _container_data_mount_root()
            _container_project_path(run_path)
            for path in (astral_db, uniref_db, scop_txt):
                _container_visible_path(Path(path))
        except ValueError as exc:
            return _error(stage, f"invalid Docker mount configuration: {exc}")
    round_path = run_path / f"round_{round_idx:03d}"
    source_paths = {
        "data_dir": str(Path(data_dir).resolve()),
        "astral_db": str(Path(astral_db).resolve()),
        "uniref_db": str(Path(uniref_db).resolve()),
        "scop_txt": str(Path(scop_txt).resolve()),
    }
    query_timeout_policy = _full_query_timeout_policy(
        stage=stage,
        run_dir=Path(run_dir).resolve(),
        round_idx=round_idx,
        data_dir=Path(data_dir).resolve(),
        reference_result_path=(
            Path(query_timeout_reference).resolve()
            if query_timeout_reference is not None
            else None
        ),
    )
    optimization_objective = objective_metadata()
    try:
        candidate_cfg = assemble([ComponentModification(**mod) for mod in mods])
    except Exception as exc:
        return EvalResult(
            stage=stage,
            status="compile_error",
            errors=[f"invalid component modifications: {type(exc).__name__}: {exc}"],
        )
    if candidate_cfg.errors:
        return EvalResult(stage=stage, status="compile_error", errors=candidate_cfg.errors)

    bundle: dict[str, Any] | None = None
    # EVERY evaluation compiles exactly once and points all of its query jobs at
    # that single artifact. A patch-less candidate (the stock baseline) is not
    # exempt: it used to fall through to hmmer_bundle_dir=None, which made each
    # query worker compile HMMER for itself -- 50 workers, 50 compiles, and no
    # single "the code we actually measured" to point at. A patch-less build is
    # harmless for the acceptance gate: _build_provenance reports
    # runtime_effective=None (not False) when cfg.c_patches is empty, and the
    # gate only rejects on False.
    from .run_hmmer_cli import DEFAULT_HMMER_ROOT, binary_bundle_identity

    if runtime == "docker":
        # Bind the content-addressed bundle identity to the exact image
        # captured above; a later environment change cannot alter it.
        build_context = f"docker:{image_ref}"
    else:
        build_context = os.getenv("SEQ_ALIGN_HMMER_BUILD_CONTEXT", "host-cluster")
    try:
        identity = binary_bundle_identity(
            candidate_cfg,
            hmmer_src_root=DEFAULT_HMMER_ROOT,
            build_context=build_context,
            profiling=profiling,
        )
    except Exception as exc:
        return EvalResult(
            stage=stage,
            status="compile_error",
            errors=[f"could not identify HMMER candidate bundle: {type(exc).__name__}: {exc}"],
        )
    bundle_id = str(identity["bundle_id"])
    bundle = {
        **identity,
        "image_ref": image_ref,
        "host_dir": str((run_path / "hmmer_bundles" / bundle_id).resolve()),
    }

    resumed = _find_resumable_job(
        round_path=round_path,
        stage=stage,
        mods=mods,
        max_iter=max_iter,
        source_paths=source_paths,
        query_timeout_policy=query_timeout_policy,
        optimization_objective=optimization_objective,
        bundle_id=str(bundle["bundle_id"]) if bundle else None,
        runtime=runtime,
        image_ref=image_ref,
    )
    if resumed is not None:
        job_dir, request, submission = resumed
        resumed_resources = SlurmResources(**submission["resources"])
        # Older submissions recorded one array job; chunked ones record one job per
        # chunk. Accept both so an interrupted run still re-attaches.
        resumed_job_ids = [str(item) for item in (submission.get("job_ids") or [])]
        if not resumed_job_ids:
            resumed_job_ids = [str(submission["job_id"])]
        return _wait_for_job(
            stage=stage,
            job_ids=resumed_job_ids,
            job_dir=job_dir,
            expected=len(request["query_paths"]),
            resources=resumed_resources,
            max_parallel=int(submission["max_parallel"]),
            submitted_epoch=float(submission["submitted_epoch"]),
            query_timeout_policy=request.get("query_timeout_policy"),
        )

    resumed_build = _find_resumable_build(
        round_path=round_path,
        stage=stage,
        mods=mods,
        max_iter=max_iter,
        source_paths=source_paths,
        query_timeout_policy=query_timeout_policy,
        optimization_objective=optimization_objective,
        bundle_id=str(bundle["bundle_id"]) if bundle else None,
        runtime=runtime,
        image_ref=image_ref,
    )
    if resumed_build is not None:
        job_dir, request, build_submission = resumed_build
        build_result = _wait_for_build(
            job_id=str(build_submission["job_id"]),
            result_path=job_dir / "build_result.json",
            resources=SlurmResources(**build_submission["resources"]),
        )
        if build_result.get("status") != "ok":
            return EvalResult(
                stage=stage,
                status="compile_error",
                errors=list(build_result.get("errors") or ["HMMER bundle build failed"]),
                meta={"build_job_id": build_submission.get("job_id"), "build": build_result},
            )
        return _submit_query_jobs(
            stage=stage,
            request_path=job_dir / "request.json",
            request=request,
            job_dir=job_dir,
            resources=resources,
            submitted_epoch=float(build_submission.get("submitted_epoch") or batch_started_epoch),
            build_submission=build_submission,
        )

    job_dir = round_path / "slurm" / f"{stage}_{time.time_ns()}"
    job_dir.mkdir(parents=True, exist_ok=False)
    request_path = job_dir / "request.json"
    query_paths = _snapshot_queries(Path(data_dir), job_dir / "queries")
    if not query_paths:
        return _error(stage, f"no query FASTAs under {data_dir}")
    query_paths_value = [str(path) for path in query_paths]
    astral_value = str(Path(astral_db).resolve())
    uniref_value = str(Path(uniref_db).resolve())
    scop_value = str(Path(scop_txt).resolve())
    work_root_value = str((job_dir / "work").resolve())
    result_dir_value = str((job_dir / "results").resolve())
    build_result_value = str((job_dir / "build_result.json").resolve())
    bundle_dir_value = str(bundle["host_dir"]) if bundle else None
    if runtime == "docker":
        query_paths_value = [str(_container_project_path(path)) for path in query_paths]
        astral_value = str(_container_visible_path(Path(astral_db)))
        uniref_value = str(_container_visible_path(Path(uniref_db)))
        scop_value = str(_container_visible_path(Path(scop_txt)))
        work_root_value = "/local/job/seqalign"
        result_dir_value = str(_container_project_path(job_dir / "results"))
        build_result_value = str(_container_project_path(job_dir / "build_result.json"))
        if bundle is not None:
            bundle_dir_value = str(_container_project_path(Path(bundle["host_dir"])))
    request = {
        "stage": stage,
        "mods": mods,
        "query_paths": query_paths_value,
        "max_iter": max_iter,
        "astral_db": astral_value,
        "uniref_db": uniref_value,
        "scop_txt": scop_value,
        "work_root": work_root_value,
        "result_dir": result_dir_value,
        "resources": asdict(resources),
        # Cores ONE query may use. Kept beside (not derived from) ``resources``:
        # the worker passes this straight to HMMER as ``--cpu``.
        "query_cpus": query_cpus,
        "queries_per_job": queries_per_job(stage),
        "source_paths": source_paths,
        "query_timeout_policy": query_timeout_policy,
        "optimization_objective": optimization_objective,
        "runtime": runtime,
        "image_ref": image_ref,
        "hmmer_bundle_dir": bundle_dir_value,
        "hmmer_bundle_id": bundle.get("bundle_id") if bundle else None,
        "hmmer_bundle_identity": bundle,
        "hmmer_build_context": bundle.get("build_context") if bundle else None,
        "build_result_path": build_result_value,
        "build_resources": asdict(build_resources_for()) if bundle else None,
        "profiling": bool(profiling),
    }
    request_path.write_text(json.dumps(request, indent=2), encoding="utf-8")

    build_submission: dict[str, Any] | None = None
    if bundle is not None:
        build_result, build_submission = _ensure_candidate_bundle(
            request_path=request_path,
            job_dir=job_dir,
            host_bundle_dir=Path(str(bundle["host_dir"])),
            bundle_id=str(bundle["bundle_id"]),
            patch_set_sha256=str(bundle["patch_set_sha256"]),
            runtime=runtime,
            image_ref=image_ref,
            submitted_epoch=batch_started_epoch,
        )
        if build_result.get("status") != "ok":
            return EvalResult(
                stage=stage,
                status="compile_error",
                errors=list(build_result.get("errors") or ["HMMER bundle build failed"]),
                meta={
                    "build_job_id": (build_submission or {}).get("job_id"),
                    "build": build_result,
                },
            )
    return _submit_query_jobs(
        stage=stage,
        request_path=request_path,
        request=request,
        job_dir=job_dir,
        resources=resources,
        submitted_epoch=batch_started_epoch,
        build_submission=build_submission,
    )


def _submit_query_jobs(
    *,
    stage: str,
    request_path: Path,
    request: dict[str, Any],
    job_dir: Path,
    resources: SlurmResources,
    submitted_epoch: float,
    build_submission: dict[str, Any] | None,
) -> EvalResult:
    """Submit the run's queries as independent chunked Slurm arrays.

    One submission per ``queries_per_job`` chunk, each keeping its own ``%N``
    throttle -- so the concurrency behaviour inside a chunk is exactly what a single
    N-element array used to give, while the job count stays bounded.

    Every worker still writes ``result_<global index>.json``: a chunk passes its
    offset through the environment and the worker adds its local array id. That is
    what keeps ``_read_results``, ``_aggregate_results`` and the resume path
    entirely unaware that chunks exist.

    Chunks are submitted only after the build has finished (the caller guarantees
    it), and they all read that one immutable bundle, so 10 jobs share 1 build.
    """
    runtime = str(request.get("runtime") or "host")
    image_ref: str | None = None
    if runtime == "docker":
        try:
            image_ref = _validated_docker_image_ref(request.get("image_ref"))
        except (KeyError, TypeError, ValueError) as exc:
            return _error(stage, f"invalid immutable Docker request: {exc}")

    max_parallel = max_parallel_for(stage)
    query_paths = request["query_paths"]
    chunk = queries_per_job(stage)

    job_ids: list[str] = []
    offsets: list[int] = []
    for offset in range(0, len(query_paths), chunk):
        count = min(chunk, len(query_paths) - offset)
        try:
            worker = (
                _docker_worker_command(
                    request_path,
                    image_ref=str(image_ref),
                    mode="query",
                    query_offset=offset,
                )
                if runtime == "docker"
                else _host_worker_command(request_path, mode="query")
            )
        except (KeyError, TypeError, ValueError) as exc:
            return _error(stage, str(exc))

        command = [
            "sbatch", "--parsable", "--partition", resources.partition,
            "--nodes", "1", "--ntasks", "1", "--cpus-per-task", str(resources.cpus),
            "--mem", resources.memory, "--time", resources.walltime,
            "--hint", "nomultithread",
            "--chdir", str(PROJECT_ROOT),
            "--output", str(job_dir / "slurm-%j.out"),
            "--error", str(job_dir / "slurm-%j.err"),
            # Explicitly omit GCC_EXEC_PREFIX and inherited compiler search paths.
            "--job-name", f"seqalign-{stage}",
            "--export",
            f"PATH,PYTHONPATH,CONDA_PREFIX,CONDA_ENVS_PATH,SEQALIGN_QUERY_OFFSET={offset}",
            "--array", f"0-{count - 1}%{max_parallel}",
            "--wrap", worker,
        ]
        submitted = subprocess.run(command, capture_output=True, text=True)
        if submitted.returncode != 0:
            # Record what already made it into the queue before reporting, so a
            # retry can fill the gaps from the recorded offsets instead of
            # resubmitting queries that are already running or done.
            _write_query_submission(
                job_dir=job_dir, job_ids=job_ids, offsets=offsets, chunk=chunk,
                submitted_epoch=submitted_epoch, resources=resources,
                max_parallel=max_parallel, query_paths=query_paths,
                build_submission=build_submission, request=request,
                runtime=runtime, image_ref=image_ref,
            )
            return _error(
                stage,
                f"sbatch failed for the chunk at offset {offset}: "
                f"{submitted.stderr.strip() or submitted.stdout.strip()}",
            )
        job_ids.append(submitted.stdout.strip().split(";", 1)[0])
        offsets.append(offset)

    submission = _write_query_submission(
        job_dir=job_dir, job_ids=job_ids, offsets=offsets, chunk=chunk,
        submitted_epoch=submitted_epoch, resources=resources,
        max_parallel=max_parallel, query_paths=query_paths,
        build_submission=build_submission, request=request,
        runtime=runtime, image_ref=image_ref,
    )
    return _wait_for_job(
        stage=stage,
        job_ids=job_ids,
        job_dir=job_dir,
        expected=len(query_paths),
        resources=resources,
        max_parallel=int(submission["max_parallel"]),
        submitted_epoch=submitted_epoch,
        query_timeout_policy=request.get("query_timeout_policy"),
    )


def _write_query_submission(
    *,
    job_dir: Path,
    job_ids: list[str],
    offsets: list[int],
    chunk: int,
    submitted_epoch: float,
    resources: SlurmResources,
    max_parallel: int,
    query_paths: list[str],
    build_submission: dict[str, Any] | None,
    request: dict[str, Any],
    runtime: str,
    image_ref: str | None,
) -> dict[str, Any]:
    """Record the chunked submission so an interrupted controller can re-attach.

    ``job_id`` is written alongside ``job_ids`` on purpose: a run submitted before
    chunking still resumes, and single-job callers keep working unchanged.
    """
    submission = {
        "job_id": job_ids[0] if job_ids else None,
        "job_ids": list(job_ids),
        "chunk_offsets": list(offsets),
        "queries_per_job": chunk,
        "submitted_epoch": submitted_epoch,
        "resources": asdict(resources),
        "max_parallel": max_parallel,
        "query_count": len(query_paths),
        "build_job_id": (build_submission or {}).get("job_id"),
        "hmmer_bundle_id": request.get("hmmer_bundle_id"),
        "runtime": runtime,
        "image_ref": image_ref,
    }
    submission_path = job_dir / "submission.json"
    submission_tmp = submission_path.with_suffix(".tmp")
    submission_tmp.write_text(json.dumps(submission, indent=2), encoding="utf-8")
    submission_tmp.replace(submission_path)
    return submission



def _ensure_candidate_bundle(
    *,
    request_path: Path,
    job_dir: Path,
    host_bundle_dir: Path,
    bundle_id: str,
    patch_set_sha256: str,
    runtime: str,
    image_ref: str | None,
    submitted_epoch: float,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Reuse a verified bundle or submit exactly one isolated build job."""
    from .run_hmmer_cli import load_binary_bundle

    build_result_path = job_dir / "build_result.json"
    if host_bundle_dir.exists():
        try:
            manifest, _, _ = load_binary_bundle(
                host_bundle_dir,
                expected_bundle_id=bundle_id,
                expected_patch_set_sha256=patch_set_sha256,
            )
        except Exception as exc:
            return ({
                "status": "compile_error",
                "errors": [f"existing immutable HMMER bundle is invalid: {type(exc).__name__}: {exc}"],
            }, None)
        result = {
            "status": "ok",
            "bundle": manifest,
            "bundle_reused": True,
            "excluded_from_cost_s": True,
        }
        _write_json_atomic(build_result_path, result)
        return result, None

    resources = build_resources_for()
    try:
        if runtime == "docker":
            image_ref = _validated_docker_image_ref(image_ref)
        worker = (
            _docker_worker_command(
                request_path,
                image_ref=image_ref,
                mode="build",
            )
            if runtime == "docker"
            else _host_worker_command(request_path, mode="build")
        )
    except ValueError as exc:
        return ({"status": "compile_error", "errors": [str(exc)]}, None)
    command = [
        "sbatch", "--parsable", "--partition", resources.partition,
        "--nodes", "1", "--ntasks", "1", "--cpus-per-task", str(resources.cpus),
        "--mem", resources.memory, "--time", resources.walltime,
        "--hint", "nomultithread",
        "--chdir", str(PROJECT_ROOT),
        "--output", str(job_dir / "build-slurm-%j.out"),
        "--error", str(job_dir / "build-slurm-%j.err"),
        "--job-name", "seqalign-build", "--export", "PATH,CONDA_PREFIX,CONDA_ENVS_PATH,PYTHONPATH",
        "--wrap", worker,
    ]
    submitted = subprocess.run(command, capture_output=True, text=True)
    if submitted.returncode != 0:
        message = submitted.stderr.strip() or submitted.stdout.strip()
        return ({"status": "compile_error", "errors": [f"build sbatch failed: {message}"]}, None)
    job_id = submitted.stdout.strip().split(";", 1)[0]
    submission = {
        "job_id": job_id,
        "submitted_epoch": submitted_epoch,
        "resources": asdict(resources),
        "hmmer_bundle_id": bundle_id,
        "runtime": runtime,
        "image_ref": image_ref,
    }
    _write_json_atomic(job_dir / "build_submission.json", submission)
    result = _wait_for_build(
        job_id=job_id,
        result_path=build_result_path,
        resources=resources,
    )
    return result, submission


def _wait_for_build(
    *,
    job_id: str,
    result_path: Path,
    resources: SlurmResources,
) -> dict[str, Any]:
    deadline = time.monotonic() + _walltime_seconds(resources.walltime) + 300
    while time.monotonic() < deadline:
        result = _read_json_object(result_path)
        if result is not None:
            return result
        state = subprocess.run(
            ["squeue", "--noheader", "--jobs", job_id, "--format", "%T"],
            capture_output=True,
            text=True,
        )
        if state.returncode != 0 or not state.stdout.strip():
            break
        time.sleep(POLL_INTERVAL_S)
    result = _read_json_object(result_path)
    if result is not None:
        return result
    return {
        "status": "compile_error",
        "errors": [f"Slurm build job {job_id} ended without a build result; inspect {result_path.parent}"],
    }


def run_build_request(request_path: Path) -> dict[str, Any]:
    """Build one candidate bundle inside its isolated Slurm allocation."""
    from . import run_hmmer_cli

    request = json.loads(request_path.read_text(encoding="utf-8"))
    try:
        mods = [ComponentModification(**mod) for mod in request["mods"]]
        cfg = assemble(mods)
        if cfg.errors:
            return {"status": "compile_error", "errors": cfg.errors}
        bundle_dir = request.get("hmmer_bundle_dir")
        bundle_id = request.get("hmmer_bundle_id")
        profiling = bool(request.get("profiling"))
        # A patch-less build is legitimate and expected: it is the stock baseline,
        # and the whole point of the bundle is that its 10 consumers point at ONE
        # compiled artifact instead of each recompiling HMMER. Requiring a C patch
        # here is what previously forced the baseline onto the per-worker-compile
        # path.
        if not bundle_dir or not bundle_id:
            return {
                "status": "compile_error",
                "errors": ["build request must contain a candidate bundle identity"],
            }
        resources = request.get("build_resources") or asdict(build_resources_for())
        manifest = run_hmmer_cli.build_binary_bundle(
            cfg,
            bundle_dir,
            cpus=int(resources["cpus"]),
            build_context=str(request.get("hmmer_build_context") or "host-cluster"),
            expected_bundle_id=str(bundle_id),
            profiling=profiling,
        )
        return {
            "status": "ok",
            "bundle": manifest,
            "bundle_reused": False,
            "excluded_from_cost_s": True,
        }
    except Exception as exc:
        return {
            "status": "compile_error",
            "errors": [f"HMMER bundle build failed: {type(exc).__name__}: {exc}"],
        }


def _read_json_object(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def _wait_for_job(
    *,
    stage: str,
    job_ids: list[str],
    job_dir: Path,
    expected: int,
    resources: SlurmResources,
    max_parallel: int,
    submitted_epoch: float,
    query_timeout_policy: dict[str, Any] | None = None,
) -> EvalResult:
    """Wait for one or more Slurm arrays and aggregate their query results.

    Completion is driven by the per-query result files, never by job state, which is
    why a chunked submission needs no special handling here: whichever jobs finish
    in whatever order, the run is complete once every ``result_<index>.json``
    exists. ``job_ids`` is only a liveness probe -- when none of them is queued or
    running any more, waiting stops and whatever is missing is reported.
    """
    job_label = ",".join(job_ids)
    timeout_s = _walltime_seconds(resources.walltime) + 1800
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        results = _read_results(job_dir / "results", expected)
        if len(results) == expected:
            return _aggregate_results(
                stage, results, job_id=job_label, resources=resources,
                max_parallel=max_parallel, job_dir=job_dir,
                batch_wall_s=max(0.0, time.time() - submitted_epoch),
                query_timeout_policy=query_timeout_policy,
                query_cpus=query_cpus_for(stage),
            )
        state = subprocess.run(
            ["squeue", "--noheader", "--jobs", job_label, "--format", "%T"],
            capture_output=True, text=True,
        )
        if state.returncode != 0 or not state.stdout.strip():
            break
        time.sleep(POLL_INTERVAL_S)
    results = _read_results(job_dir / "results", expected)
    if len(results) == expected:
        return _aggregate_results(
            stage, results, job_id=job_label, resources=resources,
            max_parallel=max_parallel, job_dir=job_dir,
            batch_wall_s=max(0.0, time.time() - submitted_epoch),
            query_timeout_policy=query_timeout_policy,
            query_cpus=query_cpus_for(stage),
        )
    return _error(stage, f"Slurm jobs {job_label} ended with {len(results)}/{expected} results; inspect {job_dir}")


def run_request(request_path: Path, query_index: int) -> EvalResult:
    """Run one immutable evaluation request inside a Slurm allocation."""
    from . import run_hmmer_cli

    request = json.loads(request_path.read_text(encoding="utf-8"))
    stage = str(request["stage"])
    try:
        mods = [ComponentModification(**mod) for mod in request["mods"]]
        cfg = assemble(mods)
        if cfg.errors:
            result = EvalResult(stage=stage, status="compile_error", errors=cfg.errors)
        else:
            query_paths = [Path(path) for path in request["query_paths"]]
            if query_index < 0 or query_index >= len(query_paths):
                return _error(stage, f"array index {query_index} is outside {len(query_paths)} queries")
            task_dir = Path(request["work_root"]) / f"task_{query_index:03d}"
            input_dir = task_dir / "input"
            input_dir.mkdir(parents=True, exist_ok=True)
            query_path = query_paths[query_index]
            staged_query = input_dir / "query.fasta"
            shutil.copy2(query_path, staged_query)
            kwargs = dict(
                data_dir=input_dir,
                # Per-query cores, NOT the job allocation. This single value is the
                # literal HMMER ``--cpu`` for this query. Falling back to the job
                # allocation would hand HMMER ``--cpu 45`` inside a 45-core job, so
                # a request written before this key existed falls back to the
                # stage's configured per-query count instead.
                cpus=int(request.get("query_cpus") or query_cpus_for(stage)),
                stage=stage,
                max_iter=int(request["max_iter"]),
                astral_db=request["astral_db"],
                uniref_db=request["uniref_db"],
                scop_txt=request["scop_txt"],
            )
            # One production backend for every component and stage keeps smoke
            # and full semantics comparable. The CLI streams both databases.
            bundle_kwargs: dict[str, Any] = {}
            if request.get("hmmer_bundle_dir"):
                bundle_kwargs = {
                    "hmmer_bundle_dir": request["hmmer_bundle_dir"],
                    "hmmer_bundle_id": request.get("hmmer_bundle_id"),
                }
            result = run_hmmer_cli.run_all_queries(
                cfg,
                **kwargs,
                work_root=task_dir / "work",
                keep_work=False,
                query_timeout=_query_timeout_for(request, query_index),
                profiling=bool(request.get("profiling")),
                **bundle_kwargs,
            )
    except run_hmmer_cli.QueryTimeoutError as exc:
        result = EvalResult(
            stage=stage,
            status="timeout",
            cost_s=round(float(exc.wall_s), 4),
            n_seqs=1,
            errors=[str(exc)],
            meta={
                "timeout": {
                    "phase": exc.label,
                    "time_limit_s": exc.timeout_s,
                    "wall_s": exc.wall_s,
                    "stdout_tail": exc.stdout_tail,
                    "stderr_tail": exc.stderr_tail,
                }
            },
        )
    except Exception as exc:
        result = _error(stage, f"Slurm worker failed: {type(exc).__name__}: {exc}")
    query_paths = request.get("query_paths") or []
    result.meta.update({
        "execution": "slurm",
        "resources": request["resources"],
        "query_index": query_index,
        "query_path": str(query_paths[query_index]) if 0 <= query_index < len(query_paths) else None,
        "query_time_limit_s": _query_timeout_for(request, query_index),
        "hmmer_bundle_id": request.get("hmmer_bundle_id"),
    })
    return result


def _snapshot_queries(data_dir: Path, destination: Path) -> list[Path]:
    """Freeze ordered query inputs before the job array enters the queue."""
    from .run_jack import query_fastas

    source_paths = [Path(path) for path in query_fastas(data_dir)]
    destination.mkdir(parents=True, exist_ok=False)
    copied: list[Path] = []
    for index, source in enumerate(source_paths):
        target = destination / f"query_{index:03d}.fasta"
        shutil.copy2(source, target)
        copied.append(target.resolve())
    return copied


def _read_results(result_dir: Path, expected: int) -> list[EvalResult]:
    results: list[EvalResult] = []
    for index in range(expected):
        path = result_dir / f"result_{index:03d}.json"
        if not path.is_file():
            continue
        try:
            results.append(EvalResult.model_validate_json(path.read_text(encoding="utf-8")))
        except Exception:
            continue
    return results


def _find_resumable_job(
    *,
    round_path: Path,
    stage: str,
    mods: list[dict[str, Any]],
    max_iter: int,
    source_paths: dict[str, str],
    query_timeout_policy: dict[str, Any] | None = None,
    optimization_objective: dict[str, Any] | None = None,
    bundle_id: str | None = None,
    runtime: str = "host",
    image_ref: str | None = None,
) -> tuple[Path, dict[str, Any], dict[str, Any]] | None:
    """Find the newest identical active job or completed successful job.

    ``submission.json`` is written immediately after sbatch succeeds. This lets
    a restarted local controller attach to the same immutable request instead
    of submitting all queries again.
    """
    slurm_root = round_path / "slurm"
    if not slurm_root.is_dir():
        return None
    candidates = sorted(slurm_root.glob(f"{stage}_*"), reverse=True)
    for job_dir in candidates:
        request_path = job_dir / "request.json"
        submission_path = job_dir / "submission.json"
        if not request_path.is_file() or not submission_path.is_file():
            continue
        try:
            request = json.loads(request_path.read_text(encoding="utf-8"))
            submission = json.loads(submission_path.read_text(encoding="utf-8"))
            expected = int(submission["query_count"])
            # Older submissions recorded one array job; chunked ones record one job
            # per chunk. Accept both so an interrupted run still resumes.
            job_ids = [str(item) for item in (submission.get("job_ids") or [])]
            if not job_ids:
                job_ids = [str(submission["job_id"])]
        except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
            continue
        try:
            matches = (
                request.get("stage") == stage
                and request.get("mods") == mods
                and int(request.get("max_iter", -1)) == max_iter
                and request.get("source_paths") == source_paths
                and request.get("query_timeout_policy") == query_timeout_policy
                and request.get("optimization_objective") == (
                    optimization_objective or objective_metadata()
                )
                and str(request.get("runtime") or "host") == runtime
                and request.get("image_ref") == image_ref
                and str(submission.get("runtime") or "host") == runtime
                and submission.get("image_ref") == image_ref
                and (bundle_id is None or request.get("hmmer_bundle_id") == bundle_id)
                and len(request.get("query_paths") or []) == expected
                # Re-attach only to a job measured under the SAME core/concurrency
                # shape. Without this, changing SEQ_ALIGN_SLURM_*_QUERY_CPUS (or -CPUS,
                # or -QUERIES_PER_JOB) and restarting would silently aggregate
                # partially-remaining results from the old shape into a record that
                # claims the new one.
                and int(request.get("query_cpus") or 0) == query_cpus_for(stage)
                and int(request.get("queries_per_job") or 0) == queries_per_job(stage)
                and dict(request.get("resources") or {}) == asdict(resources_for(stage))
            )
        except (TypeError, ValueError):
            matches = False
        if not matches:
            continue

        results = _read_results(job_dir / "results", expected)
        if len(results) == expected and all(result.status in {"ok", "timeout"} for result in results):
            return job_dir, request, submission

        state = subprocess.run(
            ["squeue", "--noheader", "--jobs", ",".join(job_ids), "--format", "%T"],
            capture_output=True,
            text=True,
        )
        if state.returncode == 0 and state.stdout.strip():
            return job_dir, request, submission
        # A finished incomplete/failed request is not reusable; submit fresh chunks
        # while retaining this job directory as immutable evidence.
        return None
    return None


def _find_resumable_build(
    *,
    round_path: Path,
    stage: str,
    mods: list[dict[str, Any]],
    max_iter: int,
    source_paths: dict[str, str],
    bundle_id: str | None,
    query_timeout_policy: dict[str, Any] | None = None,
    optimization_objective: dict[str, Any] | None = None,
    runtime: str = "host",
    image_ref: str | None = None,
) -> tuple[Path, dict[str, Any], dict[str, Any]] | None:
    """Find an identical build that has not submitted its query array yet."""
    if bundle_id is None:
        return None
    slurm_root = round_path / "slurm"
    if not slurm_root.is_dir():
        return None
    for job_dir in sorted(slurm_root.glob(f"{stage}_*"), reverse=True):
        request_path = job_dir / "request.json"
        build_submission_path = job_dir / "build_submission.json"
        if (job_dir / "submission.json").exists():
            continue
        request = _read_json_object(request_path)
        build_submission = _read_json_object(build_submission_path)
        if request is None or build_submission is None:
            continue
        try:
            matches = (
                request.get("stage") == stage
                and request.get("mods") == mods
                and int(request.get("max_iter", -1)) == max_iter
                and request.get("source_paths") == source_paths
                and request.get("query_timeout_policy") == query_timeout_policy
                and request.get("optimization_objective") == (
                    optimization_objective or objective_metadata()
                )
                and str(request.get("runtime") or "host") == runtime
                and request.get("image_ref") == image_ref
                and str(build_submission.get("runtime") or "host") == runtime
                and build_submission.get("image_ref") == image_ref
                and request.get("hmmer_bundle_id") == bundle_id
                and build_submission.get("hmmer_bundle_id") == bundle_id
                and bool(request.get("query_paths"))
            )
            job_id = str(build_submission["job_id"])
            SlurmResources(**build_submission["resources"])
        except (TypeError, ValueError, KeyError):
            continue
        if not matches:
            continue
        if _read_json_object(job_dir / "build_result.json") is not None:
            return job_dir, request, build_submission
        state = subprocess.run(
            ["squeue", "--noheader", "--jobs", job_id, "--format", "%T"],
            capture_output=True,
            text=True,
        )
        if state.returncode == 0 and state.stdout.strip():
            return job_dir, request, build_submission
        return None
    return None


def _aggregate_results(
    stage: str,
    results: list[EvalResult],
    *,
    job_id: str,
    resources: SlurmResources,
    max_parallel: int,
    job_dir: Path,
    batch_wall_s: float | None = None,
    query_timeout_policy: dict[str, Any] | None = None,
    query_cpus: int | None = None,
) -> EvalResult:
    statuses = [result.status for result in results]
    status = "ok" if all(value == "ok" for value in statuses) else "run_error"
    if statuses and all(value in {"ok", "timeout"} for value in statuses) and "timeout" in statuses:
        status = "timeout"
    if all(value == "skipped" for value in statuses):
        status = "skipped"
    scop_counts = {
        "tp": 0,
        "fp": 0,
        "ignored_same_fold_different_superfamily": 0,
        "unclassified": 0,
    }
    query_costs: list[float] = []
    query_results: list[dict[str, Any]] = []
    builds: list[dict[str, Any]] = []
    bundles: list[dict[str, Any]] = []
    references = {
        int(item["query_index"]): item
        for item in (query_timeout_policy or {}).get("queries", [])
        if isinstance(item, dict) and item.get("query_index") is not None
    }
    for fallback_index, result in enumerate(results):
        query_costs.append(float(result.cost_s))
        query_index = int(result.meta.get("query_index", fallback_index))
        reference = references.get(query_index, {})
        reference_cost = float(reference.get("reference_cost_s") or 0.0)
        profile_row = _profile_query_row(result)
        query_results.append({
            "query_index": query_index,
            "query_id": _query_id_from_result(result) or reference.get("query_id") or f"query_{query_index:03d}",
            "status": result.status,
            "tp": result.tp,
            "fp": result.fp,
            "tp_fp_ratio": result.tp_fp_ratio,
            "cost_s": result.cost_s,
            "reference": {
                "tp": reference.get("tp"),
                "fp": reference.get("fp"),
                "tp_fp_ratio": reference.get("tp_fp_ratio"),
                "cost_s": reference_cost or None,
            },
            "cost_ratio": round(result.cost_s / reference_cost, 6) if reference_cost else None,
            "time_limit_s": reference.get("time_limit_s") or result.meta.get("query_time_limit_s"),
            "iterations": result.iterations,
            "nhits": result.meta.get("nhits", profile_row.get("nhits")),
            "error": result.errors[0] if result.errors else None,
        })
        scoring = result.meta.get("scop_scoring") if isinstance(result.meta, dict) else None
        if isinstance(scoring, dict):
            for key in scop_counts:
                scop_counts[key] += int(scoring.get(key, 0) or 0)
        build = result.meta.get("build") if isinstance(result.meta, dict) else None
        if isinstance(build, dict) and build:
            builds.append(build)
        bundle = result.meta.get("hmmer_bundle") if isinstance(result.meta, dict) else None
        if isinstance(bundle, dict) and bundle:
            bundles.append(bundle)
    from . import run_hmmer_cli

    query_profiles = [
        profile
        for profile in (result.meta.get("profile") for result in results if isinstance(result.meta, dict))
        if isinstance(profile, dict)
    ]
    build_result = _read_json_object(job_dir / "build_result.json") or {}
    build_submission = _read_json_object(job_dir / "build_submission.json") or {}
    built_bundle = build_result.get("bundle")
    if not bundles and isinstance(built_bundle, dict) and built_bundle:
        bundles.append(built_bundle)
    meta: dict[str, Any] = {
        "execution": "slurm_array",
        "slurm_job_id": job_id,
        "slurm_resources_per_query": asdict(resources),
        # What the query actually received. ``query_cpus`` is the literal HMMER
        # ``--cpu`` the worker dispatched; ``job_cpus`` is the allocation it ran
        # inside. They are different numbers on purpose and both are recorded, so
        # a verdict can be audited without re-reading the sbatch command.
        "query_cpus_per_query": int(query_cpus if query_cpus is not None else resources.cpus),
        "hmmer_worker_threads_per_query": int(query_cpus if query_cpus is not None else resources.cpus),
        "queries_per_job": queries_per_job(stage),
        "job_cpus": int(resources.cpus),
        "query_count": len(results),
        "max_parallel_queries": max_parallel,
        "job_dir": str(job_dir),
        "batch_wall_s": round(float(batch_wall_s), 4) if batch_wall_s is not None else None,
        "build_job_id": build_submission.get("job_id"),
        "bundle_reused": build_result.get("bundle_reused"),
        "build_wall_s": (
            built_bundle.get("build_wall_s")
            if isinstance(built_bundle, dict)
            else None
        ),
        "cost_s_excludes_build": True,
        "query_cost_s": query_costs,
        # Each worker profiles only its own query; the local path assembles the
        # session profile inside run_all_queries, so Slurm has to assemble it here or
        # an instrumented pass can never be read back as stage shares.
        "profile": run_hmmer_cli._aggregate_profiles(
            query_profiles,
            stage=stage,
            status=status,
            cpus=int(query_cpus if query_cpus is not None else resources.cpus),
            max_iter=max((int(profile.get("max_iter") or 0) for profile in query_profiles), default=0),
            query_count=len(results),
            cost_s=round(sum(result.cost_s for result in results), 4),
        ),
        "query_results": sorted(query_results, key=lambda item: item["query_index"]),
        "query_timeout_policy": query_timeout_policy,
        "scop_scoring": {
            "policy": "same_superfamily_positive_different_fold_negative",
            **scop_counts,
        },
    }
    if builds:
        meta["build"] = builds[0]
        meta["runtime_effective"] = all(build.get("runtime_effective") is not False for build in builds)
        meta["build_consistent_across_queries"] = all(build == builds[0] for build in builds[1:])
    elif isinstance(built_bundle, dict) and isinstance(built_bundle.get("build"), dict):
        meta["build"] = built_bundle["build"]
        meta["runtime_effective"] = built_bundle.get("runtime_effective")
    if bundles:
        meta["hmmer_bundle"] = bundles[0]
        meta["bundle_consistent_across_queries"] = all(
            bundle.get("bundle_id") == bundles[0].get("bundle_id")
            and bundle.get("source_tree_sha256") == bundles[0].get("source_tree_sha256")
            and bundle.get("binaries") == bundles[0].get("binaries")
            for bundle in bundles[1:]
        )
    aggregate = EvalResult(
        stage=stage,
        status=status,
        tp=sum(result.tp for result in results),
        fp=sum(result.fp for result in results),
        cost_s=round(sum(result.cost_s for result in results), 4),
        n_seqs=sum(result.n_seqs for result in results),
        iterations=max((result.iterations for result in results), default=0),
        errors=[error for result in results for error in result.errors],
        meta=meta,
    )
    aggregate.ratio()
    return aggregate


def _error(stage: str, message: str) -> EvalResult:
    return EvalResult(stage=stage, status="run_error", errors=[message])


def _query_timeout_for(request: dict[str, Any], query_index: int) -> float | None:
    policy = request.get("query_timeout_policy")
    if not isinstance(policy, dict) or not policy.get("enabled"):
        return None
    for item in policy.get("queries") or []:
        if isinstance(item, dict) and item.get("query_index") == query_index:
            value = item.get("time_limit_s")
            return float(value) if value is not None else None
    return None


def _query_id_from_result(result: EvalResult) -> str | None:
    direct = result.meta.get("query_id") if isinstance(result.meta, dict) else None
    if direct:
        return str(direct)
    row = _profile_query_row(result)
    value = row.get("query_id")
    return str(value) if value else None


def _profile_query_row(result: EvalResult) -> dict[str, Any]:
    profile = result.meta.get("profile") if isinstance(result.meta, dict) else None
    if not isinstance(profile, dict):
        return {}
    rows = profile.get("queries") or profile.get("top_slowest_queries") or []
    if isinstance(rows, list) and rows and isinstance(rows[0], dict):
        return rows[0]
    return profile if profile.get("query_id") else {}


def _reference_query_rows(result_path: Path) -> list[dict[str, Any]]:
    try:
        payload = json.loads(result_path.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return []
    # Canonical c_patch_loop EvaluationRecord files wrap the actual EvalResult;
    # legacy round artifacts store the EvalResult at the document root.
    nested_result = payload.get("eval_result")
    if isinstance(nested_result, dict):
        payload = nested_result
    meta = payload.get("meta") or {}
    rows = meta.get("query_results")
    if isinstance(rows, list) and rows:
        return [dict(row) for row in rows if isinstance(row, dict)]

    job_dir = meta.get("job_dir")
    if job_dir:
        result_dir = Path(str(job_dir)) / "results"
        shard_rows: list[dict[str, Any]] = []
        for index, path in enumerate(sorted(result_dir.glob("result_*.json"))):
            try:
                shard = EvalResult.model_validate_json(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            shard_rows.append({
                "query_index": int(shard.meta.get("query_index", index)),
                "query_id": _query_id_from_result(shard),
                "status": shard.status,
                "tp": shard.tp,
                "fp": shard.fp,
                "tp_fp_ratio": shard.tp_fp_ratio,
                "cost_s": shard.cost_s,
            })
        if shard_rows:
            return shard_rows

    costs = meta.get("query_cost_s") or []
    return [
        {"query_index": index, "cost_s": float(cost), "status": "ok"}
        for index, cost in enumerate(costs)
    ]


def _full_query_timeout_policy(
    *,
    stage: str,
    run_dir: Path,
    round_idx: int,
    data_dir: Path,
    reference_result_path: Path | None = None,
) -> dict[str, Any] | None:
    """Build immutable per-query Full limits from the accepted reference."""
    if stage != "full" or round_idx == 0:
        return None
    multiplier = float(os.getenv("SEQ_ALIGN_FULL_QUERY_TIMEOUT_MULTIPLIER", "1.2"))
    if multiplier <= 1.0:
        raise ValueError("SEQ_ALIGN_FULL_QUERY_TIMEOUT_MULTIPLIER must be > 1.0")

    accepted = run_dir / "accepted_eval.json"
    baseline = run_dir / "round_000" / "full_eval.json"
    explicit_reference = reference_result_path is not None
    reference_path = reference_result_path or (accepted if accepted.is_file() else baseline)
    rows = _reference_query_rows(reference_path)
    if not rows:
        if explicit_reference:
            raise ValueError(
                "explicit query timeout reference has no usable query results: "
                f"{reference_path}"
            )
        return None

    from .run_jack import query_fastas
    from .run_hmmer_cli import _query_id

    query_paths = [Path(path).resolve() for path in query_fastas(data_dir)]
    by_index = {int(row.get("query_index", index)): row for index, row in enumerate(rows)}
    policy_rows: list[dict[str, Any]] = []
    for index, query_path in enumerate(query_paths):
        reference = by_index.get(index)
        if not reference:
            if explicit_reference:
                raise ValueError(
                    "explicit query timeout reference is missing query index "
                    f"{index}: {reference_path}"
                )
            return None
        reference_cost = float(reference.get("cost_s") or 0.0)
        if reference_cost <= 0:
            if explicit_reference:
                raise ValueError(
                    "explicit query timeout reference has non-positive cost_s "
                    f"for query index {index}: {reference_path}"
                )
            return None
        query_id = _query_id(str(query_path))
        reference_id = reference.get("query_id")
        if reference_id and str(reference_id) != query_id:
            raise ValueError(
                f"reference query order mismatch at index {index}: "
                f"current={query_id} reference={reference_id}"
            )
        policy_rows.append({
            "query_index": index,
            "query_id": query_id,
            "reference_status": reference.get("status"),
            "reference_cost_s": round(reference_cost, 4),
            "time_limit_s": round(reference_cost * multiplier, 4),
            "tp": reference.get("tp"),
            "fp": reference.get("fp"),
            "tp_fp_ratio": reference.get("tp_fp_ratio"),
        })
    return {
        "enabled": True,
        "scope": "full_query_total_wall_clock",
        "multiplier": multiplier,
        "reference_result_path": str(reference_path),
        "queries": policy_rows,
    }


def _walltime_seconds(value: str) -> int:
    days = 0
    if "-" in value:
        day_text, value = value.split("-", 1)
        days = int(day_text)
    parts = [int(part) for part in value.split(":")]
    if len(parts) == 2:
        hours, minutes, seconds = 0, *parts
    else:
        hours, minutes, seconds = parts
    return days * 86400 + hours * 3600 + minutes * 60 + seconds


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True, type=Path)
    parser.add_argument("--mode", choices=("query", "build"), default="query")
    args = parser.parse_args()
    request = json.loads(args.request.read_text(encoding="utf-8"))
    if args.mode == "build":
        result_path = Path(request["build_result_path"])
        result = run_build_request(args.request)
        _write_json_atomic(result_path, result)
        return 0 if result.get("status") == "ok" else 1
    # Global query index within the whole run. A chunked submission passes its
    # offset explicitly and the worker adds its own local array id; the explicit
    # override keeps a future one-job-per-query submission working unchanged.
    query_index = int(
        os.getenv("SEQALIGN_QUERY_INDEX")
        or (
            int(os.getenv("SEQALIGN_QUERY_OFFSET", "0"))
            + int(os.getenv("SLURM_ARRAY_TASK_ID", "0"))
        )
    )
    result_path = Path(request["result_dir"]) / f"result_{query_index:03d}.json"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result = run_request(args.request, query_index)
    tmp_path = result_path.with_suffix(".tmp")
    tmp_path.write_text(result.model_dump_json(indent=2), encoding="utf-8")
    tmp_path.replace(result_path)
    return 0 if result.status in {"ok", "timeout"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
