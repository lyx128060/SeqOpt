"""Content-addressed storage helpers for deterministic C-patch loop records."""

from __future__ import annotations

import hashlib
import json
import os
import re
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, TypeVar

from pydantic import BaseModel

from ..pending_mods import normalize_run_dir
from .models import (
    AlgorithmPlanRecord,
    EvaluationAttemptRecord,
    EvaluationRecord,
    ImplementationTrialRecord,
    InvocationRecord,
    PatchPlanRecord,
    QueryCompletionRecord,
    RunMigrationRecord,
    SlotEventRecord,
    PatchsetRecord,
)

_EPHEMERAL_HASH_FIELDS = {"created_at"}
_ModelT = TypeVar("_ModelT", bound=BaseModel)

_SENSITIVE_FIELD_RE = re.compile(
    r"(?:^|_)(?:api_key|access_token|refresh_token|authorization|password|passwd|credential|secret|token)(?:$|_)",
    flags=re.IGNORECASE,
)
_QUOTED_SECRET_RE = re.compile(
    r'''(?ix)
    (?P<prefix>["']?(?:api[_-]?key|access[_-]?token|refresh[_-]?token|authorization|password|passwd|credential|secret|token)["']?\s*[:=]\s*)
    (?P<quote>["'])
    .*?
    (?P=quote)
    ''',
)
_UNQUOTED_SECRET_RE = re.compile(
    r"(?i)(?P<prefix>\b(?:api[_-]?key|access[_-]?token|refresh[_-]?token|password|passwd|credential|secret|token)\s*[:=]\s*)(?P<value>[^\s,;}\]]+)",
)
_AUTHORIZATION_RE = re.compile(
    r"(?i)(?P<prefix>authorization\s*:\s*(?:bearer|basic)\s+)(?P<value>[^\s,;}\]]+)",
)
_OPENAI_KEY_RE = re.compile(r"\bsk-[A-Za-z0-9_-]+")


def utc_now() -> str:
    """Return an ISO UTC timestamp for informational metadata."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def canonical_data(value: Any, *, exclude_fields: set[str] | None = None) -> Any:
    """Convert a model/dict/list into a deterministic JSON-compatible object."""
    exclude = exclude_fields or set()
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json")
    if isinstance(value, dict):
        return {
            str(k): canonical_data(v, exclude_fields=exclude)
            for k, v in sorted(value.items(), key=lambda item: str(item[0]))
            if k not in exclude
        }
    if isinstance(value, list):
        return [canonical_data(v, exclude_fields=exclude) for v in value]
    if isinstance(value, tuple):
        return [canonical_data(v, exclude_fields=exclude) for v in value]
    return value


def canonical_json(value: Any, *, exclude_fields: set[str] | None = None) -> str:
    """Serialize to deterministic JSON."""
    return json.dumps(
        canonical_data(value, exclude_fields=exclude_fields),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sanitize_sensitive_data(value: Any, *, replacement: str = "[REDACTED]") -> Any:
    """Recursively redact credentials before diagnostic data is persisted or prompted."""
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json")
    if isinstance(value, dict):
        return {
            str(key): (
                replacement
                if _SENSITIVE_FIELD_RE.search(str(key))
                else sanitize_sensitive_data(item, replacement=replacement)
            )
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [sanitize_sensitive_data(item, replacement=replacement) for item in value]
    if isinstance(value, tuple):
        return [sanitize_sensitive_data(item, replacement=replacement) for item in value]
    if not isinstance(value, str):
        return value

    text = _AUTHORIZATION_RE.sub(lambda match: match.group("prefix") + replacement, value)
    text = _QUOTED_SECRET_RE.sub(
        lambda match: match.group("prefix") + match.group("quote") + replacement + match.group("quote"),
        text,
    )
    text = _UNQUOTED_SECRET_RE.sub(lambda match: match.group("prefix") + replacement, text)
    return _OPENAI_KEY_RE.sub(replacement, text)


def record_hash(value: Any, *, id_field: str | None = None) -> str:
    """Hash a record, excluding its own ID and informational timestamps."""
    exclude = set(_EPHEMERAL_HASH_FIELDS)
    if id_field:
        exclude.add(id_field)
    return sha256_text(canonical_json(value, exclude_fields=exclude))


def hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, data: Any, *, overwrite_same: bool = True) -> Path:
    """Write JSON atomically-ish; refuse overwriting different immutable content.

    Record IDs intentionally ignore informational timestamps. If a semantically
    identical record is regenerated with a different ``created_at``, keep the
    original file instead of treating the timestamp as a content conflict.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(canonical_data(data), indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    if path.exists():
        old = path.read_text(encoding="utf-8")
        old_data = json.loads(old)
        old_stable = canonical_json(old_data, exclude_fields=set(_EPHEMERAL_HASH_FIELDS))
        new_stable = canonical_json(data, exclude_fields=set(_EPHEMERAL_HASH_FIELDS))
        if old_stable != new_stable:
            raise FileExistsError(f"refusing to overwrite different immutable record: {path}")
        if overwrite_same:
            return path
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)
    return path


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


class LoopStore:
    """Run-local store for context packs, patchsets, evals, and experience."""

    def __init__(self, run_dir: str | Path):
        self.run_dir = normalize_run_dir(run_dir)
        self.root = self.run_dir / "c_patch_loop"
        self.context_packs_dir = self.root / "context_packs"
        self.algorithm_plans_dir = self.root / "algorithm_plans"
        self.patch_plans_dir = self.root / "patch_plans"
        self.implementation_trials_dir = self.root / "implementation_trials"
        self.patchsets_dir = self.root / "patchsets"
        self.evaluations_dir = self.root / "evaluations"
        self.evaluation_attempts_dir = self.root / "evaluation_attempts"
        self.query_completions_dir = self.root / "query_completions"
        self.invocations_dir = self.root / "invocations"
        self.invocation_summaries_dir = self.root / "invocation_summaries"
        self.slot_events_dir = self.root / "slot_events"
        self.model_traces_dir = self.root / "model_traces"
        self.migrations_dir = self.root / "migrations"
        self.experience_dir = self.root / "experience"
        self.indexes_dir = self.root / "indexes"
        self.work_dir = self.root / "work"
        for path in (
            self.context_packs_dir,
            self.algorithm_plans_dir,
            self.patch_plans_dir,
            self.implementation_trials_dir,
            self.patchsets_dir,
            self.evaluations_dir,
            self.evaluation_attempts_dir,
            self.query_completions_dir,
            self.invocations_dir,
            self.invocation_summaries_dir,
            self.slot_events_dir,
            self.model_traces_dir,
            self.migrations_dir,
            self.experience_dir,
            self.indexes_dir,
            self.work_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)

    def put_context_pack(self, record: BaseModel) -> Path:
        rid = getattr(record, "context_pack_id")
        if not rid:
            raise ValueError("context pack must have context_pack_id before storage")
        return write_json(self.context_packs_dir / f"{rid}.json", record)

    def put_patch_plan(self, record: BaseModel) -> Path:
        rid = getattr(record, "patch_plan_id")
        if not rid:
            raise ValueError("patch plan must have patch_plan_id before storage")
        return write_json(self.patch_plans_dir / f"{rid}.json", record)

    def put_algorithm_plan(self, record: BaseModel) -> Path:
        return self._put_record(self.algorithm_plans_dir, record, "algorithm_plan_id")

    def put_implementation_trial(self, record: BaseModel) -> Path:
        return self._put_record(self.implementation_trials_dir, record, "implementation_trial_id")

    def put_patchset(self, record: BaseModel) -> Path:
        rid = getattr(record, "patchset_id")
        if not rid:
            raise ValueError("patchset must have patchset_id before storage")
        return write_json(self.patchsets_dir / f"{rid}.json", record)

    def put_evaluation(self, record: BaseModel) -> Path:
        rid = getattr(record, "eval_id")
        if not rid:
            raise ValueError("evaluation must have eval_id before storage")
        return write_json(self.evaluations_dir / f"{rid}.json", record)

    def put_evaluation_attempt(self, record: BaseModel) -> Path:
        return self._put_record(self.evaluation_attempts_dir, record, "attempt_id")

    def put_query_completion(self, record: BaseModel) -> Path:
        return self._put_record(self.query_completions_dir, record, "query_completion_id")

    def put_invocation(self, record: BaseModel) -> Path:
        return self._put_record(self.invocations_dir, record, "invocation_id")

    def put_invocation_summary(self, record: BaseModel) -> Path:
        return self._put_record(self.invocation_summaries_dir, record, "invocation_summary_id")

    def put_slot_event(self, record: BaseModel) -> Path:
        return self._put_record(self.slot_events_dir, record, "slot_event_id")

    def put_model_trace(self, trace: dict[str, Any]) -> Path:
        """Persist one sanitized, content-addressed model-attempt trace."""
        data = sanitize_sensitive_data(trace)
        trace_id = str(data.get("trace_id") or record_hash(data, id_field="trace_id"))
        data["trace_id"] = trace_id
        return write_json(self.model_traces_dir / f"{trace_id}.json", data)

    def put_migration(self, record: BaseModel) -> Path:
        return self._put_record(self.migrations_dir, record, "migration_id")

    @staticmethod
    def _put_record(directory: Path, record: BaseModel, id_field: str) -> Path:
        rid = getattr(record, id_field, None)
        if not rid:
            raise ValueError(f"record must have {id_field} before storage")
        return write_json(directory / f"{rid}.json", record)

    def append_experience(self, record: BaseModel) -> Path:
        self.experience_dir.mkdir(parents=True, exist_ok=True)
        path = self.experience_dir / "experience.jsonl"
        new_data = canonical_data(record)
        rid = new_data.get("experience_id") if isinstance(new_data, dict) else None
        if rid:
            for old in self.iter_experience():
                if old.get("experience_id") != rid:
                    continue
                if canonical_json(old, exclude_fields=set(_EPHEMERAL_HASH_FIELDS)) != canonical_json(
                    new_data, exclude_fields=set(_EPHEMERAL_HASH_FIELDS)
                ):
                    raise FileExistsError(f"conflicting experience_id: {rid}")
                return path
        text = json.dumps(new_data, ensure_ascii=False, sort_keys=True) + "\n"
        with path.open("a", encoding="utf-8") as handle:
            handle.write(text)
        return path

    def write_index(self, name: str, data: Any) -> Path:
        if not name.endswith(".json"):
            name += ".json"
        path = self.indexes_dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        text = json.dumps(canonical_data(data), indent=2, ensure_ascii=False, sort_keys=True) + "\n"
        path.write_text(text, encoding="utf-8")
        return path

    def read_index(self, name: str) -> Any | None:
        if not name.endswith(".json"):
            name += ".json"
        path = self.indexes_dir / name
        if not path.exists():
            return None
        return load_json(path)

    def iter_patch_plans(self) -> list[dict[str, Any]]:
        return [load_json(path) for path in sorted(self.patch_plans_dir.glob("*.json"))]

    def iter_algorithm_plans(self) -> list[dict[str, Any]]:
        return self._iter_json(self.algorithm_plans_dir)

    def iter_implementation_trials(self) -> list[dict[str, Any]]:
        return self._iter_json(self.implementation_trials_dir)

    def iter_patchsets(self) -> list[dict[str, Any]]:
        return [load_json(path) for path in sorted(self.patchsets_dir.glob("*.json"))]

    def iter_evaluations(self) -> list[dict[str, Any]]:
        return [load_json(path) for path in sorted(self.evaluations_dir.glob("*.json"))]

    def iter_evaluation_attempts(self) -> list[dict[str, Any]]:
        return self._iter_json(self.evaluation_attempts_dir)

    def iter_query_completions(self) -> list[dict[str, Any]]:
        return self._iter_json(self.query_completions_dir)

    def iter_invocations(self) -> list[dict[str, Any]]:
        return self._iter_json(self.invocations_dir)

    def iter_invocation_summaries(self) -> list[dict[str, Any]]:
        return self._iter_json(self.invocation_summaries_dir)

    def iter_slot_events(self) -> list[dict[str, Any]]:
        return self._iter_json(self.slot_events_dir)

    def iter_model_traces(self) -> list[dict[str, Any]]:
        return self._iter_json(self.model_traces_dir)

    def iter_migrations(self) -> list[dict[str, Any]]:
        return self._iter_json(self.migrations_dir)

    @staticmethod
    def _iter_json(directory: Path) -> list[dict[str, Any]]:
        return [load_json(path) for path in sorted(directory.glob("*.json"))]

    def iter_experience(self) -> list[dict[str, Any]]:
        path = self.experience_dir / "experience.jsonl"
        if not path.exists():
            return []
        records: list[dict[str, Any]] = []
        by_id: dict[str, dict[str, Any]] = {}
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(value, dict):
                continue
            rid = value.get("experience_id")
            if not rid:
                records.append(value)
                continue
            previous = by_id.get(str(rid))
            if previous is None:
                by_id[str(rid)] = value
                records.append(value)
                continue
            if canonical_json(previous, exclude_fields=set(_EPHEMERAL_HASH_FIELDS)) != canonical_json(
                value, exclude_fields=set(_EPHEMERAL_HASH_FIELDS)
            ):
                raise FileExistsError(f"conflicting experience_id: {rid}")
        return records

    def get_patchset(self, record_id: str) -> PatchsetRecord:
        return self._get_record(self.patchsets_dir, record_id, "patchset_id", PatchsetRecord)

    def get_algorithm_plan(self, record_id: str) -> AlgorithmPlanRecord:
        return self._get_record(self.algorithm_plans_dir, record_id, "algorithm_plan_id", AlgorithmPlanRecord)

    def get_patch_plan(self, record_id: str) -> PatchPlanRecord:
        return self._get_record(self.patch_plans_dir, record_id, "patch_plan_id", PatchPlanRecord)

    def get_implementation_trial(self, record_id: str) -> ImplementationTrialRecord:
        return self._get_record(
            self.implementation_trials_dir,
            record_id,
            "implementation_trial_id",
            ImplementationTrialRecord,
        )

    def get_evaluation(self, record_id: str) -> EvaluationRecord:
        return self._get_record(self.evaluations_dir, record_id, "eval_id", EvaluationRecord)

    def get_evaluation_attempt(self, record_id: str) -> EvaluationAttemptRecord:
        return self._get_record(self.evaluation_attempts_dir, record_id, "attempt_id", EvaluationAttemptRecord)

    def get_query_completion(self, record_id: str) -> QueryCompletionRecord:
        return self._get_record(self.query_completions_dir, record_id, "query_completion_id", QueryCompletionRecord)

    def get_invocation(self, record_id: str) -> InvocationRecord:
        return self._get_record(self.invocations_dir, record_id, "invocation_id", InvocationRecord, verify_hash=False)

    @staticmethod
    def _get_record(
        directory: Path,
        record_id: str,
        id_field: str,
        model: type[_ModelT],
        *,
        verify_hash: bool = True,
    ) -> _ModelT:
        data = load_json(directory / f"{record_id}.json")
        if data.get(id_field) != record_id:
            raise ValueError(f"embedded ID mismatch for {id_field}: expected {record_id}")
        if verify_hash and len(record_id) == 64 and all(ch in "0123456789abcdef" for ch in record_id.lower()):
            if record_hash(data, id_field=id_field) != record_id:
                raise ValueError(f"content hash mismatch for {id_field}: {record_id}")
        return model.model_validate(data)

    def append_slot_event(self, previous: SlotEventRecord | None = None, **fields: Any) -> SlotEventRecord:
        if previous is not None:
            fields.setdefault("parent_slot_event_id", previous.slot_event_id)
        record = SlotEventRecord(**fields)
        record.slot_event_id = record_hash(record, id_field="slot_event_id")
        self.put_slot_event(record)
        return record

    def slot_heads(self, *, invocation_id: str | None = None) -> dict[str, SlotEventRecord]:
        events = [SlotEventRecord.model_validate(item) for item in self.iter_slot_events()]
        if invocation_id is not None:
            events = [item for item in events if item.invocation_id == invocation_id]
        parents = {item.parent_slot_event_id for item in events if item.parent_slot_event_id}
        heads = [item for item in events if item.slot_event_id not in parents]
        heads.sort(key=lambda item: (item.invocation_id, item.slot_id, item.created_at or "", item.slot_event_id or ""))
        if invocation_id is not None:
            return {item.slot_id: item for item in heads}
        return {f"{item.invocation_id}:{item.slot_id}": item for item in heads}

    def canonical_high_water(self) -> str:
        """Hash canonical controller state, excluding diagnostic summaries."""
        payload = {
            "invocations": self.iter_invocations(),
            "slot_heads": {
                key: value.model_dump(mode="json")
                for key, value in self.slot_heads().items()
            },
            "algorithm_plans": self.iter_algorithm_plans(),
            "patchsets": self.iter_patchsets(),
            "evaluations": self.iter_evaluations(),
        }
        return sha256_text(canonical_json(payload, exclude_fields=set(_EPHEMERAL_HASH_FIELDS)))

    @contextmanager
    def single_writer(self) -> Iterator[None]:
        """Hold a non-blocking advisory run-local controller lock."""
        import fcntl

        lock_path = self.root / ".controller.lock"
        fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o664)
        try:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RuntimeError("another controller owns this run") from exc
            yield
        finally:
            try:
                fcntl.flock(fd, fcntl.LOCK_UN)
            finally:
                os.close(fd)

    def import_verified_evidence(
        self,
        source: "LoopStore",
        *,
        source_run_id: str,
        target_run_id: str,
        migration_reason: str,
        ruleset_version: str,
        patchset_ids: list[str],
        evaluation_ids: list[str],
        source_artifact_hashes: dict[str, str],
        source_round_lineage: dict[str, int] | None = None,
    ) -> RunMigrationRecord:
        """Copy immutable evidence only after validating the complete manifest."""
        artifacts: list[tuple[str, Path, Path, bytes]] = []
        for kind, identifiers, source_dir, target_dir, id_field in (
            ("patchsets", patchset_ids, source.patchsets_dir, self.patchsets_dir, "patchset_id"),
            ("evaluations", evaluation_ids, source.evaluations_dir, self.evaluations_dir, "eval_id"),
        ):
            for identifier in identifiers:
                relpath = f"{kind}/{identifier}.json"
                source_path = source_dir / f"{identifier}.json"
                data_bytes = source_path.read_bytes()
                actual_hash = hashlib.sha256(data_bytes).hexdigest()
                if source_artifact_hashes.get(relpath) != actual_hash:
                    raise ValueError(f"hash mismatch for {relpath}")
                data = json.loads(data_bytes)
                if data.get(id_field) != identifier:
                    raise ValueError(f"embedded ID mismatch for {relpath}")
                schema = str(data.get("schema_version") or "1.0")
                if schema not in {"1.0", "2.0"}:
                    singular = "patchset" if kind == "patchsets" else "evaluation"
                    raise ValueError(f"unsupported {singular} schema: {schema}")
                target_path = target_dir / f"{identifier}.json"
                if target_path.exists() and target_path.read_bytes() != data_bytes:
                    raise FileExistsError(f"refusing to overwrite different immutable record: {target_path}")
                artifacts.append((relpath, source_path, target_path, data_bytes))

        migration = RunMigrationRecord(
            source_run_id=source_run_id,
            source_run_path=str(source.run_dir),
            target_run_id=target_run_id,
            target_run_path=str(self.run_dir),
            migration_reason=migration_reason,
            ruleset_version=ruleset_version,
            status="imported",
            patchset_ids=list(patchset_ids),
            evaluation_ids=list(evaluation_ids),
            source_artifact_hashes=dict(source_artifact_hashes),
            source_round_lineage=dict(source_round_lineage or {}),
            created_at=utc_now(),
        )
        migration.migration_id = record_hash(migration, id_field="migration_id")
        migration_path = self.migrations_dir / f"{migration.migration_id}.json"
        if migration_path.exists():
            write_json(migration_path, migration)
            return migration

        # All validation/conflict checks completed before the first write.
        for _, _, target_path, data_bytes in artifacts:
            if target_path.exists():
                continue
            target_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = target_path.with_suffix(target_path.suffix + ".tmp")
            tmp.write_bytes(data_bytes)
            tmp.replace(target_path)
        self.put_migration(migration)
        return migration

    def latest_experience_text(self, *, max_chars: int = 6000) -> str:
        summary = self.experience_dir / "compressed_summary.md"
        if summary.exists():
            text = summary.read_text(encoding="utf-8", errors="replace")
            return text[-max_chars:]
        raw = self.experience_dir / "experience.jsonl"
        if not raw.exists():
            return ""
        text = raw.read_text(encoding="utf-8", errors="replace")
        return text[-max_chars:]

    def recent_negative_experience(self, *, max_items: int = 8) -> list[dict[str, Any]]:
        data = self.read_index("experience_recent_negatives")
        if isinstance(data, list):
            return data[-max_items:]
        return []
