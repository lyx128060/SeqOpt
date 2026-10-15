"""Pending-modifications store: the hand-off between write_algorithm_patch and
run_short_eval / run_full_eval.

write_algorithm_patch writes a list of ComponentModification specs (one per
proposed algorithm change, Python payload or C diff) to a round-scoped JSON
file under runs/<run_dir>/round_<N>/pending_mods.json. The eval tools load
that file, feed it to assembler.assemble(), and run the variant. This decouples
"what the agent proposed" from "how it is evaluated", and keeps a durable
audit trail of every proposed change.
"""

from __future__ import annotations

import json
from pathlib import Path

PENDING_FILENAME = "pending_mods.json"
ACCEPTED_MODS_FILENAME = "accepted_mods.json"
ACCEPTED_EVAL_FILENAME = "accepted_eval.json"
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_SEQ_ALIGN_ROOT = Path(__file__).resolve().parent
_CONTAINER_SEQ_ALIGN = "/workspace/seq_align"


def normalize_run_dir(run_dir: str | Path) -> Path:
    """Return the host-visible run directory used by in-process tools.

    Custom tools execute in the host Python process, while the agent shell often
    thinks in Docker paths. Accept both host paths and the common container
    workspace path so a mistaken ``/workspace/seq_align/...`` argument cannot
    make the host try to create directories under ``/workspace``.
    """
    raw = str(run_dir).strip()
    if not raw:
        raise ValueError("run_dir cannot be empty")
    if raw == _CONTAINER_SEQ_ALIGN:
        return _SEQ_ALIGN_ROOT
    if raw.startswith(_CONTAINER_SEQ_ALIGN + "/"):
        rel = raw[len(_CONTAINER_SEQ_ALIGN) + 1:]
        if rel.startswith("runs/"):
            return (_PROJECT_ROOT / rel).resolve()
        return (_SEQ_ALIGN_ROOT / rel).resolve()

    path = Path(raw).expanduser()
    if path.is_absolute():
        return path
    return (_PROJECT_ROOT / path).resolve()


def round_dir(run_dir: str | Path, round_idx: int) -> Path:
    d = normalize_run_dir(run_dir) / f"round_{round_idx:03d}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def write_pending(run_dir: str | Path, round_idx: int, mods: list[dict]) -> Path:
    """Persist a list of modification dicts (ComponentModification.model_dump)."""
    p = round_dir(run_dir, round_idx) / PENDING_FILENAME
    p.write_text(json.dumps(mods, indent=2, ensure_ascii=False), encoding="utf-8")
    return p


def read_pending(run_dir: str | Path, round_idx: int) -> list[dict]:
    p = round_dir(run_dir, round_idx) / PENDING_FILENAME
    if not p.exists():
        return []
    return json.loads(p.read_text(encoding="utf-8"))


def read_accepted_mods(run_dir: str | Path) -> list[dict]:
    path = normalize_run_dir(run_dir) / ACCEPTED_MODS_FILENAME
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def read_accepted_eval(run_dir: str | Path) -> dict | None:
    path = normalize_run_dir(run_dir) / ACCEPTED_EVAL_FILENAME
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def write_accepted(run_dir: str | Path, mods: list[dict], result: dict) -> tuple[Path, Path]:
    """Update the variant inherited by subsequent rounds via atomic file replaces."""
    root = normalize_run_dir(run_dir)
    root.mkdir(parents=True, exist_ok=True)
    mods_path = root / ACCEPTED_MODS_FILENAME
    eval_path = root / ACCEPTED_EVAL_FILENAME
    for path, value in ((mods_path, mods), (eval_path, result)):
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
        tmp.replace(path)
    return mods_path, eval_path


def changed_components(candidate_mods: list[dict], accepted_mods: list[dict]) -> list[str]:
    candidate = {mod.get("contract_name"): mod for mod in candidate_mods}
    accepted = {mod.get("contract_name"): mod for mod in accepted_mods}
    return sorted(
        str(name) for name in set(candidate) | set(accepted)
        if candidate.get(name) != accepted.get(name)
    )
