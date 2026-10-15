"""ASTRAL sequence sampling for the seq_align benchmark.

Fixes the old version's bugs (missing `import argparse`, hardcoded relative
glob glob) and adds a smoke subset: a fixed small set of query sequences +
a truncated UniRef used by run_short_eval's smoke stage. The full benchmark
still uses the full ASTRAL + full UniRef90 (path env-overridable; repo ships
no big files).
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import random
import shutil
import subprocess
from collections.abc import Iterator

ASTRAL_DB = os.getenv("ASTRAL_DB", "./data/scop_benchmark/astral-scopedom-seqres-gd-sel-gs-bib-40-2.08.fa")
UNIREF_DB = os.getenv("UNIREF_DB", "./data/uniref90.fasta")
SCOP_TXT = os.getenv("SCOP_TXT", "./data/scop_benchmark/dir.cla.scope.2.08-stable.txt")
DIR_DATA = "./workspace/seq_align/data_test"
DEFAULT_QUERY_SEED = 996


def _extract_ids(fasta_path: str) -> list[str]:
    """Return all header IDs from a fasta (strip '>' and take first token)."""
    ids = []
    with open(fasta_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith(">"):
                ids.append(line[1:].split()[0])
    return ids


def _load_scop_ids(scop_path: str) -> set[str]:
    """Return SCOP domain IDs that have a classification entry."""
    if not os.path.exists(scop_path):
        return set()
    ids: set[str] = set()
    with open(scop_path, "r", encoding="utf-8") as handle:
        for line in handle:
            if line and not line.startswith("#"):
                fields = line.split()
                if fields:
                    ids.add(fields[0])
    return ids


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _iter_fasta_records(fasta_path: str) -> Iterator[tuple[str, list[str]]]:
    """Yield ``(record_id, lines)`` for each FASTA record."""
    record_id: str | None = None
    lines: list[str] = []
    with open(fasta_path, "r", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith(">"):
                if record_id is not None:
                    yield record_id, lines
                record_id = line[1:].split()[0]
                lines = [line]
            elif record_id is not None:
                lines.append(line)
    if record_id is not None:
        yield record_id, lines


def _write_selected_records(fasta_path: str, selected: list[str], out_path: str) -> int:
    """Write selected FASTA records without requiring seqkit."""
    selected_set = set(selected)
    written = 0
    with open(out_path, "w", encoding="utf-8") as out:
        for record_id, lines in _iter_fasta_records(fasta_path):
            if record_id in selected_set:
                out.writelines(lines)
                written += 1
    missing = selected_set - {record_id for record_id, _ in _iter_fasta_records(out_path)}
    if missing:
        raise RuntimeError(f"failed to extract {len(missing)} sampled ids from {fasta_path}")
    return written


def _split_fasta_records(fasta_path: str, dest: str) -> int:
    """Split a FASTA into one-record query files matching seqkit split2 names."""
    for old_part in glob.glob(os.path.join(dest, "all_samples.part_*.fasta")):
        os.remove(old_part)
    n = 0
    for n, (_, lines) in enumerate(_iter_fasta_records(fasta_path), start=1):
        part = os.path.join(dest, f"all_samples.part_{n:03d}.fasta")
        with open(part, "w", encoding="utf-8") as out:
            out.writelines(lines)
    return n


def _write_first_records(fasta_path: str, out_path: str, n_records: int) -> int:
    """Write the first ``n_records`` FASTA records without requiring seqkit."""
    kept = 0
    with open(out_path, "w", encoding="utf-8") as out:
        for _, lines in _iter_fasta_records(fasta_path):
            if kept >= n_records:
                break
            out.writelines(lines)
            kept += 1
    return kept


def sample_sequences(
    num_samples: int,
    dest: str = DIR_DATA,
    *,
    seed: int = DEFAULT_QUERY_SEED,
) -> None:
    os.makedirs(dest, exist_ok=True)
    for f in glob.glob(os.path.join(dest, "*.fasta")):
        os.remove(f)

    all_ids = _extract_ids(ASTRAL_DB)
    if not all_ids:
        raise RuntimeError(f"no sequences found in {ASTRAL_DB}")
    scop_ids = _load_scop_ids(SCOP_TXT)
    if scop_ids:
        annotated_ids = [record_id for record_id in all_ids if record_id in scop_ids]
        if annotated_ids:
            all_ids = annotated_ids
        else:
            print(f"[WARN] no ASTRAL IDs matched SCOP annotations in {SCOP_TXT}")
    sampled = random.Random(seed).sample(all_ids, min(num_samples, len(all_ids)))

    target_ids_file = os.path.join(dest, "target_ids.txt")
    with open(target_ids_file, "w", encoding="utf-8") as f:
        f.write("\n".join(sampled))

    manifest = {
        "seed": seed,
        "query_count": len(sampled),
        "query_ids": sampled,
        "astral_db": os.path.abspath(ASTRAL_DB),
        "astral_sha256": _sha256(ASTRAL_DB),
        "scop_txt": os.path.abspath(SCOP_TXT),
        "scop_sha256": _sha256(SCOP_TXT) if os.path.exists(SCOP_TXT) else None,
    }
    with open(os.path.join(dest, "query_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)

    out = os.path.join(dest, "all_samples.fasta")
    if shutil.which("seqkit"):
        try:
            subprocess.run(["seqkit", "grep", "-f", target_ids_file, ASTRAL_DB, "-o", out],
                           check=True, capture_output=True, text=True)
            subprocess.run(["seqkit", "split2", "-l", "1", out, "-O", f"{dest}/"],
                           check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            print(f"[WARN] seqkit sampling failed ({e}); falling back to pure Python")
            _write_selected_records(ASTRAL_DB, sampled, out)
            _split_fasta_records(out, dest)
    else:
        print("[WARN] seqkit not found; sampling with pure Python fallback")
        _write_selected_records(ASTRAL_DB, sampled, out)
        _split_fasta_records(out, dest)
    print(f"[*] sampled {len(sampled)} sequences -> {dest}")


def build_smoke_subset(
    num_samples: int = 10,
    dest: str | None = None,
    *,
    seed: int = DEFAULT_QUERY_SEED,
) -> str:
    """Build the fixed smoke subset: N ASTRAL queries + truncated UniRef.

    The truncated UniRef is the first N*5 records (quick approximation). Falls
    back to a pure-Python record-head truncation if seqkit is unavailable.
    """
    if dest is None:
        dest = os.path.join(DIR_DATA, "smoke")
    os.makedirs(dest, exist_ok=True)
    sample_sequences(num_samples, dest=dest, seed=seed)
    smoke_uniref = os.path.join(dest, "uniref_smoke.fasta")
    n = num_samples * 5
    smoke_source = UNIREF_DB if os.path.exists(UNIREF_DB) else ASTRAL_DB
    if smoke_source != UNIREF_DB:
        print(f"[WARN] UNIREF_DB not found at {UNIREF_DB}; using ASTRAL_DB for smoke targets")
    if shutil.which("seqkit"):
        try:
            subprocess.run(["seqkit", "head", "-n", str(n), smoke_source, "-o", smoke_uniref],
                           check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            print(f"[WARN] seqkit head failed ({e}); falling back to pure Python")
            _write_first_records(smoke_source, smoke_uniref, n)
    else:
        print("[WARN] seqkit not found; building smoke UniRef with pure Python fallback")
        _write_first_records(smoke_source, smoke_uniref, n)
    # The runner uses data_test/smoke for query files but resolves the smoke
    # database from its parent data_test directory. Publish both locations so
    # the layout remains compatible with existing jobs.
    parent_uniref = os.path.join(os.path.dirname(dest), "uniref_smoke.fasta")
    if os.path.abspath(parent_uniref) != os.path.abspath(smoke_uniref):
        shutil.copyfile(smoke_uniref, parent_uniref)
    print(f"[*] smoke subset -> {dest} (uniref truncated to ~{n} records)")
    return dest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--num", type=int, default=5)
    parser.add_argument("--seed", type=int, default=DEFAULT_QUERY_SEED,
                        help=f"deterministic ASTRAL query-sampling seed (default: {DEFAULT_QUERY_SEED})")
    parser.add_argument("--smoke", action="store_true",
                        help="build the fixed smoke subset instead of a full sample")
    parser.add_argument("--dest", default=DIR_DATA,
                        help="output directory for sampled queries and smoke data")
    args = parser.parse_args()
    if args.smoke:
        build_smoke_subset(num_samples=args.num, dest=os.path.join(args.dest, "smoke"), seed=args.seed)
    else:
        sample_sequences(args.num, dest=args.dest, seed=args.seed)
