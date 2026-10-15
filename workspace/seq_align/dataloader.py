"""Data preparation module (no Harness).

Thin wrapper around the existing data_loader.sample_sequences so callers get a
parameterized entry point (no hardcoded 40 / seed 996 / dest) and a manifest
path back. Sampling logic is NOT re-implemented here -- it reuses the project's
own sampler so the sampled query set stays identical to the shell/SLURM path.
"""

from __future__ import annotations

import os
from typing import Dict

from . import data_loader

__all__ = ["prepare_query_dataset", "resolve_query_dir"]


def prepare_query_dataset(
    out_dir: str,
    n_seq: int,
    seed: int = 996,
    *,
    astral_db: str | None = None,
    scop_txt: str | None = None,
) -> Dict[str, str]:
    """Sample `n_seq` ASTRAL queries into `out_dir` (deterministic by seed).

    Args:
        out_dir: destination directory (created if missing)
        n_seq: number of query sequences to sample
        seed: RNG seed (deterministic across runs)
        astral_db / scop_txt: optional overrides; fall back to the env-defaults
            used by data_loader (ASTRAL_DB / SCOP_TXT).

    Note: data_loader reads ASTRAL_DB / SCOP_TXT at import time into module
    globals, so we must overwrite those module attributes (not os.environ,
    which would be too late).

    Returns:
        {"query_dir": out_dir,
         "manifest_path": os.path.join(out_dir, "query_manifest.json")}
    """
    if astral_db:
        data_loader.ASTRAL_DB = astral_db
    if scop_txt:
        data_loader.SCOP_TXT = scop_txt

    os.makedirs(out_dir, exist_ok=True)
    data_loader.sample_sequences(n_seq, dest=out_dir, seed=seed)

    manifest = os.path.join(out_dir, "query_manifest.json")
    if not os.path.isfile(manifest):
        raise FileNotFoundError(f"manifest not generated: {manifest}")
    return {"query_dir": out_dir, "manifest_path": manifest}


def resolve_query_dir(data_dir: str) -> str:
    """Return the query directory holding all_samples.part_*.fasta."""
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"query dir not found: {data_dir}")
    return data_dir
