"""Runner module (no Harness).

Execution modes:
  local : thin wrapper around the existing pipeline entry point
          run_hmmer_cli.run_all_queries(...). That function already implements
          the exact two-phase flow (jackhmmer -N iter --chkhmm -> take last
          checkpoint -> hmmsearch against ASTRAL) and the exact wall-clock
          timing (time.perf_counter() wrapping both commands). Reusing it keeps
          the numbers identical to the shell/SLURM path. Prebuilt binaries are
          supplied via the SEQ_ALIGN_HMMER_BIN_DIR env var (how that entry point
          discovers a prebuilt bundle); an empty AssembledConfig means "stock
          CLI flags", i.e. no recompilation of C patches.
  slurm : placeholder for a later sbatch submission wrapper (Harness phase).

No hardcoded paths: every path / knob is a parameter or read from the caller.
"""

from __future__ import annotations

import os
import time
from typing import Any, Dict, Optional

from .assembler import AssembledConfig
from . import run_hmmer_cli
from .eval_schema import EvalResult

__all__ = ["run_variant_local", "submit_slurm_wrapper"]


def run_variant_local(
    *,
    variant_name: str,
    bin_dir: str,
    data_dir: str,
    astral_db: str,
    uniref_db: str,
    scop_txt: str,
    run_root: str,
    cpus: int = 8,
    max_iter: int = 3,
    stage: str = "full",
    query_timeout: Optional[float] = None,
) -> Dict[str, Any]:
    """Run one variant (its prebuilt jackhmmer/hmmsearch bundle) over the
    shared query set and return a flat metric dict.

    Args:
        variant_name: e.g. "round_022" or "round_baseline"
        bin_dir: directory containing jackhmmer + hmmsearch (bundle's bin/)
        data_dir: directory with all_samples.part_*.fasta + manifest
        astral_db / uniref_db / scop_txt: benchmark database paths
        run_root: scratch/work root (kept if you set keep_work=True below)
        cpus / max_iter / stage / query_timeout: benchmark knobs (max_iter=3)

    Returns:
        {"Variant","Status","TP","FP","TP_FP_Ratio","Cost_s","N_seqs",
         "Iterations","Errors"}
    """
    jack_bin = os.path.join(bin_dir, "jackhmmer")
    hmm_bin = os.path.join(bin_dir, "hmmsearch")
    for b in (jack_bin, hmm_bin):
        if not os.path.isfile(b):
            raise FileNotFoundError(f"bundle binary not found: {b}")

    os.makedirs(run_root, exist_ok=True)
    os.environ["SEQ_ALIGN_HMMER_BIN_DIR"] = bin_dir  # prebuilt bundle discovery

    cfg = AssembledConfig()  # empty -> stock CLI flags, no C-patch recompile
    result: EvalResult = run_hmmer_cli.run_all_queries(
        cfg,
        data_dir=data_dir,
        cpus=cpus,
        stage=stage,
        max_iter=max_iter,
        astral_db=astral_db,
        uniref_db=uniref_db,
        scop_txt=scop_txt,
        work_root=run_root,
        keep_work=True,          # preserve query_* dirs for inspection
        query_timeout=query_timeout,
    )
    # cost_s is the wall-clock aggregate already computed inside the backend;
    # keep it untouched so timing matches the shell/SLURM path exactly.
    meta = result.meta or {}
    return {
        "Variant": variant_name,
        "Status": result.status,
        "TP": result.tp,
        "FP": result.fp,
        "TP_FP_Ratio": result.tp_fp_ratio,
        "Cost_s": result.cost_s,
        "N_seqs": result.n_seqs,
        "Iterations": result.iterations,
        "Errors": "; ".join(result.errors[:3]),
        "query_results": list(meta.get("query_results") or []),
    }


def submit_slurm_wrapper(cmd: str, slurm_cfg: Dict[str, Any]) -> str:
    """Placeholder: submit a command via sbatch --parsable and return job id.

    To be implemented in the Harness phase. `slurm_cfg` may carry:
        {"cpus","mem","time","job_name","output","error","partition"}
    """
    raise NotImplementedError(
        "submit_slurm_wrapper is a placeholder; wire it during the Harness step."
    )
