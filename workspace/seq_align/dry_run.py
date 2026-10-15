"""End-to-end dry run: data sampling -> one variant run -> per-query CSV.

Small-scale integration test that wires the three modules together and checks
data flows end to end (file IO, DB access, jackhmmer+hmmsearch binaries,
SCOP scoring). It reuses the real pipeline entry points, so a passing dry run
means the pure-Python path is wired correctly.

Parallel usage (each shard = one query, submitted as its own sbatch job):

    # main job: sample once
    python3 dry_run.py --n-seq 5 --seed 996 --out <out> \
        --bin-dir <bundle_bin> --astral <a> --uniref <u> --scop <s>
        # -> writes <out>/common_query (shared by all shards)

    # shard job i (0..4), run in parallel
    python3 dry_run.py --query-dir <out>/common_query \
        --split-count 5 --split-index 0 \
        --out <out> --bin-dir <bundle_bin> --astral <a> --uniref <u> --scop <s>
        # -> writes <out>/per_query_all.0.csv

    # summary job (afterok all shards)
    # merges <out>/per_query_all.*.csv -> <out>/per_query_all.csv

A shard only sees the queries assigned to it (round-robin by part index), so
jobs can run concurrently without touching each other's work.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys
from pathlib import Path

# Allow running as a plain script (python3 dry_run.py) OR as a module
# (python3 -m seq_align.dry_run). Insert the package parent (workspace/)
# into sys.path so `import seq_align.runner` works either way.
if __package__ in (None, ""):
    sys.path.insert(
        0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )

from seq_align.dataloader import prepare_query_dataset
from seq_align.runner import run_variant_local


def partition_query_dir(src: str, split_count: int, split_index: int) -> str:
    """Return a sub-dir of `src` containing only the queries of this shard.

    Queries are the all_samples.part_*.fasta files, assigned round-robin by
    part index: file i goes to shard (i % split_count). The shard sub-dir
    symlinks its part files plus the manifest/target_ids, so the runner sees a
    complete (reduced) query set and nothing else touches the shared dir.
    """
    src_path = Path(src)
    parts = sorted(glob.glob(str(src_path / "all_samples.part_*.fasta")))
    mine = [p for i, p in enumerate(parts) if i % split_count == split_index]
    sub = src_path / f"split_{split_index}"
    sub.mkdir(exist_ok=True)
    for k, p in enumerate(mine):
        dst = sub / f"all_samples.part_{k + 1:03d}.fasta"
        if not dst.exists():
            os.symlink(os.path.abspath(p), dst)
    for name in ("query_manifest.json", "target_ids.txt"):
        f = src_path / name
        if f.exists() and not (sub / name).exists():
            os.symlink(os.path.abspath(f), sub / name)
    return str(sub)


def main() -> int:
    p = argparse.ArgumentParser(description="seq_align dry-run end-to-end")
    p.add_argument("--n-seq", type=int, default=5)
    p.add_argument("--seed", type=int, default=996)
    p.add_argument("--out", default="./dry_run_out")
    p.add_argument("--query-dir", default=None,
                   help="reuse an existing sampled query dir (skips sampling)")
    p.add_argument("--split-count", type=int, default=1,
                   help="total number of parallel shards (each a sbatch job)")
    p.add_argument("--split-index", type=int, default=0,
                   help="this job's shard index (0..split-count-1)")
    p.add_argument("--bin-dir", required=True, help="bundle dir with jackhmmer+hmmsearch")
    p.add_argument("--astral", required=True)
    p.add_argument("--uniref", required=True)
    p.add_argument("--scop", required=True)
    p.add_argument("--cpus", type=int, default=8)
    p.add_argument("--max-iter", type=int, default=3)
    args = p.parse_args()

    for label, path in (("astral", args.astral), ("uniref", args.uniref),
                        ("scop", args.scop), ("bin-dir", args.bin_dir)):
        if not os.path.exists(path):
            print(f"[FATAL] {label} not found: {path}")
            return 2

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # 1) sample (only the main job / first shard runs this; --query-dir skips it)
    if args.query_dir:
        query_dir = Path(args.query_dir)
    else:
        query_dir = out / "common_query"
        print(f"[1/3] sampling {args.n_seq} queries (seed={args.seed}) ...")
        prep = prepare_query_dataset(
            str(query_dir), args.n_seq, seed=args.seed,
            astral_db=args.astral, scop_txt=args.scop,
        )
        with open(prep["manifest_path"], "r", encoding="utf-8") as f:
            manifest = json.load(f)
        print(f"      manifest: {len(manifest.get('query_ids', []))} query ids")

    # 2) shard this job's view of the query set
    run_query_dir = query_dir
    run_root = out / "work"
    csv_name = "per_query_all.csv"
    if args.split_count > 1:
        run_query_dir = partition_query_dir(
            str(query_dir), args.split_count, args.split_index)
        run_root = out / f"work_{args.split_index}"
        csv_name = f"per_query_all.{args.split_index}.csv"
    print(f"[2/3] running variant over shard query dir {run_query_dir} ...")

    # 3) run one variant over the shared query set (the whole per-query loop
    #    lives inside run_hmmer_cli.run_all_queries, preserving the exact
    #    timing and scoring of the shell path)
    metrics = run_variant_local(
        variant_name="dry_run",
        bin_dir=args.bin_dir,
        data_dir=run_query_dir,
        astral_db=args.astral,
        uniref_db=args.uniref,
        scop_txt=args.scop,
        run_root=str(run_root),
        cpus=args.cpus,
        max_iter=args.max_iter,
        stage="full",
    )
    print(f"      status={metrics['Status']} TP={metrics['TP']} FP={metrics['FP']} "
          f"Cost={metrics['Cost_s']}s N={metrics['N_seqs']} iter={metrics['Iterations']}")

    # 4) per-query CSV straight from the backend's own scoring (identical to
    #    the log-parsing output of the shell path)
    rows = []
    for q in metrics["query_results"]:
        rows.append({
            "Variant": "dry_run",
            "Qid": q.get("query_id") or "",
            "TP": q.get("tp", 0), "FP": q.get("fp", 0),
            "Hits_total": q.get("nhits") or 0,
            "Iter": q.get("iterations", 0),
            "Cost_s": f"{q.get('cost_s', 0):.4f}",
        })
    rows.sort(key=lambda r: r["Qid"])

    csv_path = out / csv_name
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["Variant", "Qid", "TP", "FP", "Hits_total", "Iter", "Cost_s"])
        w.writeheader()
        w.writerows(rows)
    print(f"      per-query csv -> {csv_path} ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
