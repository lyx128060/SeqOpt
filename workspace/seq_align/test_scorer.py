"""Regression test for scorer.py against historical ground truth.

Purpose: prove the pure-Python scoring (tblout parse + SCOP lineage match)
produces EXACTLY the same TP/FP as the old shell/SLURM path.

Ground truth sources (pick one):
  --log   a bench6_*_<jobid>.out log. Each data line is
              [round_XXX] <qid>: iters=N hits=H tp=T fp=F cost=C.s
          The log lists queries in run order; the work dir holds query_0001,
          query_0002, ... in the same order (the old script ran queries
          serially). We pair them by position and score each query's
          hmmsearch.tbl, asserting the recorded tp/fp.
  --csv   an old per-query CSV with Qid/TP/FP columns.
  --manual "qid:tbl_path:exp_tp:exp_fp" (repeatable).

Important: tbl files are matched by POSITION (log order <-> query_00XX folder
number), never by grepping the tbl's contents for the qid. A target query can
legitimately appear as a hit inside another query's tbl, so content-grep causes
cross-query mismatches (that is the bug this rewrite removes).

Usage:
    python3 test_scorer.py --scop <dir.cla.scope> \
        --log <bench6_*.out> --tbl-dir <round_XXX/work>
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys
from pathlib import Path

# Allow running as a plain script (python3 test_scorer.py) OR as a module
# (python3 -m seq_align.test_scorer). Insert the package parent (workspace/)
# into sys.path so `import seq_align.scorer` works either way.
if __package__ in (None, ""):
    sys.path.insert(
        0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )

from seq_align.scorer import evaluate_remote_homology, load_scop


# ---------------------------------------------------------------------------
# Positional pairing of log order -> query_00XX folder
# ---------------------------------------------------------------------------

_LOG_LINE_RE = re.compile(
    r"\[\S+\]\s+(\S+):\s+iters=(\d+)\s+hits=(\d+)\s+tp=(\d+)\s+fp=(\d+)\s+cost=([0-9.]+)s"
)


def _query_dirs_sorted(work_root: Path) -> list[Path]:
    """Return query_* dirs under work_root, sorted numerically by their number."""
    return sorted(
        (d for d in work_root.glob("query_*") if d.is_dir()),
        key=lambda p: int(p.name.split("_")[-1]),
    )


def build_positional_tbl_map(log_path: str, work_root: Path) -> dict[str, str]:
    """Map qid -> hmmsearch.tbl by pairing log order with query_00XX order."""
    qids: list[str] = []
    with open(log_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            m = _LOG_LINE_RE.search(line)
            if m:
                qids.append(m.group(1))
    dirs = _query_dirs_sorted(work_root)
    if len(qids) != len(dirs):
        raise SystemExit(
            f"[FATAL] log has {len(qids)} query lines but work has "
            f"{len(dirs)} query_* dirs; cannot pair positionally. "
            f"Check --tbl-dir points at the matching round's work/."
        )
    mapping: dict[str, str] = {}
    for qid, d in zip(qids, dirs):
        tbl = d / "hmmsearch.tbl"
        if not tbl.is_file():
            raise SystemExit(f"[FATAL] {tbl} missing for {qid}")
        mapping[qid] = str(tbl)
    return mapping


# ---------------------------------------------------------------------------
# Cases
# ---------------------------------------------------------------------------

def _assert_one(scop, scop_path, qid: str, tbl: str, etp: int, efp: int) -> bool:
    res = evaluate_remote_homology(tbl, scop_path, qid, scop_cache=scop)
    ok = (res["TP"] == etp) and (res["FP"] == efp)
    print(f"{qid}: expect TP={etp} FP={efp} | "
          f"actual TP={res['TP']} FP={res['FP']} | {'PASS' if ok else 'FAIL'}")
    return ok


def run_log_cases(scop, scop_path, log_path: str, tbl_dir: str | None) -> bool:
    if not tbl_dir:
        print("[SKIP] --log requires --tbl-dir (the round's work/ dir)")
        return False
    tbl_map = build_positional_tbl_map(log_path, Path(tbl_dir))
    all_pass = True
    for qid, tbl in tbl_map.items():
        etp = efp = None
        with open(log_path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                m = _LOG_LINE_RE.search(line)
                if m and m.group(1) == qid:
                    etp, efp = int(m.group(4)), int(m.group(5))
                    break
        if etp is None:
            print(f"[SKIP] no log line for {qid}")
            all_pass = False
            continue
        all_pass = _assert_one(scop, scop_path, qid, tbl, etp, efp) and all_pass
    return all_pass


def run_csv_cases(scop, scop_path, csv_path: str, tbl_dir: str | None) -> bool:
    all_pass = True
    n = 0
    with open(csv_path, "r", encoding="utf-8", errors="replace") as f:
        for row in csv.DictReader(f):
            qid = (row.get("Qid") or row.get("qid") or row.get("SeqID")
                   or row.get("query_id") or "").strip()
            if not qid:
                continue
            tp_key = next((k for k in ("TP", "tp") if k in row), None)
            fp_key = next((k for k in ("FP", "fp") if k in row), None)
            if tp_key is None or fp_key is None:
                continue
            try:
                etp, efp = int(float(row[tp_key])), int(float(row[fp_key]))
            except (TypeError, ValueError):
                continue
            tbl = row.get("tbl") or row.get("tbl_path")
            if (not tbl) and tbl_dir:
                # exact-name match inside query_00XX dirs only
                for d in _query_dirs_sorted(Path(tbl_dir)):
                    if (d / f"{qid}.tbl").is_file():
                        tbl = str(d / f"{qid}.tbl")
                        break
                    if (d / "hmmsearch.tbl").is_file():
                        # first query dir whose tbl lists qid as its OWN query
                        # is ambiguous without a manifest; fall back to exact
                        # names above, else skip.
                        pass
            if not tbl or not os.path.isfile(tbl):
                print(f"[SKIP] no tbl for {qid}")
                all_pass = False
                continue
            all_pass = _assert_one(scop, scop_path, qid, tbl, etp, efp) and all_pass
            n += 1
    if n == 0:
        print("[WARN] no scorable rows matched; nothing asserted.")
        return False
    return all_pass


def run_manual_cases(scop, scop_path, manual: list[str]) -> bool:
    all_pass = True
    for spec in manual:
        parts = spec.split(":")
        if len(parts) != 4:
            print(f"[SKIP] malformed manual spec (want qid:tbl:tp:fp): {spec!r}")
            all_pass = False
            continue
        qid, tbl_path, etp, efp = parts[0], parts[1], int(parts[2]), int(parts[3])
        if not os.path.isfile(tbl_path):
            print(f"[SKIP] manual tbl not found: {tbl_path}")
            all_pass = False
            continue
        all_pass = _assert_one(scop, scop_path, qid, tbl_path, etp, efp) and all_pass
    return all_pass


# ---------------------------------------------------------------------------
# Entry
# ---------------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser(description="scorer regression vs historical ground truth")
    p.add_argument("--scop", required=True, help="path to dir.cla.scope.2.08-stable.txt")
    p.add_argument("--csv", default=None, help="old per_query CSV with TP/FP per qid")
    p.add_argument("--log", default=None, help="bench6_*_<jobid>.out log with per-query lines")
    p.add_argument("--tbl-dir", default=None, help="round_XXX/work dir holding query_*/hmmsearch.tbl")
    p.add_argument("--manual", action="append", default=[],
                   help='manual case "<qid>:<tbl_path>:<exp_tp>:<exp_fp>" (repeatable)')
    args = p.parse_args()

    scop = load_scop(args.scop)
    print(f"[*] loaded {len(scop)} SCOP lineages from {args.scop}")

    ok = True
    if args.csv:
        ok = run_csv_cases(scop, args.scop, args.csv, args.tbl_dir) and ok
    if args.log:
        ok = run_log_cases(scop, args.scop, args.log, args.tbl_dir) and ok
    if args.manual:
        ok = run_manual_cases(scop, args.scop, args.manual) and ok

    if not ok:
        print("\n[FAIL] at least one scorer case mismatched the historical record.")
        return 1
    print("\n[PASS] scorer TP/FP matches historical record.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
