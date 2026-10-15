"""Evaluation / reporting for the seq_align pipeline.

The per-variant TP/FP scoring lives in run_jack._score_hits and uses complete
SCOP lineages from dir.cla.scope as ground truth.
This module provides two aggregate entry points:

1. aggregate_rounds(run_dir): collect every round_<N>/full_eval.json into a
   single performance_report.csv with the unified schema
   (SeqID-ish, stage, tp, fp, tp_fp_ratio, cost_s, status, iterations).
2. score_tbl(tbl_path, scop, query_class): legacy .tbl scoring kept for any
   ad-hoc tbl produced outside the pyhmmer path.

No more three-way CSV header mismatch: everything speaks EvalResult fields.
"""

import argparse
import csv
import glob as gl
import json
import os
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[1]
SCOP_TXT = os.getenv("SCOP_TXT", str(ROOT / "data/scop_benchmark/dir.cla.scope.2.08-stable.txt"))
CSV_REPORT_PATH = SCRIPT_DIR / "performance_report.csv"
DEBUG_LOG = SCRIPT_DIR / "debug.log"


def load_scop():
    """Return domain_id -> complete SCOP SCCS.

    dir.cla.scope verbatim layout: Sid PDB Residues Classification Sunid Description
    The classification id (e.g. ``a.1.1.1``) is column index 3. Falls back to
    scanning for the first SCOP-id-shaped token.
    """
    m = {}
    if not os.path.exists(SCOP_TXT):
        return m
    with open(SCOP_TXT, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            p = line.split()
            if len(p) < 2:
                continue
            sid = p[0]
            cls = None
            if len(p) >= 4 and _looks_like_scop_id(p[3]):
                cls = p[3]
            else:
                for tok in p[1:]:
                    if _looks_like_scop_id(tok):
                        cls = tok
                        break
            if cls:
                m[sid] = cls
    return m


def _looks_like_scop_id(tok):
    if not tok or "." not in tok:
        return False
    head = tok.split(".")[0]
    return len(head) == 1 and head.isalpha()


def score_tbl(tbl_path, scop, query_lineage):
    """Score a legacy .tbl against a query's SCOP lineage.

    Same-superfamily targets are TP, different-fold targets are FP, and
    same-fold/different-superfamily or unclassified targets are ignored.
    """
    tp = fp = 0
    if query_lineage is None or not os.path.exists(tbl_path):
        return tp, fp
    query_parts = query_lineage.split(".")
    if len(query_parts) < 3:
        return tp, fp
    with open(tbl_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.split()
            if not parts:
                continue
            tid = parts[0]
            candidates = [tid, tid[1:] if tid.startswith("d") else f"d{tid}"]
            target_lineage = next((scop.get(key) for key in candidates if scop.get(key)), None)
            if target_lineage is None:
                continue
            target_parts = target_lineage.split(".")
            if len(target_parts) < 3:
                continue
            if target_parts[:3] == query_parts[:3]:
                tp += 1
            elif target_parts[:2] != query_parts[:2]:
                fp += 1
    return tp, fp


def _eval_row(round_name, data, extra_errors=None):
    errors = list(data.get("errors", []))
    if extra_errors:
        errors.extend(extra_errors)
    return [
        round_name,
        data.get("stage", ""),
        data.get("status", ""),
        data.get("tp", 0),
        data.get("fp", 0),
        f"{data.get('tp_fp_ratio', 0.0):.4f}",
        data.get("cost_s", 0.0),
        data.get("n_seqs", 0),
        data.get("iterations", 0),
        "; ".join(errors)[:120],
    ]


def aggregate_rounds(run_dir, output_path=None):
    """Collect legacy round full_eval files plus canonical C-patch loop evals."""
    rows = []
    seen_canonical_eval_ids = set()

    for jf in sorted(gl.glob(os.path.join(run_dir, "c_patch_loop", "evaluations", "*.json"))):
        try:
            record = json.loads(Path(jf).read_text(encoding="utf-8"))
        except Exception as e:
            with open(DEBUG_LOG, "a", encoding="utf-8") as df:
                df.write(f"[ERROR] {jf}: {e}\n")
            continue
        if record.get("stage") != "full":
            continue
        data = record.get("eval_result") or {}
        eval_id = record.get("eval_id")
        if eval_id:
            seen_canonical_eval_ids.add(eval_id)
        rows.append(_eval_row(
            f"round_{int(record.get('round_idx', 0)):03d}/c_patch_loop",
            data,
            extra_errors=record.get("errors", []),
        ))

    for jf in sorted(gl.glob(os.path.join(run_dir, "round_*", "full_eval.json"))):
        rd = Path(jf).parent.name  # round_003
        try:
            data = json.loads(Path(jf).read_text(encoding="utf-8"))
        except Exception as e:
            with open(DEBUG_LOG, "a", encoding="utf-8") as df:
                df.write(f"[ERROR] {jf}: {e}\n")
            continue
        if (data.get("meta") or {}).get("c_patch_loop_eval_id") in seen_canonical_eval_ids:
            continue
        rows.append(_eval_row(rd, data))
    header = ["Round", "Stage", "Status", "TP", "FP", "TP_FP_Ratio",
              "Cost_s", "N_seqs", "Iterations", "Errors"]
    report_path = Path(output_path).resolve() if output_path else CSV_REPORT_PATH
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            w.writerow(r)
    print(f"[INFO] aggregated {len(rows)} rounds -> {report_path}")
    return report_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--round", type=int, default=None,
                        help="(legacy, ignored) kept for submit_jacklatest.sh compat")
    parser.add_argument("--run-dir", default=str(ROOT / "runs/evo_seq_optimize_main"),
                        help="run directory containing round_*/full_eval.json")
    parser.add_argument("--output", default=None,
                        help="optional output CSV path; defaults to workspace performance_report.csv")
    args = parser.parse_args()
    aggregate_rounds(args.run_dir, output_path=args.output)
