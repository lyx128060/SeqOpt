#!/usr/bin/env python3
"""Sample benchmark queries for the DeepSeek-Harness optimization loop.

Reuses the validated ``seq_align.data_loader`` exactly as the shell path
(``data_loader.py -n N --dest DIR`` plus ``--smoke -n 10``) so the sampled set
matches the original workflow bit-for-bit.

stdout is a single JSON line: {"status":"ok","query_dir":...,"n_full":N,"n_smoke":10}
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
PROJECT_ROOT = _HERE.parent
SEQ_DIR = PROJECT_ROOT / "workspace" / "seq_align"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True, help="dest dir for sampled queries")
    ap.add_argument("--n-full", type=int, default=5)
    ap.add_argument("--n-smoke", type=int, default=10)
    ap.add_argument("--python", default=sys.executable)
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    loader = str(SEQ_DIR / "data_loader.py")

    try:
        subprocess.run(
            [args.python, loader, "-n", str(args.n_full), "--dest", str(data_dir)],
            check=True,
        )
        subprocess.run(
            [args.python, loader, "--smoke", "-n", str(args.n_smoke), "--dest", str(data_dir)],
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        print(json.dumps({"status": "sample_error", "error": f"data_loader exit {exc.returncode}"}), flush=True)
        return 1

    manifest = data_dir / "query_manifest.json"
    print(json.dumps({
        "status": "ok",
        "query_dir": str(data_dir),
        "n_full": args.n_full,
        "n_smoke": args.n_smoke,
        "manifest": str(manifest),
    }, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
