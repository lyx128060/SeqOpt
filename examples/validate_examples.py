#!/usr/bin/env python3
"""Validate the example proposals against the real component contracts.

What this checks (all of it offline -- no cluster, no HMMER build, no databases):

  1. ``examples/inner_patch.example.json`` names a component that is an ACTIVE
     contract, and a target function that the contract permits;
  2. the patch's ``+++`` path matches that contract's target file;
  3. the example diff actually applies to a real HMMER 3.4 tree, if one is
     present (``bash scripts/setup_hmmer.sh``) -- a real check, not a shape check;
  4. ``examples/algorithm_plan.example.json`` carries the fields the outer layer
     requires.

This script deliberately does NOT evaluate anything. A patch that applies is not
evidence that it helps, and the example patch is a no-op placeholder.

Usage:
    python examples/validate_examples.py
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
for _p in (REPO_ROOT / "workspace", REPO_ROOT / "harness_plugins"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from seq_align.components.c_kernels import (  # noqa: E402
    ACTIVE_C_KERNEL_CONTRACTS,
)

OK = "  [ok]  "
BAD = "  [FAIL]"


def main() -> int:
    failures = 0

    print("=" * 72)
    print("Validating example proposals against the live component contracts")
    print("=" * 72)
    print()
    print("active contracts:")
    for name, c in sorted(ACTIVE_C_KERNEL_CONTRACTS.items()):
        print(f"  {name:24s} -> {c.c_target_file}")
    print()

    # ------------------------------------------------------------------ inner
    inner_path = REPO_ROOT / "examples" / "inner_patch.example.json"
    inner = json.loads(inner_path.read_text())
    print(f"inner patch: {inner_path.relative_to(REPO_ROOT)}")
    component = inner.get("component_name")
    target_fn = inner.get("target_function")
    print(f"  component_name  = {component!r}")
    print(f"  target_function = {target_fn!r}")

    contract = ACTIVE_C_KERNEL_CONTRACTS.get(component)
    if contract is None:
        print(f"{BAD} {component!r} is not an active component contract")
        failures += 1
    else:
        print(f"{OK} component is an ACTIVE contract")

        allowed = set(contract.c_target_functions)
        if target_fn not in allowed:
            print(f"{BAD} {target_fn!r} is not permitted for {component!r}")
            print(f"        permitted: {sorted(allowed)}")
            failures += 1
        else:
            print(f"{OK} target function is permitted by the contract")

        diff = inner.get("unified_diff", "")
        plus = [ln for ln in diff.splitlines() if ln.startswith("+++ ")]
        if not plus:
            print(f"{BAD} unified_diff has no '+++ ' header")
            failures += 1
        else:
            declared = plus[0][len("+++ b/") :] if plus[0].startswith("+++ b/") else plus[0][6:]
            if not declared.endswith(contract.c_target_file):
                print(
                    f"{BAD} diff targets {declared!r} but the contract's file is "
                    f"{contract.c_target_file!r}"
                )
                failures += 1
            else:
                print(f"{OK} diff target path matches the contract ({contract.c_target_file})")
    print()

    # ------------------------------------------------------------------ outer
    outer_path = REPO_ROOT / "examples" / "algorithm_plan.example.json"
    outer = json.loads(outer_path.read_text())
    print(f"outer plan: {outer_path.relative_to(REPO_ROOT)}")
    print(f"  algorithm_family = {outer.get('algorithm_family')!r}")
    print(
        "  note: the family -> component binding is enforced by the controller at\n"
        "        proposal time; here we only check the file is well formed."
    )
    for field in ("algorithm_change", "speed_mechanism", "quality_preservation_argument"):
        if not outer.get(field):
            print(f"{BAD} outer plan is missing {field!r}")
            failures += 1
        else:
            print(f"{OK} has {field}")
    print()

    # ------------------------------------------------- real apply (if available)
    hmmer_root = REPO_ROOT / "tools" / "hmmer-3.4"
    print(f"HMMER source tree: {hmmer_root}")
    if not (hmmer_root / "src" / "p7_evo_align.c").is_file():
        print(
            "  [skip] not present, so the 'does the example diff apply' check is\n"
            "         skipped. Run `bash scripts/setup_hmmer.sh` to enable it\n"
            "         (this also un-skips the 37 requires_hmmer_source tests)."
        )
    else:
        proc = subprocess.run(
            ["git", "apply", "--check", "-p1", "-"],
            input=inner.get("unified_diff", ""),
            text=True,
            cwd=hmmer_root,
            capture_output=True,
        )
        if proc.returncode != 0:
            print(f"{BAD} example diff does NOT apply to the HMMER tree:")
            print((proc.stderr or proc.stdout).strip())
            failures += 1
        else:
            print(f"{OK} example diff applies cleanly to the HMMER tree")
    print()

    print("=" * 72)
    if failures:
        print(f"{failures} check(s) FAILED")
        return 1
    print("All checks passed.")
    print("This shows the examples are well formed and applyable.")
    print("It says NOTHING about performance: the example patch is a no-op.")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
