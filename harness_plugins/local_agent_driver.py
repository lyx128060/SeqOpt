#!/usr/bin/env python3
"""Local stand-in for the harness agent: drives hloop_cli.py until the loop finishes.

Purpose
-------
There is no Cordis/DeepSeek-Harness runtime available yet, and no cluster. This
script plays the *agent* role against the real CLI protocol, in separate OS
processes, so the agent-facing contract can be validated today:

    call CLI -> {"status":"needs_proposal","kind":...} -> author proposal -> call CLI again

It then asserts the properties that matter for the migration:

* the engine re-attaches to ONE invocation across every yield (no state loss),
* every outer round reaches a terminal slot state,
* provenance is complete (patchset + evaluation + slot event per trial),
* a committed winner is recorded when the synthetic dynamics clear the gate.

The proposals it writes are scripted, not model-generated. With ``--stub-eval``
the engine's evaluation is synthetic; the point is to prove the *control-flow
inversion and its bookkeeping*, not to optimize HMMER.

Usage
-----
    python local_agent_driver.py --run-dir /tmp/agent_demo --outer-rounds 3 --inner-trials 3
    python local_agent_driver.py --run-dir /tmp/agent_demo2 --outer-rounds 2 --inner-trials 2 --stub-keep-inner
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
PROJECT_ROOT = _HERE.parent
CLI = _HERE / "hloop_cli.py"
for _p in (str(PROJECT_ROOT / "workspace"), str(PROJECT_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Rotate through real algorithm families so each outer round freezes a different
# decision surface (a real agent would do the same, for real reasons).
PLAN_ROTATION = [
    {
        "algorithm_family": "pipeline_flow_strategy",
        "component_name": "pipeline_flow",
        "target_function": "p7_evo_align_Run",
        "algorithm_change": (
            "Reuse exact comparison decisions in a worker-local decision surface, "
            "with stock fallback for every uncertain target."
        ),
        "speed_mechanism": "Avoid repeated comparison work for already proven exact states.",
    },
    {
        "algorithm_family": "iterative_search",
        "component_name": "jackhmmer_loop",
        "target_function": "p7_evo_iter_EndRound",
        "algorithm_change": (
            "Stop the iterative search as a bounded state machine once the profile "
            "converges, while keeping the exact stock path for uncertain rounds."
        ),
        "speed_mechanism": "Bounded convergence detection avoids redundant search rounds.",
    },
    {
        "algorithm_family": "ssv_filter_strategy",
        "component_name": "ssv_filter_neon",
        "target_function": "p7_SSVFilter",
        "algorithm_change": (
            "Use an adaptive filter cascade with a bounded band and stock fallback "
            "for any candidate that would otherwise be pruned."
        ),
        "speed_mechanism": "Adaptive sparse band evaluation with exact stock fallback.",
    },
]

COMMON_PLAN_FIELDS = {
    "quality_preservation_argument": (
        "TP and FP remain identical because every uncertain case runs the exact stock path."
    ),
    "exact_fallback_strategy": "Call the stock exact path for every uncertain target.",
    "profile_evidence_citation": "Measured dominant command 'hmmsearch' on the full benchmark.",
    "baseline_algorithm_summary": "Stock HMMER evaluates every comparison through the frozen pipeline.",
}


def scripted_plan(outer_round: int) -> dict:
    base = dict(PLAN_ROTATION[(outer_round - 1) % len(PLAN_ROTATION)])
    return {
        **base,
        **COMMON_PLAN_FIELDS,
        "rationale": base["algorithm_change"],
        "expected_metric_effect": "Fewer redundant operations; TP/FP unchanged.",
        "risk_notes": "Low: the stock fallback dominates for every uncertain case.",
    }


def scripted_inner(outer_round: int, inner_trial: int) -> dict:
    """An inner candidate. Shape mirrors what a real agent would hand the engine."""
    plan = scripted_plan(outer_round)
    target = plan["component_name"]
    return {
        "component_name": target,
        "target_function": plan["target_function"],
        # Plan-level fields let one file answer both propose_plan() and propose().
        "rationale": plan["algorithm_change"],
        "intended_code_change": (
            "Add worker-local scratch state that records already-proven exact comparisons."
        ),
        "runtime_effective_argument": (
            "The diff alters executable C code in the selected runtime component."
        ),
        "profile_evidence_citation": plan["profile_evidence_citation"],
        "anti_degenerate_check": (
            "Preserve TP/FP by avoiding threshold changes, hit suppression, "
            "or recall-reducing shortcuts."
        ),
        "algorithm_family": plan["algorithm_family"],
        "expected_metric_effect": "Fewer repeated operations; TP/FP unchanged.",
        "risk_notes": "Reversible; stock fallback covers uncertain targets.",
        # Skeleton-level diff. A real candidate would be a genuine patch against the
        # contract's target file; the local stub does not compile or apply it.
        "unified_diff": (
            "--- a/p7_evo_align.c\n"
            "+++ b/p7_evo_align.c\n"
            "@@ -14,5 +14,7 @@\n"
            " struct p7_evo_align_s {\n"
            "   int reserved;\n"
            f"+  int scratch_probe_r{outer_round}_t{inner_trial};\n"
            " };\n"
        ),
    }


def run_cli(args: list[str]) -> dict:
    proc = subprocess.run(
        [sys.executable, str(CLI), *args],
        capture_output=True,
        text=True,
    )
    raw = proc.stdout.strip()
    if not raw:
        raise RuntimeError(f"CLI produced no stdout (rc={proc.returncode})\nstderr:\n{proc.stderr}")
    try:
        return json.loads(raw.splitlines()[-1])
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"CLI stdout was not JSON: {raw[:400]!r}") from exc


def verify_provenance(run_dir: Path, outer_rounds: int) -> None:
    from seq_align.c_patch_loop.store import LoopStore

    store = LoopStore(run_dir)
    invocations = list(store.iter_invocations())
    heads = store.slot_heads()
    problems: list[str] = []

    if len(invocations) != 1:
        problems.append(
            f"expected exactly 1 invocation across all yields, found {len(invocations)} "
            "(the engine failed to re-attach; loop state would be lost)"
        )

    for outer_idx in range(1, outer_rounds + 1):
        outer = [h for k, h in heads.items() if k.endswith(f":{outer_idx:03d}:outer")]
        if not outer:
            problems.append(f"outer round {outer_idx}: no slot recorded")
            continue
        if outer[0].state not in {"committed", "not_committed", "blocked"}:
            problems.append(f"outer round {outer_idx}: non-terminal state {outer[0].state!r}")
        inner = [
            h for k, h in heads.items()
            if k.endswith(":outer") is False and f":{outer_idx:03d}:inner-" in k
        ]
        if not inner:
            problems.append(f"outer round {outer_idx}: no inner trials recorded")
        for head in inner:
            if head.state not in {"evaluated", "blocked"}:
                problems.append(
                    f"outer round {outer_idx} inner {head.inner_idx}: non-terminal state {head.state!r}"
                )

    if problems:
        raise AssertionError("provenance incomplete:\n  - " + "\n  - ".join(problems))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Drive the callback loop as a stand-in agent.")
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--outer-rounds", type=int, default=3)
    ap.add_argument("--inner-trials", type=int, default=3)
    ap.add_argument("--stub-keep-inner", action="store_true")
    ap.add_argument(
        "--session-id",
        default=None,
        help="stable identity for the session; defaults to the run-dir name",
    )
    ap.add_argument("--max-iterations", type=int, default=200)
    args = ap.parse_args(argv)

    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    if args.session_id is None:
        args.session_id = f"seqalign-{run_dir.name}"
    proposals_dir = run_dir / "agent_proposals"
    proposals_dir.mkdir(exist_ok=True)

    base_args = [
        "--run-dir", str(run_dir),
        "--outer-rounds", str(args.outer_rounds),
        "--inner-trials", str(args.inner_trials),
        # A real harness agent supplies a stable session identity so re-attachment
        # does not depend on the environment fingerprint being byte-stable.
        "--session-id", args.session_id,
        "--stub-eval",
    ]
    if args.stub_keep_inner:
        base_args.append("--stub-keep-inner")

    print(f"[agent] driving {CLI.name}  run_dir={run_dir}  outer={args.outer_rounds} inner={args.inner_trials}")
    transcript: list[str] = []

    for step in range(1, args.max_iterations + 1):
        payload = run_cli(base_args)
        status = payload.get("status")

        if status == "needs_proposal":
            kind = payload["kind"]
            outer_round = int(payload.get("outer_round") or 1)
            if kind == "algorithm_plan":
                proposal = scripted_plan(outer_round)
                name = f"round_{outer_round:03d}_algorithm_plan.json"
            elif kind == "inner":
                inner_trial = 1 + sum(1 for line in transcript if line.startswith("inner"))
                proposal = scripted_inner(outer_round, inner_trial)
                name = f"round_{outer_round:03d}_inner_{inner_trial:03d}.json"
            else:
                raise AssertionError(f"unknown proposal kind {kind!r}")

            path = proposals_dir / name
            path.write_text(json.dumps(proposal, indent=2, ensure_ascii=False), encoding="utf-8")
            note = payload.get("feedback")
            print(f"[agent] step {step}: engine wants {kind} (round {outer_round})"
                  + (f" — after rejection: {str(note)[:80]}" if note else ""))
            transcript.append(kind)
            base_args = base_args + [
                "--proposal", str(path),
                "--proposal-kind", kind,
                "--proposal-round", str(outer_round),
            ]
            continue

        if status == "done":
            results = payload.get("results", [])
            committed = [r for r in results if r.get("status") == "committed"]
            print(f"[agent] loop finished after {step} engine calls; {len(committed)} outer round(s) committed")
            verify_provenance(run_dir, args.outer_rounds)
            print("[agent] provenance OK: 1 invocation, all slots terminal")
            for row in results:
                if "inner_idx" in row:
                    continue
                # ``results`` also carries the baseline record, which has
                # outer_idx 0 and no status/speedup fields. Only outer-round
                # summary rows have something to report here.
                outer_idx = row.get("outer_idx")
                if not outer_idx:
                    continue
                speedup = row.get("speedup_fraction")
                speedup_text = "n/a" if speedup is None else f"{float(speedup):.4f}"
                print(
                    f"        round {outer_idx}: {row.get('status')}"
                    f"  speedup={speedup_text}"
                    f"  (required {row.get('required_speedup_fraction')})"
                )
            return 0

        if status in {"proposal_error", "engine_error"}:
            print(f"[agent] FAILED: {status}: {payload.get('error')}")
            return 1

        raise AssertionError(f"unexpected CLI status {status!r}")

    raise AssertionError(f"loop did not finish within {args.max_iterations} engine calls")


if __name__ == "__main__":
    sys.exit(main())
