#!/usr/bin/env python3
"""Hierarchical-loop CLI for the callback ("agent as proposer") bridge.

Unlike ``run_round.py`` (which drives the legacy single-round ``run_loop``), this
CLI drives the *production* engine ``run_hierarchical_loop`` and exposes the
controller's control flow to an external agent one proposal at a time:

    agent  --(no proposal)-->  hloop_cli.py  --> {"status": "needs_proposal", "kind": ...}
    agent writes proposal.json
    agent  --(--proposal ...)-->  hloop_cli.py --> ... continues the same invocation ...

The loop state lives in Python (the run directory + slot events), never in the
agent. Each invocation either advances the loop to completion, or yields back with
the next thing it needs. stdout carries exactly one JSON object per invocation;
all logging goes to stderr.

Usage
-----
    # step 1: engine reports what it needs (runs the stock baseline first)
    python hloop_cli.py --run-dir RUN --outer-rounds 3 --inner-trials 3

    # step 2: answer it
    python hloop_cli.py --run-dir RUN --outer-rounds 3 --inner-trials 3 \
        --proposal proposal.json --proposal-kind algorithm_plan

    # local, cluster-free protocol test (synthetic evaluations)
    python hloop_cli.py --run-dir RUN --outer-rounds 3 --inner-trials 3 --stub-eval
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
PROJECT_ROOT = _HERE.parent
for _p in (str(PROJECT_ROOT / "workspace"), str(PROJECT_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from seq_align.c_patch_loop.callback_proposer import (  # noqa: E402
    KIND_ALGORITHM_PLAN,
    VALID_KINDS,
    CallbackProposer,
    NeedProposal,
)
from seq_align.c_patch_loop.controller import run_hierarchical_loop  # noqa: E402
from seq_align.c_patch_loop.proposer import FatalProposalError, ProposalError  # noqa: E402


def _log(message: str, *args: object) -> None:
    """Business log. MUST NOT write to stdout (the agent parses stdout as JSON)."""
    logging.info(message, *args)


def _emit(payload: dict) -> None:
    print(json.dumps(payload, ensure_ascii=False), flush=True)


def _compact_summary(item: object) -> dict | None:
    """Reduce a stored index summary to the fields an agent needs to orient."""
    if not isinstance(item, dict):
        return None
    keys = (
        "patchset_id", "eval_id", "round_idx", "stage",
        "cost_s", "tp", "fp", "tp_fp_ratio", "decision", "baseline", "speedup_fraction",
    )
    return {key: item[key] for key in keys if key in item}


def status_payload(run_dir: Path) -> dict:
    """Read-only snapshot of the loop state. MUST NOT advance the loop.

    A resumable loop needs an orientation call that is safe to repeat: without it
    an agent that lost its place has no way to ask "where am I?" except by
    submitting another proposal. This reads the same LoopStore the engine writes,
    so there is one source of truth.
    """
    from seq_align.c_patch_loop.store import LoopStore

    store = LoopStore(run_dir)
    heads = store.slot_heads()
    state_counts: dict[str, int] = {}
    slots: list[dict] = []
    for key, head in sorted(heads.items()):
        state = str(getattr(head, "state", "unknown"))
        state_counts[state] = state_counts.get(state, 0) + 1
        slots.append({"slot": key, "state": state})

    run_rows = store.read_index("hierarchical_run_summary") or store.read_index("run_summary")
    return {
        "status": "status",
        "run_dir": str(run_dir),
        "invocation_count": len(list(store.iter_invocations())),
        "slot_state_counts": state_counts,
        "slots": slots[:60],
        "slots_truncated": len(slots) > 60,
        "baseline": _compact_summary(store.read_index("baseline")),
        "current_best": _compact_summary(store.read_index("current_best")),
        "current_speed_best": _compact_summary(store.read_index("current_speed_best")),
        "round_count": len(run_rows) if isinstance(run_rows, list) else None,
        "recent_rounds": [
            _compact_summary(row) for row in (run_rows[-8:] if isinstance(run_rows, list) else [])
        ],
    }


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Drive the hierarchical seq_align loop one proposal at a time.")
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--outer-rounds", type=int, default=1)
    ap.add_argument("--inner-trials", type=int, default=3)
    ap.add_argument("--stage", default="full", choices=["smoke", "full"])
    ap.add_argument("--cpus", type=int, default=1)
    ap.add_argument("--max-iter", type=int, default=1)
    ap.add_argument("--full-max-iter", type=int, default=None)
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--astral-db", default=None)
    ap.add_argument("--uniref-db", default=None)
    ap.add_argument("--scop-txt", default=None)
    ap.add_argument("--hmmer-src-root", default=None)
    ap.add_argument("--min-commit-speedup-fraction", type=float, default=0.03)
    ap.add_argument("--eval-repeats", type=int, default=1)
    # Paired confirmation. The default acceptance path compares a freshly measured
    # candidate cost against the *stored* current-best cost, so run-to-run timing
    # drift (node load, page cache, jackhmmer --cpu 0 oversubscription) is charged
    # to the patch. Observed 2026-10-08: a deliberately score-neutral probe patch
    # "won" speedup_fraction=0.76 that way and was committed. >0 turns on the
    # interleaved paired re-measurement (runner._should_run_paired_confirmation)
    # and the decision becomes "accepted_speedup_paired".
    ap.add_argument(
        "--paired-current-best-repeats",
        type=int,
        default=0,
        help=(
            "paired re-measurements of the current best, interleaved with the candidate; "
            "0 = legacy single-shot compare against the stored cost (noise-prone)"
        ),
    )
    # Inner-trial screening. Inner trials have a zero threshold: they rank
    # candidates for the outer commit gate and never judge them, so paying the
    # full repeats+paired measurement per trial buys precision nothing consumes.
    # With screening on, each trial is measured once on the small query subset and
    # only candidates that could plausibly clear the commit gate earn the full
    # measurement. Rejections are tagged rejected_screening_not_faster and can
    # never be committed. Screening needs <data-dir>/smoke; without it the trial
    # falls through to the full measurement (fail-open).
    ap.add_argument(
        "--screen-inner-trials",
        dest="screen_inner_trials",
        action=argparse.BooleanOptionalAction,
        default=os.getenv("SEQ_ALIGN_SCREEN_INNER_TRIALS", "1").lower()
        not in {"0", "false", "no", "off"},
        help=(
            "screen each inner trial on the small query subset with one measurement, "
            "and spend the full repeats+paired measurement only on candidates that "
            "could clear the commit gate (default: on; needs <data-dir>/smoke)"
        ),
    )
    ap.add_argument(
        "--screen-min-speedup-fraction",
        type=float,
        default=float(os.getenv("SEQ_ALIGN_SCREEN_MIN_SPEEDUP_FRACTION", "0.5")),
        help=(
            "fraction of min-commit-speedup-fraction a trial must reach in the screen "
            "to earn the full measurement (default 0.5 = half the commit gate)"
        ),
    )
    ap.add_argument(
        "--screen-reject-speedup-fraction",
        type=float,
        default=float(os.getenv("SEQ_ALIGN_SCREEN_REJECT_SPEEDUP_FRACTION", "0")),
        help=(
            "WIDE-BAND hard reject for the inner screen, expressed as a speedup "
            "fraction. 0 (default) = the screen only orders candidates and never "
            "discards one; any trial whose screen signal is near the commit gate must "
            "still earn the authoritative full measurement. Set this from the measured "
            "zero-effect distribution, never from a guess."
        ),
    )
    ap.add_argument(
        "--screen-eval-repeats",
        type=int,
        default=int(os.getenv("SEQ_ALIGN_SCREEN_EVAL_REPEATS", "1")),
        help="measurement repeats for the screening pass itself (default 1)",
    )
    # Post-hoc stage profiling: one extra instrumented evaluation for the round-000
    # baseline and for each committed winner -- never for an inner trial. Defaults
    # from the environment so an operator can set it once for the whole session
    # (the plugin inherits this variable through its spawned subprocess).
    ap.add_argument(
        "--posthoc-profile",
        default=os.getenv("SEQ_ALIGN_POSTHOC_PROFILE", "auto"),
        choices=["off", "auto", "local", "slurm"],
        help="Run an instrumented stage-timing evaluation on the baseline and each committed winner.",
    )
    # The callback path must never repair in-process: every rejection is handed
    # back so the agent authors the correction. One attempt, then yield.
    ap.add_argument("--max-repair-attempts", type=int, default=1)
    ap.add_argument(
        "--proposal",
        default=None,
        help="proposal JSON authored by the agent (omit to ask the engine what it needs)",
    )
    ap.add_argument("--proposal-kind", default=KIND_ALGORITHM_PLAN, choices=list(VALID_KINDS))
    ap.add_argument(
        "--session-id",
        default=None,
        help=(
            "Stable identity for this optimization session. Pass the SAME value on "
            "every invocation: it makes re-attachment to an interrupted run "
            "guaranteed, instead of depending on the environment fingerprint being "
            "byte-stable across calls."
        ),
    )
    ap.add_argument(
        "--proposal-round",
        type=int,
        default=None,
        help="outer round this proposal is authored for (recommended: binds the proposal to one request)",
    )
    ap.add_argument(
        "--keep-work",
        action="store_true",
        help="retain per-query work directories (debugging)",
    )
    ap.add_argument(
        "--stub-eval",
        action="store_true",
        help="LOCAL ONLY: synthetic evaluations, no Slurm/compiler/databases",
    )
    ap.add_argument(
        "--stub-keep-inner",
        action="store_true",
        help=(
            "LOCAL ONLY (implies --stub-eval): keep the real inner-slot control flow so "
            "inner proposal yields are exercised; the loop then stops at the first "
            "inner proposal request instead of synthesizing trials"
        ),
    )
    ap.add_argument(
        "--status",
        action="store_true",
        help=(
            "Print a read-only snapshot of this run directory's loop state and exit "
            "WITHOUT advancing the loop. Safe to call at any time."
        ),
    )
    return ap


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        stream=sys.stderr,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    args = build_parser().parse_args(argv)
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    # Orientation is a separate, non-advancing mode: it must not construct a
    # proposer or touch the engine.
    if args.status:
        _emit(status_payload(run_dir))
        return 0

    # Make the one failure mode that silently poisons a run impossible to miss.
    # Without paired re-measurement the candidate's fresh cost is compared against
    # the STORED current-best cost, so whatever the cluster was doing during the
    # earlier measurement gets charged to the patch.
    if args.paired_current_best_repeats <= 0:
        logging.warning(
            "acceptance decision is noise-prone: --paired-current-best-repeats=%d compares this "
            "run's candidate cost against the STORED current-best cost, so run-to-run timing drift "
            "is charged to the patch (a score-neutral probe patch once 'won' 76%% that way and was "
            "committed). For a trustworthy decision pass --eval-repeats 2 "
            "--paired-current-best-repeats 2.",
            args.paired_current_best_repeats,
        )

    try:
        proposer = CallbackProposer(args.proposal, kind=args.proposal_kind, round_idx=args.proposal_round)
    except ProposalError as exc:
        _log("proposal file rejected before the run: %s", exc)
        _emit({"status": "proposal_error", "error": str(exc)})
        return 0

    runner = None
    if args.stub_eval or args.stub_keep_inner:
        from stub_runner import install_stub_runner  # noqa: PLC0415 - local-only path

        runner = install_stub_runner(run_dir, patch_inner_slot=not args.stub_keep_inner)
        _log(
            "STUB EVAL ACTIVE: synthetic evaluations, engine not exercised (inner_slot=%s)",
            "REAL" if args.stub_keep_inner else "synthetic",
        )

    try:
        results = run_hierarchical_loop(
            run_dir=run_dir,
            outer_rounds=args.outer_rounds,
            proposer=proposer,
            inner_trials=args.inner_trials,
            cpus=args.cpus,
            max_iter=args.max_iter,
            full_max_iter=args.full_max_iter,
            hmmer_src_root=args.hmmer_src_root,
            data_dir=args.data_dir,
            astral_db=args.astral_db,
            uniref_db=args.uniref_db,
            scop_txt=args.scop_txt,
            keep_work=args.keep_work,
            max_repair_attempts=args.max_repair_attempts,
            min_commit_speedup_fraction=args.min_commit_speedup_fraction,
            eval_repeats=args.eval_repeats,
            paired_current_best_repeats=args.paired_current_best_repeats,
            runner=runner,
            session_id=args.session_id,
            posthoc_profile=args.posthoc_profile,
            screen_inner_trials=args.screen_inner_trials,
            screen_min_speedup_fraction=args.screen_min_speedup_fraction,
            screen_reject_speedup_fraction=args.screen_reject_speedup_fraction,
            screen_eval_repeats=args.screen_eval_repeats,
        )
    except NeedProposal as exc:
        # Expected, non-error outcome: the engine wants the agent's next proposal.
        _log("yielding to agent: %s", exc)
        _emit(exc.to_payload())
        return 0
    except (FatalProposalError, ProposalError) as exc:
        _log("proposal failure: %s", exc)
        _emit({"status": "proposal_error", "error": str(exc), "fatal": isinstance(exc, FatalProposalError)})
        return 0
    except Exception as exc:  # noqa: BLE001 - surface engine faults to the agent
        _log("engine error: %r", exc)
        _emit({"status": "engine_error", "error": repr(exc)})
        return 1

    _emit({"status": "done", "outer_rounds": args.outer_rounds, "results": results})
    return 0


if __name__ == "__main__":
    sys.exit(main())
