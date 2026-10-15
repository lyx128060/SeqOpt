# Examples

This directory provides **valid example inputs that require no cluster**.

| File | Purpose |
|---|---|
| `algorithm_plan.example.json` | Example outer proposal (algorithm plan) |
| `inner_patch.example.json` | Example inner proposal (C patch) |
| `validate_examples.py` | Validates the two files above against the **real contracts** |

> ⚠️ **These two JSON files are format examples, not valid optimization patches.**
> Their `unified_diff` points at real text from stock HMMER 3.4 and applies cleanly,
> but there is **no claim that they compile or produce a speedup**. Do not treat them as experimental results.

---

## Validating the examples (offline; no cluster / HMMER / database needed)

```bash
.venv/bin/python examples/validate_examples.py
```

It will:

1. print every active component contract;
2. check that the `component_name` of `inner_patch.example.json` is an active contract
   and that its `target_function` is in the permitted set;
3. check that the `+++` path of the diff matches the contract's target file;
4. check that the outer plan contains the fields required by the controller;
5. **if** `tools/hmmer-3.4/` exists, actually verify with `git apply --check` that the example
   patch applies cleanly (if not, skip explicitly and explain how to enable it).

The last line states explicitly: **this only proves that the example is correctly formatted and applicable; it says nothing about performance.**

---

## End-to-end protocol demo (synthetic evaluation)

To observe the full round trip of "engine requests a proposal → receives the proposal → continues", use the real CLI entry point:

```bash
# Synthetic evaluation: does not touch Slurm, does not compile, needs no database
.venv/bin/python harness_plugins/hloop_cli.py \
    --run-dir /tmp/seqalign-demo \
    --outer-rounds 2 --inner-trials 2 --stub-eval

# Show status only, do not advance the loop
.venv/bin/python harness_plugins/hloop_cli.py --run-dir /tmp/seqalign-demo --status
```

stdout outputs **only** one line of JSON (which can be piped directly to `jq`); all logs go to stderr.
`--stub-eval` explicitly prints `STUB EVAL ACTIVE`: these are **synthetic** evaluation numbers that only prove the control flow and bookkeeping are correct, and are **not** scientific conclusions.
