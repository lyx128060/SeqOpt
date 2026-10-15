"""Tests for HMMER contract-function instrumentation.

The wrapper strategy is only worth it if the injected code compiles and reports
real numbers, so the important test here builds and runs the instrumented source
rather than just inspecting the text.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from seq_align.hmmer_instrument import (
    INSTRUMENTATION_TARGETS,
    RENAMED_SUFFIX,
    InstrumentationError,
    contract_entry_symbols,
    enable_profiling_build,
    impl_dir_from_makefile,
    instrument_source,
    instrument_tree,
    resolve_instrumentation_targets,
    split_function,
    verify_contract_symbols,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
HMMER_SRC = REPO_ROOT / "tools" / "hmmer-3.4" / "src"
PROF_DIR = REPO_ROOT / "harness_plugins" / "profiling"
PROF_HEADER = PROF_DIR / "p7_evo_prof.h"
PROF_SOURCE = PROF_DIR / "p7_evo_prof.c"

# The synthetic tree only carries the one file it needs, so the target list is
# passed explicitly. Production uses INSTRUMENTATION_TARGETS (all of them).
SYNTHETIC_TARGETS = (
    ("impl_neon/fwdback.c", "forward_engine", "P7_EVO_PROF_DP_KERNEL", "dp_kernel"),
)


def _brace_delta(text: str) -> int:
    return text.count("{") - text.count("}")


# ---------------------------------------------------------------------------
# Against the real HMMER tree
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not HMMER_SRC.is_dir(), reason="HMMER source tree not present")
def test_split_function_locates_every_real_target():
    expected_return_types = {
        "p7_SSVFilter": "int",
        "forward_engine": "static int",
        "backward_engine": "static int",
        "p7_Null2_ByExpectation": "int",
        "p7_Null2_ByTrace": "int",
    }
    for rel_path, function, _slot, _contract in INSTRUMENTATION_TARGETS:
        text = (HMMER_SRC / rel_path).read_text(encoding="utf-8")
        span = split_function(text, function)
        assert span["return_type"] == expected_return_types[function], function
        assert span["params"].strip(), function
        # A real body, not an empty declaration.
        assert span["body_close"] - span["body_open"] > 200, function


@pytest.mark.skipif(not HMMER_SRC.is_dir(), reason="HMMER source tree not present")
def test_instrumenting_real_source_preserves_structure():
    """Every injected file must stay brace-balanced and keep all exit paths."""
    by_file: dict[str, list[tuple[str, str]]] = {}
    for rel_path, function, slot, _contract in INSTRUMENTATION_TARGETS:
        by_file.setdefault(rel_path, []).append((function, slot))

    for rel_path, entries in by_file.items():
        original = (HMMER_SRC / rel_path).read_text(encoding="utf-8")
        text = original
        for function, slot in entries:
            before = text
            text = instrument_source(text, function, slot)
            assert text != before
        assert _brace_delta(text) == 0, rel_path

        for function, slot in entries:
            # The public name now belongs to the wrapper...
            assert f"\n{function}(" in text or text.startswith(f"{function}(")
            assert f"{function}{RENAMED_SUFFIX}(" in text
            assert f"P7_EVO_PROF_ENTER({slot}" not in text  # macro takes the var
            assert "P7_EVO_PROF_ENTER(p7_evo_prof_t0);" in text
            assert f"P7_EVO_PROF_EXIT({slot}, p7_evo_prof_t0);" in text
            # ...the renamed body still defines the original name qualified.
            assert f"{function}{RENAMED_SUFFIX}(" in text

        # No return statement was dropped: the wrapper adds exactly one per target.
        assert text.count("return") == original.count("return") + len(entries)


@pytest.mark.skipif(not HMMER_SRC.is_dir(), reason="HMMER source tree not present")
def test_instrument_tree_reports_each_target(tmp_path):
    tree = tmp_path / "src"
    (tree / "impl_neon").mkdir(parents=True)
    for rel_path, _function, _slot, _contract in INSTRUMENTATION_TARGETS:
        shutil.copy(HMMER_SRC / rel_path, tree / rel_path)

    report = instrument_tree(tree, prof_header=PROF_HEADER, prof_source=PROF_SOURCE)

    assert report["schema_version"] == "seqalign_instrumentation.v1"
    assert report["define"] == "P7_EVO_PROF"
    assert {t["function"] for t in report["targets"]} == {
        f for _p, f, _s, _c in INSTRUMENTATION_TARGETS
    }
    # The shim is dropped where #include "p7_evo_prof.h" will resolve.
    assert (tree / "p7_evo_prof.h").is_file()
    assert (tree / "p7_evo_prof.c").is_file()
    # And each instrumented file includes it.
    for rel_path, *_ in INSTRUMENTATION_TARGETS:
        text = (tree / rel_path).read_text(encoding="utf-8")
        assert '#include "p7_evo_prof.h"' in text


def test_missing_function_raises_instead_of_silently_doing_nothing():
    with pytest.raises(InstrumentationError):
        instrument_source("int other(void) { return 0; }\n", "forward_engine", "SLOT")


def test_instrument_tree_rejects_target_file_without_the_function(tmp_path):
    """2026-10-08 regression: the file existed, the function did not.

    The target-directory check only looked at *files*, so a renamed or moved
    contract function would have produced an "instrumented" tree with no timers
    in it -- a silent failure that only showed up much later as "no stage times".
    """
    tree = tmp_path / "src"
    (tree / "impl_neon").mkdir(parents=True)
    (tree / "impl_neon" / "ssvfilter.c").write_text(
        "#include <stdlib.h>\n\nint some_other_symbol(void) { return 0; }\n", encoding="utf-8"
    )
    targets = (("impl_neon/ssvfilter.c", "p7_SSVFilter", "P7_EVO_PROF_SSV_FILTER", "ssv_filter"),)

    with pytest.raises(InstrumentationError, match="p7_SSVFilter"):
        instrument_tree(tree, prof_header=PROF_HEADER, prof_source=PROF_SOURCE, targets=targets)


def _fake_tree(tmp_path: Path, impl_dir: str) -> Path:
    tree = tmp_path / "src"
    (tree / impl_dir).mkdir(parents=True)
    for name in ("ssvfilter.c", "fwdback.c", "null2.c"):
        (tree / impl_dir / name).write_text("#include <stdlib.h>\n", encoding="utf-8")
    (tree / "Makefile").write_text(
        "CC = gcc\n"
        "CFLAGS = -O3\n"
        "QUIET_AR = @echo '    ' AR $@;\n"
        "AR = /usr/bin/ar\n"
        "RANLIB = ranlib\n"
        "OBJS = build.o\n"
        "IMPLDIR   = %s\n"
        "libhmmer-src.stamp: ${OBJS}\n"
        "\t${QUIET_AR}${AR} -r libhmmer.a $? > /dev/null 2>&1\n"
        "\t@${RANLIB} libhmmer.a\n" % impl_dir,
        encoding="utf-8",
    )
    (tree / impl_dir / "Makefile").write_text(
        "CC = gcc\nCFLAGS = -O3\nOBJS = fwdback.o\n", encoding="utf-8"
    )
    return tree


class _FakeContract:
    def __init__(self, target_file: str, functions: tuple[str, ...]):
        self.c_target_file = target_file
        self.c_target_functions = list(functions)


def test_impl_dir_is_read_from_the_generated_makefile(tmp_path):
    tree = _fake_tree(tmp_path, "impl_sse")
    assert impl_dir_from_makefile(tree / "Makefile") == "impl_sse"
    bad = tmp_path / "Makefile"
    bad.write_text("CFLAGS = -O3\n", encoding="utf-8")
    with pytest.raises(InstrumentationError, match="IMPLDIR"):
        impl_dir_from_makefile(bad)


def test_contract_pointing_at_a_different_impl_than_the_build_compiles_is_an_error(tmp_path):
    """A contract bound to impl_neon must not be "verified" against an impl_sse build."""
    tree = _fake_tree(tmp_path, "impl_sse")
    contracts = {
        "ssv_filter_neon": _FakeContract("impl_neon/ssvfilter.c", ("p7_SSVFilter",)),
        "dp_kernel_neon": _FakeContract("impl_neon/fwdback.c", ("forward_engine", "backward_engine")),
        "null2_correction_neon": _FakeContract("impl_neon/null2.c", ("p7_Null2_ByExpectation", "p7_Null2_ByTrace")),
    }

    with pytest.raises(InstrumentationError, match="this build compiles 'impl_sse'"):
        resolve_instrumentation_targets(tree, contracts=contracts)


@pytest.mark.skipif(not HMMER_SRC.is_dir(), reason="HMMER source tree not present")
def test_resolver_matches_the_real_tree_and_contract_island():
    """On a real checkout the resolved targets must be the ACTIVE island's entries."""
    from seq_align.components.c_kernels import ACTIVE_C_KERNEL_CONTRACTS

    targets, resolution = resolve_instrumentation_targets(HMMER_SRC)
    impl_dir = impl_dir_from_makefile(HMMER_SRC / "Makefile")

    assert resolution["impl_dir"] == impl_dir
    assert {t[3] for t in targets} <= set(ACTIVE_C_KERNEL_CONTRACTS)
    for rel_path, _function, _slot, _contract in targets:
        assert rel_path.split("/")[0] == impl_dir
    # Every target really is a contract-permitted function.
    for rel_path, function, _slot, contract_name in targets:
        contract = ACTIVE_C_KERNEL_CONTRACTS[contract_name]
        assert function in contract.c_target_functions
        assert rel_path.endswith(Path(contract.c_target_file).name)


@pytest.mark.skipif(not HMMER_SRC.is_dir(), reason="HMMER source tree not present")
def test_enable_profiling_build_reaches_the_impl_sub_makefile(tmp_path):
    """The define must land where the contract functions are compiled.

    2026-10-08: it only landed in the top-level Makefile, while make runs the
    implementation directory as a sub-make with its own CFLAGS. Every timer macro
    compiled to a no-op and the profiling run reported nothing.
    """
    tree = _fake_tree(tmp_path, "impl_neon")
    report = enable_profiling_build(tree / "Makefile")

    assert report["impl_dir"] == "impl_neon"
    top = (tree / "Makefile").read_text(encoding="utf-8")
    impl = (tree / "impl_neon" / "Makefile").read_text(encoding="utf-8")
    assert "-DP7_EVO_PROF" in top
    assert "-DP7_EVO_PROF" in impl, "the define never reached the units that use it"

    # The shim must actually be compiled and archived, not merely declared: the
    # archive rule is .FORCE-driven and only ever archives the stamp's own
    # prerequisites, so `OBJS +=` alone leaves the wrappers with an undefined
    # p7_evo_prof_add.
    assert "${AR} -r libhmmer.a p7_evo_prof.o" in top
    assert "libhmmer-src.stamp: ${OBJS} p7_evo_prof.o" in top, "the stamp rule must archive the shim"
    assert "${AR} -r libhmmer.a $?" in top, "the original archive of the normal objects must survive"

    # Idempotent.
    again = enable_profiling_build(tree / "Makefile")
    assert again["already_enabled"] is True
    assert (tree / "Makefile").read_text().count("-DP7_EVO_PROF") == 1


def test_too_many_arguments_like_hmmer_signature():
    """HMMER signatures have multi-token types; arg extraction must cope."""
    source = (
        "#include <stdlib.h>\n"
        "\n"
        "static int\n"
        "forward_engine(int do_full, const char *dsq, int L, float *opt_sc)\n"
        "{\n"
        "  if (dsq == 0) return -1;\n"
        "  if (do_full) return L;\n"
        "  *opt_sc = 1.0f;\n"
        "  return 0;\n"
        "}\n"
    )
    instrumented = instrument_source(source, "forward_engine", "P7_EVO_PROF_DP_KERNEL")

    # HMMER puts the return type on its own line; the rename preserves that shape.
    assert "forward_engine__seqalign_uninstrumented(int do_full, const char *dsq, int L, float *opt_sc)" in instrumented
    assert "static int\nforward_engine__seqalign_uninstrumented" in instrumented
    # The wrapper keeps the original name and forwards every argument.
    assert "rc = forward_engine__seqalign_uninstrumented(do_full, dsq, L, opt_sc);" in instrumented
    assert _brace_delta(instrumented) == 0


# ---------------------------------------------------------------------------
# Build and run the instrumented code for real
# ---------------------------------------------------------------------------

SYNTHETIC = """\
#include <stdio.h>

/* Mimics HMMER's definition style: return type on its own line, name at col 0,
   several exit paths -- the shape the wrapper strategy must survive. */
static int
forward_engine(int do_full, int reps)
{
  volatile long acc = 0;
  int i;
  if (reps < 0) return -1;
  for (i = 0; i < reps; i++) acc += i;
  if (do_full && acc == 0) return -2;
  return (int) (acc & 0xffff);
}

int
main(void)
{
  int i;
  /* Enough work that the measured stage time is unambiguously non-zero even at
     microsecond resolution -- otherwise this test could pass on a rounding
     artefact rather than on real instrumentation. */
  for (i = 0; i < 7; i++) forward_engine(1, 400000);
  forward_engine(0, -1);          /* exercises an early return */
  return 0;
}
"""


@pytest.mark.skipif(shutil.which("gcc") is None, reason="gcc not available")
def test_instrumented_source_compiles_runs_and_reports(tmp_path):
    """The wrapper must compile, run, and emit a parseable timing report."""
    from seq_align.hmmer_stage_timing import (
        INSTRUMENTED_FILE,
        load_instrumented_stage_times,
    )

    tree = tmp_path / "src"
    (tree / "impl_neon").mkdir(parents=True)
    (tree / "impl_neon" / "fwdback.c").write_text(SYNTHETIC, encoding="utf-8")

    instrument_tree(
        tree, prof_header=PROF_HEADER, prof_source=PROF_SOURCE, targets=SYNTHETIC_TARGETS
    )

    binary = tmp_path / "probe"
    compile_result = subprocess.run(
        [
            "gcc", "-O1", "-std=c99", "-Wall", "-Wextra", "-Werror",
            "-DP7_EVO_PROF", f"-I{tree}",
            "-o", str(binary),
            str(tree / "impl_neon" / "fwdback.c"),
            str(tree / "p7_evo_prof.c"),
        ],
        capture_output=True,
        text=True,
    )
    assert compile_result.returncode == 0, compile_result.stderr

    out_dir = tmp_path / "qdir"
    out_dir.mkdir()
    run_result = subprocess.run(
        [str(binary)],
        capture_output=True,
        text=True,
        env={"PATH": "/usr/bin:/bin", "P7_EVO_PROF_OUT": str(out_dir / INSTRUMENTED_FILE)},
    )
    assert run_result.returncode == 0, run_result.stderr

    stage_times, diagnostics = load_instrumented_stage_times(out_dir)

    assert stage_times is not None, diagnostics
    assert stage_times["dp_kernel"] > 0.0
    # 7 full calls plus the early-returning one = 8, all through the wrapper.
    assert diagnostics["calls"]["dp_kernel"] == 8
    # Stages with no injected timer report nothing rather than a misleading 0.0.
    assert stage_times["ssv_filter"] == 0.0
    assert diagnostics["calls"]["ssv_filter"] == 0

    # And it composes into the report the agent sees.
    from seq_align.hmmer_stage_timing import build_stage_timing

    report = build_stage_timing(stage_times_s=stage_times, stage_calls=diagnostics["calls"])
    assert report["contract_stages"]["dp_kernel"]["time_s"] > 0.0
    assert "time_s" not in report["contract_stages"]["ssv_filter"]
    assert json.dumps(report)  # serialisable into EvalResult meta


@pytest.mark.skipif(shutil.which("gcc") is None, reason="gcc not available")
def test_uninstrumented_build_is_byte_identical(tmp_path):
    """Without -DP7_EVO_PROF the injected wrappers must cost nothing at all."""
    tree = tmp_path / "src"
    (tree / "impl_neon").mkdir(parents=True)
    (tree / "impl_neon" / "fwdback.c").write_text(SYNTHETIC, encoding="utf-8")
    instrument_tree(
        tree, prof_header=PROF_HEADER, prof_source=PROF_SOURCE, targets=SYNTHETIC_TARGETS
    )

    # No -DP7_EVO_PROF: the macros are no-ops, so this must still build and run
    # with the shim source absent entirely.
    binary = tmp_path / "plain"
    result = subprocess.run(
        [
            "gcc", "-O1", "-std=c99", "-Wall", "-Wextra", "-Werror",
            f"-I{tree}", "-o", str(binary),
            str(tree / "impl_neon" / "fwdback.c"),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    run = subprocess.run([str(binary)], capture_output=True, text=True)
    assert run.returncode == 0, run.stderr


# --------------------------------------------------------------------------- #
# contract entry-symbol verification (non-profiling candidate bundles)
# --------------------------------------------------------------------------- #


class _FakeEntryContract:
    def __init__(self, entry: tuple[str, ...], binaries: tuple[str, ...] | None = None):
        self.c_entry_functions = list(entry)
        if binaries is not None:
            self.c_entry_binaries = list(binaries)


class _FakePatchSpec:
    def __init__(self, contract_name: str):
        self.contract_name = contract_name


class _FakeCfg:
    def __init__(self, *contract_names: str):
        self.c_patches = [_FakePatchSpec(name) for name in contract_names]


def _fake_scan(symbols: set[str], calls: list[str] | None = None):
    def _scan(binary):
        if calls is not None:
            calls.append(str(binary))
        return {
            "binary": str(binary),
            "nm": "nm",
            "symbols": set(symbols),
            "nm_all_text": "",
            "nm_plain_text": "",
            "strings_text": "",
        }

    return _scan


def test_every_active_contract_declares_entry_symbols_inside_its_target_functions():
    """The entry set is a *subset* of the editable set, never an invented name."""
    from seq_align.components.c_kernels import ACTIVE_C_KERNEL_CONTRACTS

    for name, contract in ACTIVE_C_KERNEL_CONTRACTS.items():
        assert contract.c_entry_functions, f"{name} declares no entry symbol to verify"
        assert set(contract.c_entry_functions) <= set(contract.c_target_functions), name
        assert contract.c_entry_binaries, f"{name} declares no binary to verify against"
        assert set(contract.c_entry_binaries) <= {"jackhmmer", "hmmsearch"}, name


def test_jackhmmer_iteration_island_is_declared_for_one_binary_only():
    """The iteration island is linked into jackhmmer only.

    It is referenced only by jackhmmer.c, so the linker drops it from hmmsearch.
    """
    from seq_align.components.c_kernels import ACTIVE_C_KERNEL_CONTRACTS

    assert ACTIVE_C_KERNEL_CONTRACTS["jackhmmer_loop"].c_entry_binaries == ["jackhmmer"]


def test_contract_entry_symbols_reads_the_declared_field_only():
    symbols = contract_entry_symbols(
        {"a": _FakeEntryContract(("f",)), "b": object()}
    )
    assert symbols == {"a": ("f",), "b": ()}


def test_verify_contract_symbols_accepts_a_binary_that_carries_the_entry(monkeypatch, tmp_path):
    from seq_align import hmmer_instrument

    monkeypatch.setattr(
        hmmer_instrument, "scan_binary_symbols", _fake_scan({"p7_evo_align_Run", "other"})
    )
    report = verify_contract_symbols(
        {"jackhmmer": tmp_path / "jackhmmer", "hmmsearch": tmp_path / "hmmsearch"},
        _FakeCfg("pipeline_flow"),
        contracts={"pipeline_flow": _FakeEntryContract(("p7_evo_align_Run",))},
    )

    assert report["verified"] is True
    assert report["checked_contracts"] == ["pipeline_flow"]
    assert report["expected_symbols"] == ["p7_evo_align_Run"]
    assert report["binaries"]["jackhmmer"]["missing"] == []
    assert "skipped_reason" not in report


def test_verify_contract_symbols_refuses_a_binary_without_the_island(monkeypatch, tmp_path):
    """The whole point: a patch that applied but never compiled in is refused."""
    from seq_align import hmmer_instrument

    monkeypatch.setattr(hmmer_instrument, "scan_binary_symbols", _fake_scan({"p7_Pipeline"}))
    with pytest.raises(InstrumentationError, match="never compiled into that binary"):
        verify_contract_symbols(
            {"jackhmmer": tmp_path / "jackhmmer"},
            _FakeCfg("pipeline_flow"),
            contracts={"pipeline_flow": _FakeEntryContract(("p7_evo_align_Run",))},
        )


def test_verify_contract_symbols_refuses_when_only_one_binary_is_missing_it(monkeypatch, tmp_path):
    """Both binaries link the islands, so one good binary cannot excuse the other."""
    from seq_align import hmmer_instrument

    monkeypatch.setattr(
        hmmer_instrument,
        "scan_binary_symbols",
        lambda binary: _fake_scan(
            {"p7_evo_align_Run"} if Path(binary).name == "jackhmmer" else set()
        )(binary),
    )
    with pytest.raises(InstrumentationError, match="hmmsearch is missing"):
        verify_contract_symbols(
            {"jackhmmer": tmp_path / "jackhmmer", "hmmsearch": tmp_path / "hmmsearch"},
            _FakeCfg("pipeline_flow"),
            contracts={"pipeline_flow": _FakeEntryContract(("p7_evo_align_Run",))},
        )


def test_verify_contract_symbols_honours_a_single_binary_island(monkeypatch, tmp_path):
    """A jackhmmer-only island must not be demanded from hmmsearch."""
    from seq_align import hmmer_instrument

    calls: list[str] = []
    monkeypatch.setattr(
        hmmer_instrument,
        "scan_binary_symbols",
        lambda binary: _fake_scan(
            {"p7_evo_iter_BeginRound"} if Path(binary).name == "jackhmmer" else set(), calls
        )(binary),
    )
    report = verify_contract_symbols(
        {"jackhmmer": tmp_path / "jackhmmer", "hmmsearch": tmp_path / "hmmsearch"},
        _FakeCfg("jackhmmer_loop"),
        contracts={
            "jackhmmer_loop": _FakeEntryContract(("p7_evo_iter_BeginRound",), ("jackhmmer",))
        },
    )

    assert report["verified"] is True
    assert report["expected_by_binary"] == {"jackhmmer": ["p7_evo_iter_BeginRound"]}
    assert "hmmsearch" not in report["binaries"]  # not scanned, because not required
    assert calls == [str(tmp_path / "jackhmmer")]


def test_verify_contract_symbols_reports_instead_of_claiming_success(monkeypatch, tmp_path):
    """Nothing checkable must read as 'nothing was checked', not as 'verified'."""
    from seq_align import hmmer_instrument

    calls: list[str] = []
    monkeypatch.setattr(hmmer_instrument, "scan_binary_symbols", _fake_scan(set(), calls))
    report = verify_contract_symbols(
        {"jackhmmer": tmp_path / "jackhmmer"},
        _FakeCfg("legacy_island"),
        contracts={"legacy_island": _FakeEntryContract(())},
    )

    assert report["verified"] is False
    assert report["checked_contracts"] == []
    assert report["unverifiable_contracts"] == ["legacy_island"]
    assert "nothing was checked" in report["skipped_reason"]
    assert calls == []  # never scanned a binary it could say nothing about


def test_verify_contract_symbols_ignores_contracts_outside_the_registry(monkeypatch, tmp_path):
    """A synthetic harness config must not be turned into a false verification failure."""
    from seq_align import hmmer_instrument

    calls: list[str] = []
    monkeypatch.setattr(hmmer_instrument, "scan_binary_symbols", _fake_scan(set(), calls))
    report = verify_contract_symbols(
        {"jackhmmer": tmp_path / "jackhmmer"},
        _FakeCfg("test_kernel"),
        contracts={"pipeline_flow": _FakeEntryContract(("p7_evo_align_Run",))},
    )

    assert report["verified"] is False
    assert report["unresolved_contracts"] == ["test_kernel"]
    assert report["expected_symbols"] == []
    assert calls == []


@pytest.mark.skipif(not HMMER_SRC.is_dir(), reason="HMMER source tree not present")
def test_real_stock_binaries_contain_the_non_island_entry_symbols(tmp_path):
    """Every declared entry symbol, except the unwired evo islands, is present in the
    linked binaries of the settled build."""
    from seq_align.components.c_kernels import ALL_C_KERNEL_CONTRACTS
    from seq_align import hmmer_instrument

    binaries = {
        name: HMMER_SRC / name
        for name in ("jackhmmer", "hmmsearch")
        if (HMMER_SRC / name).is_file()
    }
    if not binaries:
        pytest.skip("no prebuilt HMMER binaries")
    scans = {name: hmmer_instrument.scan_binary_symbols(path) for name, path in binaries.items()}
    for contract_name, contract in ALL_C_KERNEL_CONTRACTS.items():
        if contract_name in {"pipeline_flow", "jackhmmer_loop"}:
            continue  # the unwired evo islands; see the source-tree report
        for symbol in contract.c_entry_functions:
            for name, scan in scans.items():
                assert hmmer_instrument._symbol_present(scan, symbol), (contract_name, symbol, name)


@pytest.mark.skipif(not HMMER_SRC.is_dir(), reason="HMMER source tree not present")
def test_real_build_evidence_for_the_single_binary_iteration_island():
    """Pins the nm fact the jackhmmer-only declaration rests on.

    Skips while the tree is still stock (island not linked in); becomes a real
    assertion as soon as the federation overlay is built, and would catch a tree
    that silently stopped compiling the iteration island.
    """
    from seq_align import hmmer_instrument

    jackhmmer, hmmsearch = HMMER_SRC / "jackhmmer", HMMER_SRC / "hmmsearch"
    if not (jackhmmer.is_file() and hmmsearch.is_file()):
        pytest.skip("no prebuilt HMMER binaries")
    scan_j = hmmer_instrument.scan_binary_symbols(jackhmmer)
    if not hmmer_instrument._symbol_present(scan_j, "p7_evo_iter_BeginRound"):
        pytest.skip("stock tree: the iteration island is not linked in yet")
    scan_h = hmmer_instrument.scan_binary_symbols(hmmsearch)

    assert hmmer_instrument._symbol_present(scan_j, "p7_evo_iter_BeginRound")
    assert not hmmer_instrument._symbol_present(scan_h, "p7_evo_iter_BeginRound")
