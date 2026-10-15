"""Make ``import seq_align...`` work from a plain ``pytest tests/`` checkout.

`workspace/` is the sys.path root for the seq_align package and `harness_plugins/`
is the root for the callback CLI helpers (`stub_runner`, `hloop_cli`). The
production entrypoints bootstrap these paths themselves; doing it here means the
test suite needs no environment setup.
"""

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent
for _path in (ROOT / "workspace", ROOT / "harness_plugins"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

# Every knob the engine reads from the environment. Sourcing `env.sh` exports
# these into the operator's shell -- and therefore into any test run started from
# that shell -- which makes results depend on who ran the suite:
#
#   * ``SEQ_ALIGN_EVAL_BACKEND=slurm`` silently moves every ``PatchRunner`` that
#     does not pass an explicit backend onto the Slurm path, so a test that stubs
#     ``run_hmmer_cli.run_all_queries`` instead reaches the real submitter;
#   * ``SEQ_ALIGN_SLURM_*`` changes the core/concurrency shape a submission records;
#   * the screen and profiling policies change which evaluations run at all.
#
# An autouse fixture removes them so a test starts from the code's own defaults.
# A test that needs a specific value sets it with its own ``monkeypatch``, and the
# test body runs after fixture setup, so the explicit value wins and is still
# undone at teardown.
_AMBIENT_PREFIXES = ("SEQ_ALIGN_",)

# The engine's default search database is the 60.9 GB UniRef90 file, and a test that
# builds a runner without naming one fingerprints it -- minutes of hashing per test.
# Comparisons and checks therefore run against a small (288 MB) UniProt/SwissProt
# database; UniRef90 remains the database for baselines and for the numbers a run
# reports. Point ``SEQALIGN_TEST_UNIREF_DB`` at your own copy of that file. When it is
# absent the affected tests are skipped rather than silently fingerprinted against
# the 60.9 GB default.
TEST_UNIREF_DB = os.environ.get(
    "SEQALIGN_TEST_UNIREF_DB", str(ROOT / "data" / "uniprot_sprot.fasta")
)


@pytest.fixture(autouse=True)
def _use_the_small_test_database(monkeypatch: pytest.MonkeyPatch) -> None:
    from seq_align import run_jack

    monkeypatch.setenv("UNIREF_DB", TEST_UNIREF_DB)
    monkeypatch.setattr(run_jack, "UNIREF_DB", TEST_UNIREF_DB, raising=False)


@pytest.fixture(autouse=True)
def _isolate_seq_align_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in sorted(os.environ):
        if name.startswith(_AMBIENT_PREFIXES):
            monkeypatch.delenv(name, raising=False)


# --------------------------------------------------------------------------- #
# Optional heavy prerequisites
# --------------------------------------------------------------------------- #
# Some tests read a real HMMER source tree: they synthesise diffs against the
# files the component contracts point at, so without the tree they would produce
# an empty diff and fail for a reason that has nothing to do with the code under
# test. The tree is deliberately NOT vendored (98 MB, and HMMER's own BSD-3
# licence should travel with HMMER rather than with this repository), so those
# tests declare `requires_hmmer_source` and are skipped when it is absent.
#
# To enable them:  bash scripts/setup_hmmer.sh
def _hmmer_source_root() -> Path:
    from seq_align.run_hmmer_cli import DEFAULT_HMMER_SRC_ROOT

    return Path(DEFAULT_HMMER_SRC_ROOT)


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "requires_hmmer_source: needs an HMMER 3.4 source tree under tools/hmmer-3.4/",
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if _hmmer_source_root().is_dir():
        return
    skip = pytest.mark.skip(
        reason=(
            "HMMER 3.4 source tree not present under tools/hmmer-3.4/; "
            "run `bash scripts/setup_hmmer.sh` to enable this test"
        )
    )
    for item in items:
        if "requires_hmmer_source" in item.keywords:
            item.add_marker(skip)

