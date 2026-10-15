"""Regression net for the built-in patch applier and the git fallback.

Context: `_apply_patches` used to require `git` on PATH. On the target cluster
git exists only on the login node, so the compile step inside the Slurm job
failed with a bare `FileNotFoundError: 'git'`. `unified_diff` is the stdlib
replacement; these tests pin its semantics and, where git is installed, assert
that it produces byte-identical files to `git apply --recount`.

Two divergences from git are pinned deliberately (see `_git_apply` probes):
git refuses a hunk whose last body line is an addition unless `--unidiff-zero`
is passed, and git renumbers a second `--- /+++` section against the file as
already modified. The fallback accepts both, which is why it can rescue a patch
git rejects. It is never *less* careful about context.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from seq_align import run_hmmer_cli
from seq_align.assembler import AssembledConfig
from seq_align.components.base import ComponentKind, ComponentSpec
from seq_align.unified_diff import PatchApplyError, apply_unified_diff, parse_unified_diff

SOURCE = """\
#include <stdio.h>

static int calc_band_1(const float *dsq, int M, int N)
{
    int band = 0;
    for (int i = 0; i < M; i++) {
        band += (int) dsq[i];
    }
    return band;
}

int p7_SSVFilter(const float *dsq, int M, int N)
{
    return calc_band_1(dsq, M, N) > 0;
}
"""

OTHER_SOURCE = """\
int other_helper(int value)
{
    return value;
}
"""

# One section, one hunk. Header counts are accurate; every hunk ends with a
# context line, which is what real `git diff` output always does.
SIMPLE_HUNK = (
    "--- a/src/kernel.c\n"
    "+++ b/src/kernel.c\n"
    "@@ -3,6 +3,7 @@\n"
    " static int calc_band_1(const float *dsq, int M, int N)\n"
    " {\n"
    "     int band = 0;\n"
    "+    int probe = 1;\n"
    "     for (int i = 0; i < M; i++) {\n"
    "         band += (int) dsq[i];\n"
    "     }\n"
)

# Second hunk of the same section: `+` side is 13 because the first hunk added
# a line, `-` side stays 12 because hunks are numbered against the original.
TAIL_HUNK = (
    "@@ -12,4 +13,4 @@\n"
    " int p7_SSVFilter(const float *dsq, int M, int N)\n"
    " {\n"
    "-    return calc_band_1(dsq, M, N) > 0;\n"
    "+    return calc_band_1(dsq, M, N) > 1;\n"
    " }\n"
)

TWO_HUNKS = SIMPLE_HUNK + TAIL_HUNK

# A hunk whose last body line is an addition. git needs `--unidiff-zero`.
ADDITION_TAILED_HUNK = (
    "--- a/src/kernel.c\n"
    "+++ b/src/kernel.c\n"
    "@@ -3,3 +3,4 @@\n"
    " static int calc_band_1(const float *dsq, int M, int N)\n"
    " {\n"
    "     int band = 0;\n"
    "+    int probe = 1;\n"
)

OTHER_HUNK = (
    "--- a/src/other.c\n"
    "+++ b/src/other.c\n"
    "@@ -1,3 +1,4 @@\n"
    " int other_helper(int value)\n"
    " {\n"
    "+    /* touched */\n"
    "     return value;\n"
)

# Replaces the whole body of a file with an insertion; the blank source line is
# context, written as a single space (never an actually empty line).
BLANK_CONTEXT_HUNK = (
    "--- a/src/kernel.c\n"
    "+++ b/src/kernel.c\n"
    "@@ -11,2 +11,3 @@\n"
    " \n"
    "+/* inserted */\n"
    " int p7_SSVFilter(const float *dsq, int M, int N)\n"
)


def _write_source(root: Path, text: str = SOURCE, relative: str = "src/kernel.c") -> Path:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _cfg(patch: str, target_file: str = "kernel.c") -> AssembledConfig:
    cfg = AssembledConfig()
    cfg.c_patches = [
        ComponentSpec(
            contract_name="test_kernel",
            kind=ComponentKind.C_PATCH,
            c_patch=patch,
            c_target_file=target_file,
            c_target_functions=["calc_band_1"],
        )
    ]
    cfg.needs_recompile = True
    return cfg


def _git_apply(root: Path, patch_text: str) -> bool:
    patch_file = root.parent / "probe.patch"
    patch_file.parent.mkdir(parents=True, exist_ok=True)
    patch_file.write_text(patch_text, encoding="utf-8", newline="\n")
    proc = subprocess.run(
        ["git", "apply", "--recount", str(patch_file)],
        cwd=str(root), capture_output=True, text=True,
    )
    return proc.returncode == 0


def _no_git(monkeypatch) -> None:
    real_which = shutil.which
    monkeypatch.setattr(
        run_hmmer_cli.shutil,
        "which",
        lambda name, *args, **kwargs: None if name == "git" else real_which(name, *args, **kwargs),
    )


requires_git = pytest.mark.skipif(shutil.which("git") is None, reason="git is not installed")


# --------------------------------------------------------------------------
# parse
# --------------------------------------------------------------------------


def test_parse_splits_hunks_within_a_section():
    sections = parse_unified_diff(TWO_HUNKS)
    assert len(sections) == 1
    assert sections[0].new_path == "b/src/kernel.c"
    assert len(sections[0].hunks) == 2
    assert sections[0].hunks[1].old_start == 12
    assert sections[0].hunks[1].new_start == 13


def test_parse_splits_two_sections():
    sections = parse_unified_diff(SIMPLE_HUNK + OTHER_HUNK)
    assert [section.new_path for section in sections] == ["b/src/kernel.c", "b/src/other.c"]


def test_parse_accepts_header_without_counts():
    sections = parse_unified_diff("--- a/src/k.c\n+++ b/src/k.c\n@@ -1 +1 @@\n-int a = 1;\n+int a = 2;\n")
    assert sections[0].hunks[0].old_count == 1
    assert sections[0].hunks[0].new_count == 1


def test_parse_tolerates_a_blank_separator_between_sections():
    sections = parse_unified_diff(SIMPLE_HUNK + "\n" + OTHER_HUNK)
    assert len(sections[0].hunks) == 1
    assert len(sections[1].hunks) == 1


def test_parse_rejects_header_without_plus_line():
    with pytest.raises(PatchApplyError, match="not followed by a"):
        parse_unified_diff("--- a/src/k.c\n@@ -1 +1 @@\n-a\n+b\n")


def test_parse_rejects_empty_patch():
    with pytest.raises(PatchApplyError, match="no file sections"):
        apply_unified_diff("", Path("."))


def test_rejects_hunk_with_no_body(tmp_path):
    """A model that runs out of output mid-hunk leaves the header with no body."""
    _write_source(tmp_path)
    with pytest.raises(PatchApplyError, match="empty body"):
        apply_unified_diff("--- a/src/kernel.c\n+++ b/src/kernel.c\n@@ -3,6 +3,7 @@\n", tmp_path)


# --------------------------------------------------------------------------
# apply
# --------------------------------------------------------------------------


def test_applies_simple_hunk(tmp_path):
    target = _write_source(tmp_path)
    touched = apply_unified_diff(SIMPLE_HUNK, tmp_path)
    assert touched == ["src/kernel.c"]
    text = target.read_text(encoding="utf-8")
    assert text.count("int probe = 1;") == 1


def test_applies_multiple_hunks_with_correct_offset_bookkeeping(tmp_path):
    target = _write_source(tmp_path)
    apply_unified_diff(TWO_HUNKS, tmp_path)
    text = target.read_text(encoding="utf-8")
    assert "int probe = 1;" in text
    assert "return calc_band_1(dsq, M, N) > 1;" in text
    assert "return calc_band_1(dsq, M, N) > 0;" not in text


def test_applies_to_multiple_files(tmp_path):
    _write_source(tmp_path)
    _write_source(tmp_path, OTHER_SOURCE, relative="src/other.c")
    touched = apply_unified_diff(SIMPLE_HUNK + OTHER_HUNK, tmp_path)
    assert touched == ["src/kernel.c", "src/other.c"]
    assert "/* touched */" in (tmp_path / "src" / "other.c").read_text(encoding="utf-8")


def test_recounts_wrong_hunk_header_counts(tmp_path):
    # Counts deliberately wrong: --recount semantics means the body wins.
    patch = SIMPLE_HUNK.replace("@@ -3,6 +3,7 @@", "@@ -3,99 +3,99 @@")
    target = _write_source(tmp_path)
    apply_unified_diff(patch, tmp_path)
    assert "int probe = 1;" in target.read_text(encoding="utf-8")


def test_tolerates_line_offset_drift(tmp_path):
    """Model-written diffs routinely carry stale line numbers."""
    patch = SIMPLE_HUNK.replace("@@ -3,6 +3,7 @@", "@@ -103,6 +104,7 @@")
    target = _write_source(tmp_path)
    apply_unified_diff(patch, tmp_path)
    assert "int probe = 1;" in target.read_text(encoding="utf-8")


def test_handles_a_blank_context_line(tmp_path):
    target = _write_source(tmp_path)
    apply_unified_diff(BLANK_CONTEXT_HUNK, tmp_path)
    lines = target.read_text(encoding="utf-8").split("\n")
    assert lines[10] == ""
    assert lines[11] == "/* inserted */"
    assert lines[12] == "int p7_SSVFilter(const float *dsq, int M, int N)"


def test_resolves_paths_without_src_prefix(tmp_path):
    _write_source(tmp_path)
    patch = SIMPLE_HUNK.replace("a/src/kernel.c", "a/kernel.c").replace("b/src/kernel.c", "b/kernel.c")
    assert apply_unified_diff(patch, tmp_path) == ["src/kernel.c"]


def test_handles_pure_insertion_hunk(tmp_path):
    """`@@ -N,0 +M,K @@` inserts after the recorded line N.

    git places a zero-context hunk at the end of the file instead (probed with
    git 2.34); the recorded line is the documented meaning of the format, so the
    fallback follows it.
    """
    _write_source(tmp_path)
    patch = "--- a/src/kernel.c\n+++ b/src/kernel.c\n@@ -10,0 +11,1 @@\n+/* inserted */\n"
    target = _write_source(tmp_path, SOURCE)
    apply_unified_diff(patch, tmp_path)
    lines = target.read_text(encoding="utf-8").split("\n")
    assert lines[10] == "/* inserted */"
    assert lines[11] == ""


def test_preserves_trailing_newline(tmp_path):
    target = _write_source(tmp_path)
    assert target.read_text(encoding="utf-8").endswith("}\n")
    apply_unified_diff(SIMPLE_HUNK, tmp_path)
    assert target.read_text(encoding="utf-8").endswith("}\n")


def test_second_section_of_the_same_file_uses_original_line_numbers(tmp_path):
    """git renumbers a second section against the already-patched file; we do
    not, because `_apply_patches` concatenates one independently-numbered diff
    per component."""
    target = _write_source(tmp_path)
    apply_unified_diff(SIMPLE_HUNK + TAIL_HUNK.replace("@@ -12,4 +13,4 @@", "--- a/src/kernel.c\n+++ b/src/kernel.c\n@@ -12,4 +13,4 @@"), tmp_path)
    text = target.read_text(encoding="utf-8")
    assert "int probe = 1;" in text
    assert "return calc_band_1(dsq, M, N) > 1;" in text


# --------------------------------------------------------------------------
# refusals -- a wrong C patch must fail loudly, never silently
# --------------------------------------------------------------------------


def test_rejects_context_mismatch(tmp_path):
    _write_source(tmp_path)
    patch = SIMPLE_HUNK.replace("    int band = 0;", "    int bandwidth = 0;")
    with pytest.raises(PatchApplyError, match="does not match this source tree"):
        apply_unified_diff(patch, tmp_path)


def test_leaves_file_untouched_when_a_later_hunk_fails(tmp_path):
    """Nothing is written unless every section parses and applies."""
    target = _write_source(tmp_path)
    before = target.read_text(encoding="utf-8")
    broken = SIMPLE_HUNK + TAIL_HUNK.replace("int p7_SSVFilter", "int p7_UnknownFilter")
    with pytest.raises(PatchApplyError):
        apply_unified_diff(broken, tmp_path)
    assert target.read_text(encoding="utf-8") == before


def test_rejects_generic_context_far_from_recorded_position(tmp_path):
    """`}` / `}` / `}` is not a fingerprint: never relocate on it."""
    _write_source(tmp_path, "int p7_SSVFilter(void)\n{\n    return 0;\n}\n}\n}\n}\n}\n")
    patch = "--- a/src/kernel.c\n+++ b/src/kernel.c\n@@ -40,3 +40,4 @@\n }\n }\n+}\n }\n"
    with pytest.raises(PatchApplyError, match="too generic to relocate safely"):
        apply_unified_diff(patch, tmp_path)


def test_refuses_ambiguous_context(tmp_path):
    block = ["static int calc_band_1_x(void)\n", "{\n", "    return 7;\n"]
    lines = ["int a(void);\n", "int b(void);\n"] + block + ["int c(void);\n"] + block + ["int d(void);\n"]
    _write_source(tmp_path, "".join(lines))
    patch = (
        "--- a/src/kernel.c\n+++ b/src/kernel.c\n"
        "@@ -5,3 +5,3 @@\n"
        " static int calc_band_1_x(void)\n"
        " {\n"
        "-    return 7;\n"
        "+    return 8;\n"
    )
    with pytest.raises(PatchApplyError, match="equally"):
        apply_unified_diff(patch, tmp_path)


def test_rejects_missing_file(tmp_path):
    _write_source(tmp_path)
    patch = SIMPLE_HUNK.replace("src/kernel.c", "src/absent.c")
    with pytest.raises(PatchApplyError, match="does not exist"):
        apply_unified_diff(patch, tmp_path)


def test_rejects_absolute_path(tmp_path):
    _write_source(tmp_path)
    patch = SIMPLE_HUNK.replace("a/src/kernel.c", "/etc/kernel.c").replace("b/src/kernel.c", "/etc/kernel.c")
    with pytest.raises(PatchApplyError, match="absolute paths"):
        apply_unified_diff(patch, tmp_path)


def test_rejects_parent_traversal(tmp_path):
    _write_source(tmp_path)
    patch = SIMPLE_HUNK.replace("a/src/kernel.c", "a/../../kernel.c").replace("b/src/kernel.c", "b/../../kernel.c")
    with pytest.raises(PatchApplyError, match="outside the source tree"):
        apply_unified_diff(patch, tmp_path)


def test_rejects_file_creation(tmp_path):
    _write_source(tmp_path)
    patch = SIMPLE_HUNK.replace("--- a/src/kernel.c", "--- /dev/null")
    with pytest.raises(PatchApplyError, match="creating or deleting files"):
        apply_unified_diff(patch, tmp_path)


def test_rejects_file_deletion(tmp_path):
    _write_source(tmp_path)
    patch = SIMPLE_HUNK.replace("+++ b/src/kernel.c", "+++ /dev/null")
    with pytest.raises(PatchApplyError, match="creating or deleting files"):
        apply_unified_diff(patch, tmp_path)


# --------------------------------------------------------------------------
# the decisive check: the fallback must agree with git byte for byte
# --------------------------------------------------------------------------

_GIT_CASES = {
    "simple": SIMPLE_HUNK,
    "two_hunks": TWO_HUNKS,
    "recount": SIMPLE_HUNK.replace("@@ -3,6 +3,7 @@", "@@ -3,99 +3,99 @@"),
    "stale_line_numbers": SIMPLE_HUNK.replace("@@ -3,6 +3,7 @@", "@@ -103,6 +104,7 @@"),
    "blank_context_line": BLANK_CONTEXT_HUNK,
}


@requires_git
@pytest.mark.parametrize("name", sorted(_GIT_CASES))
def test_fallback_matches_git_apply_byte_for_byte(tmp_path, name):
    patch = _GIT_CASES[name]

    git_root = tmp_path / "git" / "hmmer-src"
    _write_source(git_root)
    _write_source(git_root, OTHER_SOURCE, relative="src/other.c")
    assert _git_apply(git_root, patch)

    python_root = tmp_path / "python" / "hmmer-src"
    _write_source(python_root)
    _write_source(python_root, OTHER_SOURCE, relative="src/other.c")
    apply_unified_diff(patch, python_root)

    for relative in ("src/kernel.c", "src/other.c"):
        assert (python_root / relative).read_bytes() == (git_root / relative).read_bytes()


@requires_git
def test_fallback_covers_a_patch_git_refuses(tmp_path):
    """Divergence worth keeping: git wants a trailing context line (or
    `--unidiff-zero`) and renumbers repeat sections; we accept both."""
    git_root = tmp_path / "git" / "hmmer-src"
    _write_source(git_root)
    assert not _git_apply(git_root, ADDITION_TAILED_HUNK)

    python_root = tmp_path / "python" / "hmmer-src"
    _write_source(python_root)
    apply_unified_diff(ADDITION_TAILED_HUNK, python_root)
    assert "int probe = 1;" in (python_root / "src" / "kernel.c").read_text(encoding="utf-8")


@requires_git
def test_git_recount_swallows_the_next_sections_header(tmp_path):
    """Why a two-component patch only applies through the fallback.

    `git apply --recount` scans a hunk body until it meets a line that cannot be
    one, and `--- a/src/other.c` can be one (a removal whose content is
    `-- a/src/other.c`). `_normalize_patch_paths` emits bare `--- `/`+++ `
    sections with no `diff --git` line, so git cannot apply them.
    """
    git_root = tmp_path / "git" / "hmmer-src"
    _write_source(git_root)
    _write_source(git_root, OTHER_SOURCE, relative="src/other.c")
    assert not _git_apply(git_root, SIMPLE_HUNK + OTHER_HUNK)

    python_root = tmp_path / "python" / "hmmer-src"
    _write_source(python_root)
    _write_source(python_root, OTHER_SOURCE, relative="src/other.c")
    assert apply_unified_diff(SIMPLE_HUNK + OTHER_HUNK, python_root) == ["src/kernel.c", "src/other.c"]
    assert "int probe = 1;" in (python_root / "src" / "kernel.c").read_text(encoding="utf-8")
    assert "/* touched */" in (python_root / "src" / "other.c").read_text(encoding="utf-8")


# --------------------------------------------------------------------------
# _apply_patches: tool selection
# --------------------------------------------------------------------------


def _prepare(tmp_path: Path) -> Path:
    build_root = tmp_path / "hmmer-src"
    _write_source(build_root)
    return build_root


@requires_git
def test_apply_patches_uses_git_when_available(tmp_path, monkeypatch):
    monkeypatch.delenv("SEQ_ALIGN_PATCH_TOOL", raising=False)
    build_root = _prepare(tmp_path)
    assert run_hmmer_cli._apply_patches(build_root, _cfg(SIMPLE_HUNK)) == "git"
    assert "int probe = 1;" in (build_root / "src" / "kernel.c").read_text(encoding="utf-8")


def test_apply_patches_falls_back_when_git_is_absent(tmp_path, monkeypatch):
    """The cluster case: no git on the compute node."""
    monkeypatch.delenv("SEQ_ALIGN_PATCH_TOOL", raising=False)
    _no_git(monkeypatch)
    build_root = _prepare(tmp_path)
    assert run_hmmer_cli._apply_patches(build_root, _cfg(SIMPLE_HUNK)) == "python"
    assert "int probe = 1;" in (build_root / "src" / "kernel.c").read_text(encoding="utf-8")


def test_fallback_and_git_agree_through_apply_patches(tmp_path, monkeypatch):
    monkeypatch.delenv("SEQ_ALIGN_PATCH_TOOL", raising=False)
    if shutil.which("git") is None:
        pytest.skip("git is not installed")

    git_root = tmp_path / "a" / "hmmer-src"
    _write_source(git_root)
    assert run_hmmer_cli._apply_patches(git_root, _cfg(TWO_HUNKS)) == "git"

    _no_git(monkeypatch)
    py_root = tmp_path / "b" / "hmmer-src"
    _write_source(py_root)
    assert run_hmmer_cli._apply_patches(py_root, _cfg(TWO_HUNKS)) == "python"

    assert (py_root / "src" / "kernel.c").read_bytes() == (git_root / "src" / "kernel.c").read_bytes()


@requires_git
def test_apply_patches_recovers_a_patch_git_refuses(tmp_path, monkeypatch):
    monkeypatch.delenv("SEQ_ALIGN_PATCH_TOOL", raising=False)
    build_root = _prepare(tmp_path)
    assert run_hmmer_cli._apply_patches(build_root, _cfg(ADDITION_TAILED_HUNK)) == "python"
    assert "int probe = 1;" in (build_root / "src" / "kernel.c").read_text(encoding="utf-8")


@requires_git
def test_apply_patches_handles_a_two_component_patch(tmp_path, monkeypatch):
    """Two components -> two bare file sections, which git --recount cannot
    apply. Before the fallback existed this was a hard build failure."""
    monkeypatch.delenv("SEQ_ALIGN_PATCH_TOOL", raising=False)
    build_root = tmp_path / "hmmer-src"
    _write_source(build_root)
    _write_source(build_root, OTHER_SOURCE, relative="src/other.c")
    cfg = AssembledConfig()
    cfg.c_patches = [
        ComponentSpec(
            contract_name="kernel", kind=ComponentKind.C_PATCH, c_patch=SIMPLE_HUNK,
            c_target_file="kernel.c", c_target_functions=["calc_band_1"],
        ),
        ComponentSpec(
            contract_name="other", kind=ComponentKind.C_PATCH, c_patch=OTHER_HUNK,
            c_target_file="other.c", c_target_functions=["other_helper"],
        ),
    ]
    cfg.needs_recompile = True
    assert run_hmmer_cli._apply_patches(build_root, cfg) == "python"
    assert "int probe = 1;" in (build_root / "src" / "kernel.c").read_text(encoding="utf-8")
    assert "/* touched */" in (build_root / "src" / "other.c").read_text(encoding="utf-8")


def test_apply_patches_without_patches_is_a_noop(tmp_path):
    cfg = AssembledConfig()
    cfg.c_patches = []
    assert run_hmmer_cli._apply_patches(tmp_path, cfg) == "none"


@requires_git
def test_apply_patches_reports_both_appliers_when_the_patch_is_bad(tmp_path, monkeypatch):
    monkeypatch.delenv("SEQ_ALIGN_PATCH_TOOL", raising=False)
    build_root = _prepare(tmp_path)
    bad = SIMPLE_HUNK.replace("    int band = 0;", "    int bandwidth = 0;")
    with pytest.raises(RuntimeError) as excinfo:
        run_hmmer_cli._apply_patches(build_root, _cfg(bad))
    message = str(excinfo.value)
    assert "git (" in message
    assert "built-in applier" in message


def test_patch_tool_can_be_pinned_to_git(tmp_path, monkeypatch):
    monkeypatch.setenv("SEQ_ALIGN_PATCH_TOOL", "git")
    _no_git(monkeypatch)
    build_root = _prepare(tmp_path)
    with pytest.raises(RuntimeError, match="forbids the built-in applier"):
        run_hmmer_cli._apply_patches(build_root, _cfg(SIMPLE_HUNK))


@requires_git
def test_patch_tool_can_be_pinned_to_python(tmp_path, monkeypatch):
    monkeypatch.setenv("SEQ_ALIGN_PATCH_TOOL", "python")
    build_root = _prepare(tmp_path)
    assert run_hmmer_cli._apply_patches(build_root, _cfg(SIMPLE_HUNK)) == "python"


def test_patch_tool_rejects_unknown_value(tmp_path, monkeypatch):
    monkeypatch.setenv("SEQ_ALIGN_PATCH_TOOL", "patch")
    with pytest.raises(RuntimeError, match="SEQ_ALIGN_PATCH_TOOL must be one of"):
        run_hmmer_cli._apply_patches(_prepare(tmp_path), _cfg(SIMPLE_HUNK))
