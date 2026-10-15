"""Stdlib-only unified-diff application.

Why this module exists
----------------------
The HMMER build applies the agent's C patch to a private copy of the source
tree, and ``git apply`` used to be the only applier. git is not guaranteed to
exist where the build actually runs: on the target cluster ``/usr/bin/git`` is
present on the login node and *absent* on the compute nodes, and because the
compile runs inside the Slurm job the failure surfaced as a bare
``FileNotFoundError: [Errno 2] No such file or directory: 'git'`` buried in the
engine's repair feedback. Nothing in that message says "install git on the
compute nodes".

``git apply`` is still tried first, so behaviour is unchanged wherever git
exists; this module is the fallback, letting a patch be applied with nothing
but the standard library. It also removes the latent hazard that patch
application depended on the submitting shell's PATH.

Semantics deliberately mirror ``git apply --recount``
-----------------------------------------------------
* Hunk line counts are **recomputed from the hunk body**, never trusted from the
  ``@@`` header -- that is exactly what ``--recount`` means, and model-authored
  diffs routinely get the counts wrong.
* A hunk's recorded position is a *hint*: the context is located by content, so
  small line-offset drift still applies.
* Anything that cannot be located unambiguously, or that would run past the end
  of the file, raises :class:`PatchApplyError` instead of being silently
  mis-applied. A mis-applied C patch is far worse than a loud failure: it would
  still compile, and only the evaluation numbers would be quietly wrong.

Deliberately unsupported, and rejected loudly rather than ignored: file
creation/deletion, renames, mode changes, binary patches. Those never occur in
this workflow (every contract component patches an existing file).

Divergences from git (verified against git 2.34, see the unit tests)
-------------------------------------------------------------------
git remains the first choice, so these only matter on the fallback path.

* ``--recount`` makes git's scanner keep consuming hunk body lines until it hits
  a line that cannot be one. ``--- a/other.c`` *can* be one (a removal whose
  content is ``-- a/other.c``), so **git apply --recount cannot apply a patch
  with two bare file sections** -- it swallows the second section's header into
  the first hunk. Real ``git diff`` output never shows this because every
  section is preceded by a ``diff --git`` line, which ``_normalize_patch_paths``
  does not emit. The same scanner also swallows a trailing empty line, so a hunk
  whose last body line is an addition is refused unless ``--unidiff-zero`` is
  passed.
* For a zero-context insertion (``@@ -N,0 +M,K @@``) git lands at the end of the
  file; we insert after the recorded line N, which is what the format says.

So the fallback is strictly *more* capable than git here, which is why a patch
git rejects can still be applied. It is never less careful: the context check
below is the same in both directions, and nothing is written unless the whole
patch applies.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

__all__ = [
    "PatchApplyError",
    "FileSection",
    "Hunk",
    "apply_unified_diff",
    "parse_unified_diff",
]

_DEV_NULL = {"/dev/null", "dev/null"}

_HUNK_HEADER = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@[ \t]?(.*)$")

# Header lines that end a file section. Note `--- ` is handled specially: a
# removed body line whose content starts with `-- ` is rendered `--- …`, so it
# only counts as a file header when the next line is a `+++ ` header.
_SECTION_BREAKS = (
    "diff --git ",
    "diff --cc ",
    "diff --combined ",
    "diff --no-index ",
    "index ",
    "old mode ",
    "new mode ",
    "deleted file mode ",
    "new file mode ",
    "similarity index ",
    "dissimilarity index ",
    "rename from ",
    "rename to ",
    "Binary files ",
    "GIT binary patch",
)

# A context block this short is not a fingerprint: `}` / `}` / `}` occurs
# hundreds of times in a C file. Such a hunk is only ever applied at its
# recorded position.
_DISTINCTIVE_MIN_LINES = 6
_DISTINCTIVE_MIN_WIDTH = 8

# How far a short, generic-ish hunk may slide. Long, distinctive blocks are
# looked for anywhere in the file.
_NEARBY_WINDOW = 40
_LONG_BLOCK_MIN_LINES = 3


class PatchApplyError(RuntimeError):
    """Raised when a patch cannot be applied with confidence."""


@dataclass(frozen=True)
class Hunk:
    old_start: int
    old_count: int
    new_start: int
    new_count: int
    trailer: str
    body: tuple[str, ...]

    @property
    def label(self) -> str:
        return f"@@ -{self.old_start},{self.old_count} +{self.new_start},{self.new_count} @@"

    def old_block(self) -> list[str]:
        """Context + removed lines: what must be present in the file."""
        return [_content(line) for line in self.body if _is_context(line) or _is_removed(line)]

    def new_block(self) -> list[str]:
        """Context + added lines: what replaces it."""
        return [_content(line) for line in self.body if _is_context(line) or _is_added(line)]


@dataclass(frozen=True)
class FileSection:
    old_path: str
    new_path: str
    hunks: tuple[Hunk, ...]


def _is_context(line: str) -> bool:
    # A truly empty line is a context line holding an empty line: some diff
    # writers emit it that way instead of a single space.
    return line == "" or line.startswith(" ")


def _is_removed(line: str) -> bool:
    return line.startswith("-")


def _is_added(line: str) -> bool:
    return line.startswith("+")


def _content(line: str) -> str:
    return line[1:] if line else ""


def _header_path(text: str) -> str:
    # `a/foo.c`, or `a/foo.c\t2026-01-01 00:00:00` from diff -u.
    return text.split("\t", 1)[0].strip()


def _starts_file_header(lines: list[str], index: int) -> bool:
    """True when ``lines[index]`` (a `--- ` line) really opens a file section."""
    nxt = lines[index + 1] if index + 1 < len(lines) else ""
    return nxt.startswith("+++ ")


def parse_unified_diff(patch_text: str) -> list[FileSection]:
    """Parse a unified diff into file sections.

    Raises :class:`PatchApplyError` on a structurally malformed patch -- in
    particular on a hunk with an empty body, which is what a truncated patch
    looks like and would otherwise be applied as a silent no-op.
    """
    lines = patch_text.splitlines()
    sections: list[FileSection] = []
    index = 0
    total = len(lines)

    while index < total:
        if not lines[index].startswith("--- "):
            index += 1
            continue

        old_path = _header_path(lines[index][4:])
        index += 1
        if index >= total or not lines[index].startswith("+++ "):
            raise PatchApplyError(
                f"malformed patch: `--- {old_path}` is not followed by a `+++ ` header"
            )
        new_path = _header_path(lines[index][4:])
        index += 1

        hunks: list[Hunk] = []
        while index < total:
            line = lines[index]
            if line.startswith(_SECTION_BREAKS):
                break
            if line.startswith("--- ") and _starts_file_header(lines, index):
                break
            if line.startswith("@@") and not _HUNK_HEADER.match(line):
                raise PatchApplyError(f"malformed hunk header: {line!r}")

            match = _HUNK_HEADER.match(line)
            if match is None:
                if line.strip():
                    raise PatchApplyError(
                        f"malformed patch: expected a hunk header before {line!r}"
                    )
                index += 1
                continue

            old_start = int(match.group(1))
            old_count = int(match.group(2)) if match.group(2) is not None else 1
            new_start = int(match.group(3))
            new_count = int(match.group(4)) if match.group(4) is not None else 1
            trailer = match.group(5) or ""
            index += 1

            body: list[str] = []
            while index < total:
                current = lines[index]
                if current.startswith("\\"):  # "\ No newline at end of file"
                    index += 1
                    continue
                if current.startswith("@@") or current.startswith(_SECTION_BREAKS):
                    break
                if current.startswith("--- ") and _starts_file_header(lines, index):
                    break
                if _is_context(current) or _is_added(current) or _is_removed(current):
                    body.append(current)
                    index += 1
                    continue
                break

            hunks.append(
                Hunk(
                    old_start=old_start,
                    old_count=old_count,
                    new_start=new_start,
                    new_count=new_count,
                    trailer=trailer,
                    body=_trim_blank_edges(tuple(body)),
                )
            )

        if not hunks:
            raise PatchApplyError(f"malformed patch: file section `{new_path or old_path}` has no hunks")
        sections.append(FileSection(old_path=old_path, new_path=new_path, hunks=tuple(hunks)))

    return sections


def _trim_blank_edges(body: tuple[str, ...]) -> tuple[str, ...]:
    """Drop blank context lines from the hunk's edges.

    Edge context lines carry no locating information and are copied through
    unchanged, so removing them cannot alter the file's content -- it only
    stops a blank separator line the model wrote between hunks from being
    mistaken for context.
    """
    start = 0
    end = len(body)
    while start < end and _content(body[start]).strip() == "" and _is_context(body[start]):
        start += 1
    while end > start and _content(body[end - 1]).strip() == "" and _is_context(body[end - 1]):
        end -= 1
    return body[start:end]


def apply_unified_diff(
    patch_text: str,
    root: Path,
    *,
    search_window: int | None = None,
) -> list[str]:
    """Apply ``patch_text`` under ``root``; return the touched relative paths.

    ``root`` is the same directory ``git apply`` would be run from, and patch
    paths are resolved relative to it.
    """
    sections = parse_unified_diff(patch_text)
    if not sections:
        raise PatchApplyError("patch contains no file sections (nothing to apply)")

    root = Path(root)
    buffers: dict[str, list[str]] = {}
    touched: list[str] = []

    for section in sections:
        relative = _resolve_relative(section, root)
        target = root / relative
        if relative not in buffers:
            if not target.is_file():
                raise PatchApplyError(f"{relative}: patched file does not exist under {root}")
            buffers[relative] = _read_lines(target)
        _apply_section(buffers[relative], section, relative, search_window)
        if relative not in touched:
            touched.append(relative)

    for relative, lines in buffers.items():
        _write_lines(root / relative, lines)
    return touched


def _resolve_relative(section: FileSection, root: Path) -> str:
    for side, header in (("old", section.old_path), ("new", section.new_path)):
        token = header.strip()
        if len(token) >= 2 and token.startswith('"') and token.endswith('"'):
            token = token[1:-1]
        if token in _DEV_NULL:
            # /dev/null on either side is a creation or a deletion. Both are
            # outside this workflow -- every contract component patches an
            # existing file -- and silently treating them as a normal edit
            # would be wrong in a way that still compiles.
            raise PatchApplyError(
                f"creating or deleting files is not supported ({side} path is {header})"
            )

    header = section.new_path or section.old_path
    token = header.strip()
    if len(token) >= 2 and token.startswith('"') and token.endswith('"'):
        token = token[1:-1]
    if token.startswith(("a/", "b/")):
        token = token[2:]
    if not token:
        raise PatchApplyError("patch section has an empty target path")
    if token.startswith("/"):
        raise PatchApplyError(f"absolute paths are not supported ({header})")
    parts = Path(token).parts
    if ".." in parts:
        raise PatchApplyError(f"paths outside the source tree are not supported ({header})")
    relative = Path(*[part for part in parts if part not in (".", "")]).as_posix()
    if not relative:
        raise PatchApplyError(f"patch section resolves to the source tree root ({header})")
    if not (root / relative).exists() and (root / "src" / relative).is_file():
        # Tolerate a diff written without the `src/` prefix that
        # _normalize_patch_paths would have added.
        relative = (Path("src") / relative).as_posix()
    return relative


def _read_lines(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8", errors="surrogateescape", newline="") as handle:
        return handle.read().split("\n")


def _write_lines(path: Path, lines: list[str]) -> None:
    with path.open("w", encoding="utf-8", errors="surrogateescape", newline="\n") as handle:
        handle.write("\n".join(lines))


def _apply_section(
    lines: list[str],
    section: FileSection,
    relative: str,
    search_window: int | None,
) -> None:
    delta = 0  # cumulative (new - old) line counts of hunks already applied
    for number, hunk in enumerate(section.hunks, start=1):
        label = f"{relative}: hunk #{number} {hunk.label}"
        if not hunk.body:
            raise PatchApplyError(
                f"{label}: hunk has an empty body -- the patch looks truncated, refusing to apply"
            )
        old_block = hunk.old_block()
        new_block = hunk.new_block()

        if not old_block:
            # Pure insertion (`@@ -N,0 +M,K @@`): insert after line N.
            position = min(max(hunk.old_start, 0), len(lines))
            logger.warning("%s: pure insertion at line %d (no context to verify)", label, position + 1)
        else:
            expected = hunk.old_start - 1 + delta
            window = search_window if search_window is not None else _window_for(old_block)
            position = _locate(lines, old_block, expected, window, label)

        lines[position : position + len(old_block)] = new_block
        delta += len(new_block) - len(old_block)


def _window_for(block: list[str]) -> int | None:
    if len(block) >= _LONG_BLOCK_MIN_LINES:
        return None  # long context is a fingerprint; look anywhere
    return _NEARBY_WINDOW


def _is_distinctive(block: list[str]) -> bool:
    if len(block) >= _DISTINCTIVE_MIN_LINES:
        return True
    return any(len(line.strip()) >= _DISTINCTIVE_MIN_WIDTH for line in block)


def _matches_at(lines: list[str], block: list[str], index: int) -> bool:
    if index < 0 or index + len(block) > len(lines):
        return False
    return lines[index : index + len(block)] == block


def _locate(
    lines: list[str],
    block: list[str],
    expected: int,
    window: int | None,
    label: str,
) -> int:
    size = len(block)

    # The recorded position is trusted as-is: if the context is there, that is
    # where git would apply it too. Only a *miss* needs the safety checks below
    # (`_matches_at` handles an out-of-range `expected`).
    if _matches_at(lines, block, expected):
        return expected

    if not _is_distinctive(block):
        raise PatchApplyError(
            f"{label}: context does not match at the recorded position (line {expected + 1}) "
            f"and its {size} short line(s) are too generic to relocate safely"
        )

    limit = len(lines) - size
    if limit < 0:
        raise PatchApplyError(
            f"{label}: hunk needs {size} context lines but the file only has {len(lines)}"
        )

    # Scan the whole file and keep every match at the smallest distance from the
    # recorded position. Iterating by index rather than by distance matters: a
    # stale-but-wildly-wrong line number must still find its context.
    best: list[int] = []
    best_distance: int | None = None
    for index in range(limit + 1):
        distance = abs(index - expected)
        if window is not None and distance > window:
            continue
        if best_distance is not None and distance > best_distance:
            continue
        if lines[index : index + size] != block:
            continue
        if best_distance is None or distance < best_distance:
            best, best_distance = [index], distance
        else:
            best.append(index)

    if not best:
        where = (
            f"within {window} line(s) of the recorded position (line {expected + 1})"
            if window is not None
            else f"anywhere in the file (recorded line {expected + 1}, {len(lines)} lines)"
        )
        raise PatchApplyError(
            f"{label}: context not found {where}; the patch does not match this source tree"
        )

    if len(best) > 1:
        raise PatchApplyError(
            f"{label}: context matches at lines {[index + 1 for index in sorted(best)]}, equally "
            f"distant from the recorded line {expected + 1}; refusing to guess"
        )

    if best[0] != expected:
        logger.warning(
            "%s: context found at line %d instead of the recorded %d (offset %+d)",
            label,
            best[0] + 1,
            expected + 1,
            best[0] - expected,
        )
    return best[0]
