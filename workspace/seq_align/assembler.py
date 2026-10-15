"""Assembler: validate + assemble component modifications into a runnable config.

This replaces the old format_check.py (which only validated F1-F4 floats) with
a contract-aware assembler. It:

1. Looks up each ComponentModification's ComponentContract.
2. For PYTHON components, calls the component module's materialize() to turn
   the python_payload into a ComponentSpec (pyhmmer hook / kwargs).
3. For C_PATCH components, verifies the unified diff only touches the
   contract's c_target_functions in c_target_file (the structural constraint),
   then carries the verified diff for the sandbox to apply+recompile.
4. Returns an AssembledConfig consumed by the CLI evaluator to map portable
   policy payloads to flags and to apply any verified C patches before rebuild.

The diff verification is the heart of "Python constrains the C optimization
space": it parses the unified diff's +++/--- headers and @@ hunks and rejects
any patch that touches a file or function outside the contract. An LLM that
tries to edit unrelated code is stopped here, before the sandbox ever runs.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from .eval_schema import EvalResult
from .c_patch_loop.models import ActualDiffFootprint, DiffHunkFootprint
from .components.base import (
    ComponentContract,
    ComponentKind,
    ComponentModification,
    ComponentSpec,
)


# --- component registry ------------------------------------------------------

def _python_contracts() -> dict[str, tuple[ComponentContract, callable]]:
    """Map of PYTHON component contract -> materialize() callable."""
    from .components import (
        inclusion_strategy,
        convergence_rule,
        scoring_system,
        filter_cascade,
        weighting_strategy,
        eff_seq_strategy,
    )
    out: dict[str, tuple[ComponentContract, callable]] = {}
    for mod in (inclusion_strategy, convergence_rule, scoring_system,
                filter_cascade, weighting_strategy, eff_seq_strategy):
        out[mod.CONTRACT.name] = (mod.CONTRACT, mod.materialize)
    return out


def _c_contracts() -> dict[str, ComponentContract]:
    from .components.c_kernels import ALL_C_KERNEL_CONTRACTS
    return dict(ALL_C_KERNEL_CONTRACTS)


def all_contracts() -> dict[str, ComponentContract]:
    out = {name: c for name, (c, _) in _python_contracts().items()}
    out.update(_c_contracts())
    return out


# --- unified-diff verification (the structural constraint) -------------------

_HUNK_HEADER_RE = re.compile(
    r"^@@ -(?P<old_start>\d+)(?:,(?P<old_count>\d+))? "
    r"\+(?P<new_start>\d+)(?:,(?P<new_count>\d+))? @@"
)


def verify_c_patch(
    contract: ComponentContract,
    diff: str,
    hmmer_src_root: str | Path | None = None,
) -> list[str]:
    """Verify a unified diff respects the contract's allowed file/functions.

    Returns a list of human-readable violations (empty == accepted). The check is
    source-aware: every changed line in every hunk must fall inside one of the
    contract's allowed C function bodies in the target source file.
    """
    violations: list[str] = []
    sections = _parse_diff_sections(diff)
    if not sections:
        return ["Patch has no file header; expected a unified diff."]
    if len(sections) > 1:
        violations.append("Patch touches multiple files; one C component patch may modify only its target file.")
        return violations

    target_file = contract.c_target_file or ""
    allowed_paths = _allowed_diff_paths(target_file)
    if not target_file:
        violations.append(f"Contract {contract.name} has no c_target_file.")
        return violations

    root = Path(hmmer_src_root) if hmmer_src_root is not None else _default_hmmer_src_root()
    allowed_functions = set(contract.c_target_functions or [])
    if not allowed_functions and not contract.c_allow_translation_unit:
        violations.append(f"Contract {contract.name} has no c_target_functions.")
        return violations

    for section in sections:
        old_path = section.get("old")
        new_path = section.get("new")
        if old_path == "/dev/null" or new_path == "/dev/null":
            violations.append("Patch creates or deletes a file; C component patches may only edit existing files.")
            continue
        touched = new_path or old_path
        if not touched:
            violations.append("Patch file section has no ---/+++ header.")
            continue
        if touched not in allowed_paths:
            violations.append(
                f"Patch touches '{touched}' but contract allows only "
                f"{sorted(allowed_paths)}."
            )
            continue

        source_file = _resolve_source_file(root, touched, target_file)
        if not source_file.exists():
            violations.append(f"Cannot verify patch hunks: source file not found: {source_file}")
            continue
        if contract.c_allow_translation_unit:
            hunks = section.get("hunks", [])
            if not hunks:
                violations.append(f"Patch section for {touched} has no hunks.")
            elif not any(_changed_old_locations(hunk) for hunk in hunks):
                violations.append("Patch hunks have no changed lines.")
            continue

        ranges = _function_ranges(source_file, allowed_functions)
        missing = sorted(allowed_functions - set(ranges))
        if missing:
            violations.append(
                f"Cannot verify patch hunks: allowed functions not found in "
                f"{source_file}: {missing}"
            )
            continue

        hunks = section.get("hunks", [])
        if not hunks:
            violations.append(f"Patch section for {touched} has no hunks.")
            continue
        for hunk in hunks:
            changed = _changed_old_locations(hunk)
            if not changed:
                violations.append("Patch hunk has no changed lines.")
                continue
            for line_no, op in changed:
                if not _location_in_allowed_function(line_no, ranges, op):
                    violations.append(
                        f"Patch changes {touched}:{line_no} outside allowed functions "
                        f"{sorted(allowed_functions)}."
                    )
                    break

    return violations


def _default_hmmer_src_root() -> Path:
    return Path(__file__).resolve().parents[2] / "tools" / "hmmer-3.4" / "src"


def _allowed_diff_paths(target_file: str) -> set[str]:
    target = _normalize_diff_path(target_file)
    paths = {target, f"src/{target}", Path(target).name}
    return {p for p in paths if p}


def _normalize_diff_path(path: str | None) -> str | None:
    if path is None:
        return None
    out = path.strip().split("\t", 1)[0].strip()
    if out in {"/dev/null", "dev/null"}:
        return "/dev/null"
    if out.startswith("a/") or out.startswith("b/"):
        out = out[2:]
    if out.startswith("./"):
        out = out[2:]
    return out


def _parse_diff_sections(diff: str) -> list[dict[str, Any]]:
    sections: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    current_hunk: list[str] | None = None

    for line in diff.splitlines():
        if line.startswith("diff --git "):
            if current is not None:
                sections.append(current)
            current = {"old": None, "new": None, "hunks": []}
            current_hunk = None
            parts = line.split()
            if len(parts) >= 4:
                current["old"] = _normalize_diff_path(parts[2])
                current["new"] = _normalize_diff_path(parts[3])
            continue
        if line.startswith("--- "):
            if current is None:
                current = {"old": None, "new": None, "hunks": []}
            current["old"] = _normalize_diff_path(line[4:])
            current_hunk = None
            continue
        if line.startswith("+++ "):
            if current is None:
                current = {"old": None, "new": None, "hunks": []}
            current["new"] = _normalize_diff_path(line[4:])
            current_hunk = None
            continue
        if line.startswith("@@ "):
            if current is None:
                current = {"old": None, "new": None, "hunks": []}
            current_hunk = [line]
            current["hunks"].append(current_hunk)
            continue
        if current_hunk is not None:
            current_hunk.append(line)

    if current is not None:
        sections.append(current)
    return sections


def _resolve_source_file(root: Path, touched: str, target_file: str) -> Path:
    target = target_file.strip().lstrip("./")
    candidates = [
        root / touched,
        root / target,
        root / "src" / target,
    ]
    if root.name == "src":
        candidates.extend([root.parent / touched, root.parent / target, root.parent / "src" / target])
    if target.startswith("easel/"):
        candidates.append(root / target.removeprefix("easel/"))
        if root.name == "src":
            candidates.append(root.parent / target)
    candidates.extend([
        root / "src" / Path(target).name,
        root / Path(target).name,
    ])
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return root / target if target.startswith("easel/") else root / "src" / target


def _function_ranges(source_file: Path, names: set[str]) -> dict[str, tuple[int, int]]:
    lines = source_file.read_text(encoding="utf-8", errors="replace").splitlines()
    all_ranges = _all_function_ranges(lines)
    return {name: all_ranges[name] for name in names if name in all_ranges}


_NON_FUNCTION_NAMES = {
    "if", "for", "while", "switch", "return", "sizeof", "alignof",
    "_Alignof", "_Generic", "_Static_assert",
}


def _all_function_ranges(lines: list[str]) -> dict[str, tuple[int, int]]:
    """Return file-scope C function definitions and their 1-based ranges.

    This intentionally scans definitions, not arbitrary ``name(`` text.  The
    older contract verifier could mistake a call followed by an inner control
    block for a definition.  Walking only top-level declaration headers also
    excludes prototypes, function-pointer variables, and control statements.
    """
    ranges: dict[str, tuple[int, int]] = {}
    in_block_comment = False
    in_macro = False
    header_start: int | None = None
    header_parts: list[str] = []
    idx = 0

    while idx < len(lines):
        raw = lines[idx]
        code, in_block_comment = _strip_c_comments_and_strings(raw, in_block_comment)
        stripped = code.strip()

        if in_macro:
            in_macro = raw.rstrip().endswith("\\")
            idx += 1
            continue
        if stripped.startswith("#"):
            in_macro = raw.rstrip().endswith("\\")
            idx += 1
            continue
        if not stripped:
            idx += 1
            continue

        if header_start is None:
            header_start = idx
        header_parts.append(stripped)
        joined = " ".join(header_parts)

        semi = code.find(";")
        brace = code.find("{")
        if semi != -1 and (brace == -1 or semi < brace):
            header_start = None
            header_parts = []
            idx += 1
            continue
        if brace == -1:
            idx += 1
            continue

        prefix = joined.split("{", 1)[0].strip()
        name = _definition_name(prefix)
        end_idx = _matching_function_end(lines, idx)
        if name is not None and end_idx is not None:
            ranges[name] = (header_start + 1, end_idx + 1)

        # Whether this was a function, struct/union, or initializer, its brace
        # body is one complete top-level construct. Do not inspect nested calls.
        if end_idx is not None:
            idx = end_idx + 1
        else:
            idx += 1
        header_start = None
        header_parts = []

    return ranges


def _definition_name(prefix: str) -> str | None:
    if not prefix or prefix.startswith("typedef ") or "=" in prefix:
        return None
    if not prefix.rstrip().endswith(")"):
        return None
    matches = list(re.finditer(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\(", prefix))
    if not matches:
        return None
    name = matches[-1].group(1)
    return None if name in _NON_FUNCTION_NAMES else name


def _definition_brace_index(lines: list[str], start_idx: int) -> int | None:
    for idx in range(start_idx, min(start_idx + 8, len(lines))):
        line = lines[idx]
        semi = line.find(";")
        brace = line.find("{")
        if brace != -1 and (semi == -1 or brace < semi):
            return idx
        if semi != -1 and (brace == -1 or semi < brace):
            return None
    return None


def _matching_function_end(lines: list[str], brace_idx: int) -> int | None:
    depth = 0
    started = False
    in_block_comment = False
    for idx in range(brace_idx, len(lines)):
        code, in_block_comment = _strip_c_comments_and_strings(lines[idx], in_block_comment)
        for ch in code:
            if ch == "{":
                depth += 1
                started = True
            elif ch == "}":
                depth -= 1
                if started and depth == 0:
                    return idx
    return None


def _strip_c_comments_and_strings(line: str, in_block_comment: bool) -> tuple[str, bool]:
    out: list[str] = []
    i = 0
    in_string: str | None = None
    while i < len(line):
        ch = line[i]
        nxt = line[i + 1] if i + 1 < len(line) else ""
        if in_block_comment:
            if ch == "*" and nxt == "/":
                in_block_comment = False
                i += 2
            else:
                i += 1
            continue
        if in_string:
            if ch == "\\":
                i += 2
                continue
            if ch == in_string:
                in_string = None
            i += 1
            continue
        if ch == "/" and nxt == "*":
            in_block_comment = True
            i += 2
            continue
        if ch == "/" and nxt == "/":
            break
        if ch in {'"', "'"}:
            in_string = ch
            i += 1
            continue
        out.append(ch)
        i += 1
    return "".join(out), in_block_comment


def _changed_old_locations(hunk: list[str]) -> list[tuple[int, str]]:
    if not hunk:
        return []
    m = _HUNK_HEADER_RE.match(hunk[0])
    if not m:
        return []
    old_line = int(m.group("old_start"))
    changed: list[tuple[int, str]] = []
    for line in hunk[1:]:
        if not line or line.startswith("\\"):
            continue
        op = line[0]
        if op == " ":
            old_line += 1
        elif op == "-":
            changed.append((old_line, op))
            old_line += 1
        elif op == "+":
            changed.append((old_line, op))
    return changed


def extract_actual_diff_footprint(
    diff: str,
    target_file: str,
    *,
    hmmer_src_root: str | Path | None = None,
) -> ActualDiffFootprint:
    """Attribute the actual pre/post diff lines to file-scope C functions.

    The footprint is computed against the source version the incremental diff
    was authored from.  It is deliberately separate from the broad component
    contract check: callers can compare ``actual_functions`` with the narrower
    outer/inner plan function sets before accepting a generated patch.
    """
    sections = _parse_diff_sections(diff)
    if len(sections) != 1:
        return ActualDiffFootprint(
            extraction_status="error",
            target_file=target_file,
            errors=[f"expected exactly one diff file section, found {len(sections)}"],
        )

    section = sections[0]
    old_path = section.get("old")
    new_path = section.get("new")
    touched = new_path or old_path
    if old_path == "/dev/null" or new_path == "/dev/null":
        return ActualDiffFootprint(
            extraction_status="error",
            target_file=touched,
            errors=["diff creates or deletes a file"],
        )
    allowed_paths = _allowed_diff_paths(target_file)
    if not touched or touched not in allowed_paths:
        return ActualDiffFootprint(
            extraction_status="error",
            target_file=touched,
            errors=[
                f"diff targets {touched!r}, outside planned file {target_file!r} "
                f"({sorted(allowed_paths)})"
            ],
        )

    root = Path(hmmer_src_root) if hmmer_src_root is not None else _default_hmmer_src_root()
    source_file = _resolve_source_file(root, touched, target_file)
    if not source_file.is_file():
        return ActualDiffFootprint(
            extraction_status="error",
            target_file=touched,
            errors=[f"source file not found: {source_file}"],
        )

    pre_lines = source_file.read_text(encoding="utf-8", errors="replace").splitlines()
    hunks = section.get("hunks") or []
    post_lines, apply_errors = _apply_hunks_to_lines(pre_lines, hunks)
    if apply_errors:
        return ActualDiffFootprint(
            extraction_status="error",
            target_file=touched,
            errors=apply_errors,
        )

    pre_ranges = _all_function_ranges(pre_lines)
    post_ranges = _all_function_ranges(post_lines)
    hunk_records: list[DiffHunkFootprint] = []
    actual: list[str] = []
    errors: list[str] = []
    for hunk_idx, hunk in enumerate(hunks, start=1):
        header = _hunk_coordinates(hunk)
        if header is None:
            errors.append(f"hunk {hunk_idx} has an invalid header")
            continue
        old_start, old_count, new_start, new_count = header
        changed_old = _changed_line_numbers(hunk, side="old")
        changed_new = _changed_line_numbers(hunk, side="new")
        pre_functions = _functions_at_lines(changed_old, pre_ranges)
        post_functions = _functions_at_lines(changed_new, post_ranges)
        for name in [*pre_functions, *post_functions]:
            if name not in actual:
                actual.append(name)
        hunk_records.append(DiffHunkFootprint(
            old_start=old_start,
            old_count=old_count,
            new_start=new_start,
            new_count=new_count,
            changed_old_lines=changed_old,
            changed_new_lines=changed_new,
            pre_functions=pre_functions,
            post_functions=post_functions,
        ))

    if not hunk_records:
        errors.append("diff has no valid hunks")
    return ActualDiffFootprint(
        extraction_status="ok" if not errors else "error",
        target_file=touched,
        hunks=hunk_records,
        pre_functions=_ordered_unique(
            name for item in hunk_records for name in item.pre_functions
        ),
        post_functions=_ordered_unique(
            name for item in hunk_records for name in item.post_functions
        ),
        actual_functions=actual,
        errors=errors,
    )


def _hunk_coordinates(hunk: list[str]) -> tuple[int, int, int, int] | None:
    if not hunk:
        return None
    match = _HUNK_HEADER_RE.match(hunk[0])
    if match is None:
        return None
    return (
        int(match.group("old_start")),
        int(match.group("old_count") or 1),
        int(match.group("new_start")),
        int(match.group("new_count") or 1),
    )


def _apply_hunks_to_lines(
    original: list[str],
    hunks: list[list[str]],
) -> tuple[list[str], list[str]]:
    output: list[str] = []
    source_cursor = 0
    errors: list[str] = []
    for hunk_idx, hunk in enumerate(hunks, start=1):
        coordinates = _hunk_coordinates(hunk)
        if coordinates is None:
            errors.append(f"hunk {hunk_idx} has an invalid header")
            continue
        old_start, old_count, _new_start, new_count = coordinates
        hunk_start = old_start if old_count == 0 else max(0, old_start - 1)
        if hunk_start < source_cursor or hunk_start > len(original):
            errors.append(f"hunk {hunk_idx} has an out-of-order old range")
            continue
        output.extend(original[source_cursor:hunk_start])
        source_idx = hunk_start
        seen_old = 0
        seen_new = 0
        for raw in hunk[1:]:
            if raw.startswith("\\"):
                continue
            if not raw:
                errors.append(f"hunk {hunk_idx} contains an unprefixed line")
                continue
            op, payload = raw[0], raw[1:]
            if op in {" ", "-"}:
                if source_idx >= len(original) or original[source_idx] != payload:
                    errors.append(
                        f"hunk {hunk_idx} does not match source at line {source_idx + 1}"
                    )
                    break
                if op == " ":
                    output.append(original[source_idx])
                    seen_new += 1
                source_idx += 1
                seen_old += 1
            elif op == "+":
                output.append(payload)
                seen_new += 1
            else:
                errors.append(f"hunk {hunk_idx} contains invalid prefix {op!r}")
                break
        if seen_old != old_count or seen_new != new_count:
            errors.append(
                f"hunk {hunk_idx} count mismatch: expected old/new "
                f"{old_count}/{new_count}, saw {seen_old}/{seen_new}"
            )
        source_cursor = source_idx
    output.extend(original[source_cursor:])
    return output, _ordered_unique(errors)


def _changed_line_numbers(hunk: list[str], *, side: str) -> list[int]:
    coordinates = _hunk_coordinates(hunk)
    if coordinates is None:
        return []
    old_line, _old_count, new_line, _new_count = coordinates
    changed: list[int] = []
    for raw in hunk[1:]:
        if not raw or raw.startswith("\\"):
            continue
        op = raw[0]
        if op == " ":
            old_line += 1
            new_line += 1
        elif op == "-":
            if side == "old":
                changed.append(old_line)
            old_line += 1
        elif op == "+":
            if side == "new":
                changed.append(new_line)
            new_line += 1
    return changed


def _functions_at_lines(
    line_numbers: list[int],
    ranges: dict[str, tuple[int, int]],
) -> list[str]:
    out: list[str] = []
    for line_no in line_numbers:
        for name, (start, end) in ranges.items():
            if start <= line_no <= end and name not in out:
                out.append(name)
    return out


def _ordered_unique(values: Any) -> list[Any]:
    out: list[Any] = []
    for value in values:
        if value not in out:
            out.append(value)
    return out


def _location_in_allowed_function(line_no: int, ranges: dict[str, tuple[int, int]], op: str) -> bool:
    for start, end in ranges.values():
        if start <= line_no <= end:
            return True
        if op == "+" and start <= line_no + 1 <= end:
            return True
    return False


# --- assembled config -------------------------------------------------------

class AssembledConfig:
    """Output of assemble(): everything run_jack.py needs to run a variant."""

    def __init__(self):
        self.specs: dict[str, ComponentSpec] = {}
        self.c_patches: list[ComponentSpec] = []  # C_PATCH specs needing recompile
        self.needs_recompile: bool = False
        self.errors: list[str] = []

    def select_hits(self):
        """Return the select_hits *factory* (callable producing the callback).

        run_variant calls this factory once to obtain the actual select_hits
        callback (so per-run state is fresh). Returning the factory rather than
        calling it here keeps state isolation at the run boundary.
        """
        f = self.specs.get("inclusion_strategy")
        return f.hook_factory if f and f.hook_factory else None

    def should_stop_factory(self):
        """Return the convergence *factory* (callable producing the predicate).

        run_variant calls this factory once to obtain the should_stop predicate.
        """
        f = self.specs.get("convergence_rule")
        return f.hook_factory if f and f.hook_factory else None

    def builder_kwargs(self) -> dict:
        kw: dict = {}
        for n in ("scoring_system", "weighting_strategy", "eff_seq_strategy"):
            s = self.specs.get(n)
            if s:
                kw.update(s.builder_kwargs)
        return kw

    def pipeline_kwargs(self) -> dict:
        kw: dict = {}
        s = self.specs.get("filter_cascade")
        if s:
            kw.update(s.pipeline_kwargs)
        return kw


def assemble(
    mods: list[ComponentModification],
    *,
    hmmer_src_root: str | Path | None = None,
) -> AssembledConfig:
    """Validate + assemble a list of component modifications."""
    py = _python_contracts()
    cc = _c_contracts()
    cfg = AssembledConfig()

    for mod in mods:
        if mod.contract_name in py:
            contract, materialize = py[mod.contract_name]
            try:
                spec = materialize(mod)
                cfg.specs[mod.contract_name] = spec
            except Exception as e:  # ValueError from materialize on bad payload
                cfg.errors.append(f"{mod.contract_name}: {e}")
        elif mod.contract_name in cc:
            contract = cc[mod.contract_name]
            diff = mod.c_patch or ""
            violations = verify_c_patch(contract, diff, hmmer_src_root=hmmer_src_root)
            if violations:
                cfg.errors.extend(
                    f"{mod.contract_name}: {v}" for v in violations
                )
            else:
                spec = ComponentSpec(
                    contract_name=mod.contract_name,
                    kind=ComponentKind.C_PATCH,
                    c_patch=diff,
                    c_target_file=contract.c_target_file,
                    c_target_functions=contract.c_target_functions,
                    c_allow_translation_unit=contract.c_allow_translation_unit,
                )
                cfg.c_patches.append(spec)
                cfg.specs[mod.contract_name] = spec
                cfg.needs_recompile = True
        else:
            cfg.errors.append(
                f"Unknown component '{mod.contract_name}'. "
                f"Known: {sorted(list(py.keys()) + list(cc.keys()))}"
            )

    return cfg


def assemble_to_eval(mods: list[ComponentModification], stage: str = "smoke") -> tuple[AssembledConfig | None, EvalResult]:
    """Assemble and on failure return an EvalResult describing why.

    Convenience for the eval tools: if assembly fails (bad payload, patch out
    of contract), produce a structured EvalResult with status=compile_error/
    run_error and the violations, rather than raising.
    """
    cfg = assemble(mods)
    if cfg.errors:
        res = EvalResult(
            stage=stage,
            status="compile_error",
            errors=cfg.errors,
        )
        return None, res
    return cfg, EvalResult(stage=stage, status="ok")
