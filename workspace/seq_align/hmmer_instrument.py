"""Inject per-stage timers into the HMMER contract functions (profiling build only).

Design choice: wrap, do not sprinkle
------------------------------------
The obvious approach -- insert a timer at function entry and before every ``return``
-- is fragile. ``p7_SSVFilter`` alone has six ``return`` statements; missing one
under-reports the stage, and a ``return`` inside a macro or a comment breaks the
edit. Instead each target is *renamed* and a thin timing wrapper is emitted in its
place::

    static int p7_SSVFilter__seqalign_uninstrumented(...) { ...original body... }

    int p7_SSVFilter(...)                       /* injected wrapper */
    {
      P7_EVO_PROF_ENTER(t0);
      int rc = p7_SSVFilter__seqalign_uninstrumented(...);
      P7_EVO_PROF_EXIT(P7_EVO_PROF_SSV_FILTER, t0);
      return rc;
    }

One insertion point, every exit path covered, and the wrapper keeps the original
name so the public header still matches. The renamed body is only ever called
through the wrapper, so no caller needs to change.

This module is pure text transformation: no build, no Slurm, no filesystem
assumptions beyond the source tree it is handed. That keeps it unit-testable
against the real HMMER sources.

Scope limit
-----------
Only *contract* functions are instrumented (see components/c_kernels.py), so the
profiling build stays semantically aligned with the candidate an agent produced.
``domain_def`` has no editable contract function, so it is deliberately absent
rather than reached for via a non-contract file.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

# (source path relative to the hmmer src root, function, contract slot macro, contract name)
#
# This is the *declared* target set the text transform can work with. Which
# implementation directory a build actually compiles is a property of the
# generated Makefile (see ``impl_dir_from_makefile``), so the profiling build
# resolves the concrete targets at build time with
# ``resolve_instrumentation_targets`` instead of trusting this tuple.
INSTRUMENTATION_TARGETS: tuple[tuple[str, str, str, str], ...] = (
    ("impl_neon/ssvfilter.c", "p7_SSVFilter", "P7_EVO_PROF_SSV_FILTER", "ssv_filter"),
    ("impl_neon/fwdback.c", "forward_engine", "P7_EVO_PROF_DP_KERNEL", "dp_kernel"),
    ("impl_neon/fwdback.c", "backward_engine", "P7_EVO_PROF_DP_KERNEL", "dp_kernel"),
    ("impl_neon/null2.c", "p7_Null2_ByExpectation", "P7_EVO_PROF_NULL2", "null2_correction"),
    ("impl_neon/null2.c", "p7_Null2_ByTrace", "P7_EVO_PROF_NULL2", "null2_correction"),
)

# Which contract functions must be wrapped per active contract island, and the
# timer slot each one reports into. 2026-10-08: the wrapper symbols were present
# in a "profiling" binary while the timers had never been compiled in, so this
# mapping is now verified against both the contract's allowed-function list and
# the real source text before a build is allowed to publish.
CONTRACT_ENTRY_SYMBOLS: tuple[tuple[str, tuple[str, ...], str], ...] = (
    ("ssv_filter_neon", ("p7_SSVFilter",), "P7_EVO_PROF_SSV_FILTER"),
    ("dp_kernel_neon", ("forward_engine", "backward_engine"), "P7_EVO_PROF_DP_KERNEL"),
    ("null2_correction_neon", ("p7_Null2_ByExpectation", "p7_Null2_ByTrace"), "P7_EVO_PROF_NULL2"),
)

# Profiling-shim functions a profiling binary must contain. If these are absent
# the timers were not compiled in, which is exactly the silent failure this
# module now refuses to publish.
REQUIRED_SHIM_SYMBOLS: tuple[str, ...] = ("p7_evo_prof_add", "p7_evo_prof_report")

RENAMED_SUFFIX = "__seqalign_uninstrumented"
TIMER_VAR = "p7_evo_prof_t0"
PROF_HEADER = "p7_evo_prof.h"
PROF_SOURCE = "p7_evo_prof.c"
PROF_DEFINE = "P7_EVO_PROF"

_IMPLDIR_RE = re.compile(r"^\s*IMPLDIR\s*[:?]?=\s*(?P<value>\S+)", re.MULTILINE)


class InstrumentationError(RuntimeError):
    """The source did not look the way this transform requires."""


def _match_brace(text: str, open_index: int) -> int:
    """Return the index of the ``}`` matching the ``{`` at ``open_index``.

    Skips string literals, character literals and comments so braces inside them
    do not confuse the count.
    """
    depth = 0
    i = open_index
    n = len(text)
    while i < n:
        ch = text[i]
        if ch == "/" and i + 1 < n and text[i + 1] == "/":
            i = text.find("\n", i)
            if i == -1:
                break
            continue
        if ch == "/" and i + 1 < n and text[i + 1] == "*":
            end = text.find("*/", i + 2)
            i = n if end == -1 else end + 2
            continue
        if ch == '"' or ch == "'":
            quote = ch
            i += 1
            while i < n:
                if text[i] == "\\":
                    i += 2
                    continue
                if text[i] == quote:
                    break
                i += 1
            i += 1
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return i
        i += 1
    raise InstrumentationError("unbalanced braces while locating function body")


def split_function(text: str, name: str) -> dict[str, Any]:
    """Locate a top-level function definition by name.

    Returns the signature text, the body span and the full definition span, all as
    character indices into ``text``.
    """
    # A definition in HMMER's style starts the name at column 0.
    pattern = re.compile(rf"^{re.escape(name)}\s*\(", re.MULTILINE)
    match = pattern.search(text)
    if match is None:
        raise InstrumentationError(f"function definition not found: {name}")
    if pattern.search(text, match.end()) is not None:
        raise InstrumentationError(f"function defined more than once: {name}")

    name_start = match.start()

    # The return type sits on the preceding non-empty line(s).
    line_start = text.rfind("\n", 0, name_start) + 1
    type_end = line_start
    type_start = type_end
    probe = type_end - 1
    while probe > 0 and text[probe] in " \t\n":
        probe -= 1
    line_end = probe + 1
    line_start2 = text.rfind("\n", 0, line_end) + 1
    type_start = line_start2
    return_type = text[type_start:type_end].strip()
    if not return_type:
        raise InstrumentationError(f"could not determine return type for {name}")

    # Parameter list: from the opening paren to its match.
    open_paren = text.index("(", match.start())
    depth = 0
    i = open_paren
    while i < len(text):
        if text[i] == "(":
            depth += 1
        elif text[i] == ")":
            depth -= 1
            if depth == 0:
                break
        i += 1
    close_paren = i

    params = text[open_paren + 1 : close_paren]

    # Body: the next '{' at or after the parameter list.
    open_brace = text.index("{", close_paren)
    close_brace = _match_brace(text, open_brace)

    return {
        "name": name,
        "return_type": return_type,
        "params": params,
        "signature_start": type_start,
        "signature_end": close_paren + 1,
        "body_open": open_brace,
        "body_close": close_brace,
        "definition_start": type_start,
        "definition_end": close_brace + 1,
    }


def _split_params(params: str) -> list[str]:
    parts: list[str] = []
    depth = 0
    current: list[str] = []
    for ch in params:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        if ch == "," and depth == 0:
            parts.append("".join(current))
            current = []
            continue
        current.append(ch)
    tail = "".join(current).strip()
    if tail:
        parts.append(tail)
    return [p for p in (part.strip() for part in parts) if p]


def _arg_names(params: str) -> list[str]:
    names: list[str] = []
    for part in _split_params(params):
        # Drop any trailing array suffix, then take the last identifier.
        cleaned = re.sub(r"\[[^\]]*\]\s*$", "", part.strip())
        tokens = cleaned.replace("*", " * ").split()
        names.append(tokens[-1].strip())
    return names


def _wrapper_source_v2(span: dict[str, Any], slot_macro: str) -> str:
    args = ", ".join(_arg_names(span["params"]))
    return_type = span["return_type"]
    # A pointer return type's last token is e.g. "*P7_OPROFILE"; take the type word.
    return_word = re.sub(r"^\*+", "", return_type.split()[-1])
    return (
        f"\n/* ---- injected by the SeqOpt profiling build ---- */\n"
        f"{return_type}\n"
        f"{span['name']}({span['params']})\n"
        f"{{\n"
        f"  {return_word} rc;\n"
        f"  P7_EVO_PROF_ENTER({TIMER_VAR});\n"
        f"  rc = {span['name']}{RENAMED_SUFFIX}({args});\n"
        f"  P7_EVO_PROF_EXIT({slot_macro}, {TIMER_VAR});\n"
        f"  return rc;\n"
        f"}}\n"
    )


def instrument_source(text: str, name: str, slot_macro: str) -> str:
    """Return ``text`` with ``name`` wrapped by a timing wrapper."""
    span = split_function(text, name)
    original = text[span["definition_start"] : span["definition_end"]]
    renamed = original.replace(name, f"{name}{RENAMED_SUFFIX}", 1)
    wrapper = _wrapper_source_v2(span, slot_macro)
    # Rename the definition, then append the wrapper immediately after it.
    head = text[: span["definition_start"]]
    tail = text[span["definition_end"] :]
    return f"{head}{renamed}{wrapper}{tail}"


def impl_dir_from_makefile(makefile: str | Path) -> str:
    """Return the ``impl_*`` directory the configured HMMER build compiles.

    ``make`` selects the implementation through ``IMPLDIR`` in the generated
    ``src/Makefile``. Reading it back is what keeps the instrumentation and the
    contract surface anchored to the code that is actually compiled instead of to
    whatever the source layout assumed.
    """
    path = Path(makefile)
    if not path.is_file():
        raise InstrumentationError(f"configured Makefile not found: {path}")
    match = _IMPLDIR_RE.search(path.read_text(encoding="utf-8"))
    if match is None:
        raise InstrumentationError(
            f"{path} has no IMPLDIR variable; the HMMER build layout no longer matches this transform"
        )
    impl = match.group("value").strip().strip('"').strip("'")
    if not impl:
        raise InstrumentationError(f"{path} declares an empty IMPLDIR")
    return impl


def _entry_symbols(profiling: bool) -> tuple[tuple[str, tuple[str, ...], str], ...]:
    if profiling:
        return CONTRACT_ENTRY_SYMBOLS
    return tuple(
        (name, tuple(functions), "")
        for name, functions, _slot in (
            (
                contract_name,
                tuple(getattr(contract, "c_entry_functions", ()) or ()),
                "",
            )
            for contract_name, contract in _active_contracts().items()
        )
    )


def _active_contracts() -> dict[str, Any]:
    from .components.c_kernels import ACTIVE_C_KERNEL_CONTRACTS  # noqa: PLC0415

    return dict(ACTIVE_C_KERNEL_CONTRACTS)


def resolve_contract_targets(
    src_root: str | Path,
    *,
    makefile: str | Path | None = None,
    contracts: dict[str, Any] | None = None,
) -> tuple[tuple[tuple[str, str, str, str], ...], dict[str, Any]]:
    """Resolve the entry symbols of every active contract island against the build.

    Used for non-profiling candidate bundles: it answers "would a patch to this
    contract land in the executable this build produces", using the same
    ``IMPLDIR`` binding as the profiling resolver.
    """
    src_root = Path(src_root)
    contracts = dict(contracts) if contracts is not None else _active_contracts()
    targets, report = _resolve_targets(
        src_root,
        makefile=makefile,
        contracts=contracts,
        entry_symbols=tuple(
            (name, tuple(getattr(contract, "c_entry_functions", ()) or ()), "")
            for name, contract in contracts.items()
        ),
        require_function=True,
    )
    report["mode"] = "contract_entry_symbols"
    return targets, report


def _resolve_targets(
    src_root: Path,
    *,
    makefile: str | Path | None,
    contracts: dict[str, Any],
    entry_symbols: tuple[tuple[str, tuple[str, ...], str], ...],
    require_function: bool,
) -> tuple[tuple[tuple[str, str, str, str], ...], dict[str, Any]]:
    resolved_makefile = Path(makefile) if makefile is not None else src_root / "Makefile"
    impl_dir = impl_dir_from_makefile(resolved_makefile)
    targets: list[tuple[str, str, str, str]] = []
    report: dict[str, Any] = {
        "impl_dir": impl_dir,
        "makefile": str(resolved_makefile),
        "targets": [],
    }
    for contract_name, functions, slot_macro in entry_symbols:
        contract = contracts.get(contract_name)
        if contract is None:
            raise InstrumentationError(
                f"active contract island {contract_name!r} is missing; cannot resolve instrumentation targets"
            )
        target_file = str(getattr(contract, "c_target_file", "") or "")
        impl_prefix = Path(target_file).parts[0] if target_file else ""
        if impl_prefix != impl_dir:
            raise InstrumentationError(
                f"contract {contract_name!r} targets {target_file!r}, but this build compiles "
                f"{impl_dir!r} (from {resolved_makefile}). Instrumenting it would time code that is not "
                "linked into the benchmark binary."
            )
        allowed = set(getattr(contract, "c_target_functions", ()) or ())
        # Contract target files are written both with and without the impl_*
        # prefix depending on the island; resolve against the real tree and keep
        # the impl-relative path the instrumentation report should carry.
        source = src_root / target_file
        rel_path = target_file
        if not source.is_file():
            source = src_root / impl_dir / Path(target_file).name
            rel_path = f"{impl_dir}/{Path(target_file).name}"
        if not source.is_file():
            raise InstrumentationError(
                f"instrumentation target file missing for {contract_name!r}: {source}"
            )
        text = source.read_text(encoding="utf-8")
        for function in functions:
            if function not in allowed:
                raise InstrumentationError(
                    f"{contract_name!r} does not permit instrumenting {function!r}; "
                    f"allowed functions: {sorted(allowed)}"
                )
            if require_function and not re.search(rf"^{re.escape(function)}\s*\(", text, re.MULTILINE):
                raise InstrumentationError(
                    f"function definition not found for {contract_name!r}: {function!r} in {source}"
                )
            targets.append((rel_path, function, slot_macro, contract_name))
            report["targets"].append(
                {
                    "contract": contract_name,
                    "file": rel_path,
                    "function": function,
                    "slot_macro": slot_macro,
                }
            )
    return tuple(targets), report


def resolve_instrumentation_targets(
    src_root: str | Path,
    *,
    makefile: str | Path | None = None,
    contracts: dict[str, Any] | None = None,
) -> tuple[tuple[tuple[str, str, str, str], ...], dict[str, Any]]:
    """Resolve the instrumentation targets against the implementation actually built.

    Returns ``(targets, resolution)``. The resolution report is stored in the
    bundle manifest, so "which impl was instrumented, and did every target
    resolve" is auditable after the fact instead of being inferred from a run
    that produced no stage times.

    Raises ``InstrumentationError`` when the compiled implementation cannot carry
    the timers: unknown ``IMPLDIR``, a contract whose target file is missing, an
    entry function the contract does not permit, or a function name the source no
    longer defines.
    """
    return _resolve_targets(
        Path(src_root),
        makefile=makefile,
        contracts=dict(contracts) if contracts is not None else _active_contracts(),
        entry_symbols=CONTRACT_ENTRY_SYMBOLS,
        require_function=True,
    )


def instrument_tree(
    src_root: str | Path,
    *,
    prof_header: Path,
    prof_source: Path,
    targets: tuple[tuple[str, str, str, str], ...] = INSTRUMENTATION_TARGETS,
) -> dict[str, Any]:
    """Instrument every target in a copied HMMER source tree.

    ``src_root`` is the tree containing ``impl_neon/``. Files are rewritten in
    place; the profiler shim is dropped next to them so ``#include "p7_evo_prof.h"``
    resolves. Returns a per-target report for the build manifest.

    Missing targets are an error, not a warning: a target that has moved or been
    renamed means the source layout no longer matches this transform, and a
    profiling build that silently instruments nothing would report all-zero stage
    times that look like measurements.
    """
    src_root = Path(src_root)
    header_dest = src_root / PROF_HEADER
    source_dest = src_root / PROF_SOURCE
    header_dest.write_text(Path(prof_header).read_text(encoding="utf-8"), encoding="utf-8")
    source_dest.write_text(Path(prof_source).read_text(encoding="utf-8"), encoding="utf-8")

    report: dict[str, Any] = {
        "schema_version": "seqalign_instrumentation.v1",
        "header": PROF_HEADER,
        "source": PROF_SOURCE,
        "define": PROF_DEFINE,
        "targets": [],
    }

    # Group by file so each file is read/written once.
    by_file: dict[str, list[tuple[str, str, str]]] = {}
    for rel_path, function, slot_macro, contract in targets:
        by_file.setdefault(rel_path, []).append((function, slot_macro, contract))

    for rel_path, entries in by_file.items():
        target = src_root / rel_path
        if not target.is_file():
            raise InstrumentationError(f"instrumentation target missing: {rel_path}")
        text = target.read_text(encoding="utf-8")

        # The shim must be included before anything fixes the POSIX feature set.
        if f'#include "{PROF_HEADER}"' not in text:
            first_include = re.search(r"^#include .*$", text, re.MULTILINE)
            if first_include is None:
                raise InstrumentationError(f"no #include found in {rel_path}")
            text = (
                text[: first_include.start()]
                + f'#include "{PROF_HEADER}"\n'
                + text[first_include.start() :]
            )

        for function, slot_macro, contract in entries:
            # File-exists is not enough: a target file that still exists but no
            # longer defines the contract function would previously produce an
            # "instrumented" tree with no timers in it.
            if not re.search(rf"^{re.escape(function)}\s*\(", text, re.MULTILINE):
                raise InstrumentationError(
                    f"function definition not found for instrumentation target: {function} in {rel_path}"
                )
            text = instrument_source(text, function, slot_macro)
            report["targets"].append(
                {
                    "file": rel_path,
                    "function": function,
                    "renamed_to": f"{function}{RENAMED_SUFFIX}",
                    "contract": contract,
                    "slot_macro": slot_macro,
                }
            )

        target.write_text(text, encoding="utf-8")

    return report


def enable_profiling_build(makefile: str | Path) -> dict[str, Any]:
    """Turn a configured HMMER build into the profiling build.

    Three edits, all applied to *generated* Makefiles (never to a tracked
    ``Makefile.in``):

    1. ``src/<IMPLDIR>/Makefile``: add ``-D<PROF_DEFINE>``. This is the edit that
       actually matters. ``make`` runs the implementation directory as a sub-make
       with its own ``CFLAGS``, so a define appended to the top-level Makefile
       never reaches the units that call ``P7_EVO_PROF_ENTER``/``EXIT`` -- and
       those macros then compile to no-ops. That is exactly how a "profiling"
       build once shipped with the renamed wrapper symbols linked but not a single
       timer. The define lands *first* in the sub-Makefile's CFLAGS so the -I
       paths follow and the shim header resolves from ``..``.
    2. ``src/Makefile``: also add the define (for any unit compiled at that level).
    3. ``src/Makefile``: archive ``p7_evo_prof.o`` into ``libhmmer.a``. The
       archive rule is ``.FORCE``-driven and its ``$?`` list only contains the
       stamp's prerequisites, so appending to ``OBJS`` is not enough -- a second
       (incremental) build would compile the shim and never add it to the archive,
       which leaves the wrappers with an undefined ``p7_evo_prof_add``.

    Idempotent: a second call adds nothing.
    """
    path = Path(makefile)
    if not path.is_file():
        raise InstrumentationError(f"configured Makefile not found: {path}")

    text = path.read_text(encoding="utf-8")
    object_name = PROF_SOURCE[:-2] + ".o"  # p7_evo_prof.c -> p7_evo_prof.o
    marker = "# ---- SeqOpt profiling build ----"
    impl_dir = impl_dir_from_makefile(path)
    impl_makefile = path.parent / impl_dir / "Makefile"
    if not impl_makefile.is_file():
        raise InstrumentationError(
            f"implementation Makefile not found: {impl_makefile} (IMPLDIR={impl_dir!r}); "
            "cannot make the timers reach the compiled units"
        )
    archive_needle = "${QUIET_AR}${AR} -r libhmmer.a $? > /dev/null 2>&1"
    archive_line = next(
        (line for line in text.splitlines() if line.strip() == archive_needle),
        None,
    )
    if archive_line is None:
        raise InstrumentationError(
            f"{path} has no libhmmer.a archive rule this transform recognises "
            "(expected `ar -r libhmmer.a $?`); refusing to publish an unarchived profiling shim"
        )

    report: dict[str, Any] = {
        "makefile": str(path),
        "impl_dir": impl_dir,
        "impl_makefile": str(impl_makefile),
        "define": PROF_DEFINE,
        "object": object_name,
    }
    if marker in text and marker in impl_makefile.read_text(encoding="utf-8"):
        return {**report, "already_enabled": True, "edited_makefiles": [str(path), str(impl_makefile)]}
    if "OBJS" not in text:
        raise InstrumentationError(
            f"{path} has no OBJS variable; the HMMER build layout no longer matches this transform"
        )

    def _append(target: Path, lines: list[str]) -> None:
        current = target.read_text(encoding="utf-8")
        if marker in current:
            return
        target.write_text(current.rstrip("\n") + "\n" + "\n".join(lines) + "\n", encoding="utf-8")

    # Top level: the define, the shim's own compile rule, and the archive of the
    # shim object into the archive every HMMER program links against.
    #
    # The stamp rule gets `p7_evo_prof.o` as an extra prerequisite (and that is a
    # statement about the *existing* rule, so the original recipe still runs).
    # Without it the rule's `$?` -- which only contains the stamp's own
    # prerequisites -- never mentions the shim, so an incremental build compiles
    # it and still links the wrappers against an undefined `p7_evo_prof_add`.
    top_lines = [
        f"\n{marker}",
        "# Injected by workspace/seq_align/hmmer_instrument.enable_profiling_build.",
        f"CFLAGS += -D{PROF_DEFINE}",
        "",
        f"{object_name}: {PROF_SOURCE}",
        f"\t${{QUIET_CC}}${{CC}} ${{CFLAGS}} ${{PTHREAD_CFLAGS}} ${{PIC_CFLAGS}} ${{NEON_CFLAGS}} "
        f"${{SSE_CFLAGS}} ${{VMX_CFLAGS}} ${{DEFS}} ${{CPPFLAGS}} ${{MYINCDIRS}} -o $@ -c $<",
        f"libhmmer-src.stamp: ${{OBJS}} {object_name}",
    ]
    _append(path, top_lines)
    # The existing rule's recipe must also run so $? still archives the normal
    # objects; prepend the shim archive to it rather than replacing it.
    patched = path.read_text(encoding="utf-8")
    assert archive_line in patched
    leading = archive_line[: len(archive_line) - len(archive_line.lstrip())]
    patched = patched.replace(
        archive_line,
        f"{leading}${{QUIET_AR}}${{AR}} -r libhmmer.a {object_name} > /dev/null 2>&1\n{archive_line}",
        1,
    )
    path.write_text(patched, encoding="utf-8")

    # Implementation sub-make: the define only (the shim object is compiled at the
    # top level, because that is where libhmmer.a is created).
    _append(
        impl_makefile,
        [
            f"\n{marker}",
            "# make runs this directory as a sub-make with its own CFLAGS, so the define",
            "# has to be repeated here or every P7_EVO_PROF_* macro compiles to a no-op.",
            f"CFLAGS += -D{PROF_DEFINE}",
        ],
    )
    return {
        **report,
        "already_enabled": False,
        "edited_makefiles": [str(path), str(impl_makefile)],
        "archive_patched": True,
    }


def scan_binary_symbols(binary: str | Path) -> dict[str, Any]:
    """Return the raw symbol views of a binary: global+local names, strings, and nm command.

    ``nm`` without ``-a`` hides local symbols, and a contract function that is
    ``static`` in its translation unit links into the program as a *local* symbol.
    Verification therefore reads ``nm -a``; the plain ``nm`` count is kept only so
    the report shows both numbers side by side instead of one ambiguous figure.
    """
    import shutil as _shutil
    import subprocess

    binary = Path(binary)
    if not binary.is_file():
        raise InstrumentationError(f"cannot verify instrumentation: binary not found: {binary}")
    nm = _shutil.which("nm")
    if nm is None:
        raise InstrumentationError("cannot verify instrumentation: nm is not available on this host")
    proc = subprocess.run([nm, "-a", str(binary)], capture_output=True, text=True)
    if proc.returncode != 0:
        raise InstrumentationError(
            f"cannot verify instrumentation: nm -a failed on {binary}: {(proc.stderr or '').strip()[:400]}"
        )
    plain = subprocess.run([nm, str(binary)], capture_output=True, text=True)
    names: set[str] = set()
    for line in proc.stdout.splitlines():
        parts = line.split()
        if not parts:
            continue
        names.add(parts[-1])
        # `nm -a` may print `name` or `name.0` for a localised symbol.
        names.add(parts[-1].split(".")[0])
    strings_bin = _shutil.which("strings")
    text = ""
    if strings_bin is not None:
        text = subprocess.run([strings_bin, str(binary)], capture_output=True, text=True).stdout
    return {
        "binary": str(binary),
        "nm": nm,
        "symbols": names,
        "nm_all_text": proc.stdout,
        "nm_plain_text": plain.stdout,
        "strings_text": text,
    }


def _symbol_present(scan: dict[str, Any], symbol: str) -> bool:
    if symbol in scan["symbols"]:
        return True
    return symbol in (scan.get("nm_all_text") or "") or symbol in (scan.get("strings_text") or "")


def verify_instrumented_binary(
    binary: str | Path,
    *,
    renamed_functions: tuple[str, ...] = (),
    shim_symbols: tuple[str, ...] = REQUIRED_SHIM_SYMBOLS,
) -> dict[str, Any]:
    """Fail unless a supposedly instrumented binary really contains the timers.

    A profiling bundle that compiles the wrappers but not the shim looks
    instrumented (the renamed bodies are there) and reports nothing at runtime.
    Checking the published artifact is the only place that mistake is visible, so
    the build refuses to publish when the symbols are absent.

    ``renamed_functions`` must not be empty when callers intend to check the
    wrappers: an empty list used to make this function report
    ``renamed_functions: []`` even for a binary that contained all five bodies,
    which silently reduced the gate to the shim check alone.
    """
    scan = scan_binary_symbols(binary)
    found_renamed = sorted(
        f"{name}{RENAMED_SUFFIX}" for name in renamed_functions if _symbol_present(scan, f"{name}{RENAMED_SUFFIX}")
    )
    missing_renamed = sorted(
        f"{name}{RENAMED_SUFFIX}" for name in renamed_functions if not _symbol_present(scan, f"{name}{RENAMED_SUFFIX}")
    )
    found_shim = sorted(symbol for symbol in shim_symbols if _symbol_present(scan, symbol))
    missing_shim = sorted(symbol for symbol in shim_symbols if not _symbol_present(scan, symbol))
    if missing_renamed or missing_shim:
        raise InstrumentationError(
            "profiling build did not reach the binary: "
            f"missing instrumented bodies {missing_renamed or '[]'}, missing shim symbols {missing_shim or '[]'} "
            f"in {scan['binary']}. The P7_EVO_PROF define most likely did not reach the implementation "
            "sub-Makefile, or the shim object was never archived."
        )
    if not renamed_functions:
        raise InstrumentationError(
            f"refusing to certify {scan['binary']} without an expected renamed-function list: "
            "an empty list cannot distinguish 'all wrappers present' from 'no wrapper was checked'"
        )
    return {
        "binary": scan["binary"],
        "renamed_functions": found_renamed,
        "shim_symbols": found_shim,
        "symbol_scan": {
            "nm": scan["nm"],
            "nm_all_seqalign_uninstrumented": sum(
                1 for line in scan["nm_all_text"].splitlines() if "seqalign_uninstrumented" in line
            ),
            "nm_plain_p7_evo_prof": sum(
                1 for line in scan["nm_plain_text"].splitlines() if "p7_evo_prof" in line
            ),
            "strings_p7_evo_prof": sum(1 for line in scan["strings_text"].splitlines() if "p7_evo_prof" in line),
        },
    }


def _registered_contracts() -> dict[str, Any]:
    """Every contract a replayed or freshly authored patch may name."""
    from .components.c_kernels import ALL_C_KERNEL_CONTRACTS  # noqa: PLC0415

    return dict(ALL_C_KERNEL_CONTRACTS)


def contract_entry_symbols(
    contracts: dict[str, Any] | None = None,
) -> dict[str, tuple[str, ...]]:
    """Return the declared entry symbols per contract island; empty means undeclared."""
    registry = dict(contracts) if contracts is not None else _registered_contracts()
    return {
        name: tuple(getattr(contract, "c_entry_functions", ()) or ())
        for name, contract in registry.items()
    }


def verify_contract_symbols(
    binaries: dict[str, str | Path],
    cfg: Any,
    *,
    contracts: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Verify that every entry symbol of every contract named by ``cfg.c_patches`` is
    present in the binaries it declares (``c_entry_binaries``).

    Symbols are read from ``nm -a`` because these entries are frequently ``static``.
    A contract that declares no symbols is reported under ``unverifiable_contracts``;
    a patch naming a contract outside the registry is reported under
    ``unresolved_contracts``. A symbol missing from a declared binary raises.
    """
    registry = dict(contracts) if contracts is not None else _registered_contracts()
    expected: list[str] = []
    expected_by_binary: dict[str, list[str]] = {}
    checked: list[str] = []
    unverifiable: list[str] = []
    unresolved: list[str] = []
    for spec in getattr(cfg, "c_patches", None) or ():
        name = str(getattr(spec, "contract_name", "") or "")
        contract = registry.get(name)
        if contract is None:
            if name and name not in unresolved:
                unresolved.append(name)
            continue
        symbols = tuple(getattr(contract, "c_entry_functions", ()) or ())
        if not symbols:
            if name not in unverifiable:
                unverifiable.append(name)
            continue
        if name not in checked:
            checked.append(name)
        required_in = tuple(getattr(contract, "c_entry_binaries", ()) or ()) or tuple(sorted(binaries))
        for binary in required_in:
            if binary not in binaries:
                continue
            for symbol in symbols:
                if symbol not in expected:
                    expected.append(symbol)
                slot = expected_by_binary.setdefault(binary, [])
                if symbol not in slot:
                    slot.append(symbol)
    report: dict[str, Any] = {
        "mode": "contract_entry_symbols",
        "checked_contracts": sorted(checked),
        "unverifiable_contracts": sorted(unverifiable),
        "unresolved_contracts": sorted(unresolved),
        "expected_symbols": expected,
        "expected_by_binary": {name: sorted(symbols) for name, symbols in sorted(expected_by_binary.items())},
        "binaries": {},
        "verified": bool(expected_by_binary),
    }
    if not expected_by_binary:
        report["skipped_reason"] = (
            "no contract touched by this patch set declares an entry symbol for any of these "
            "binaries, so nothing was scanned; this means 'nothing was checked', not 'the patch "
            "compiled in'"
        )
        return report
    missing: dict[str, list[str]] = {}
    for binary, symbols in sorted(expected_by_binary.items()):
        scan = scan_binary_symbols(binaries[binary])
        absent = sorted(symbol for symbol in symbols if not _symbol_present(scan, symbol))
        report["binaries"][binary] = {
            "binary": scan["binary"],
            "required": sorted(symbols),
            "missing": absent,
        }
        if absent:
            missing[binary] = absent
    if missing:
        raise InstrumentationError(
            "candidate bundle does not contain the contract entry symbols its own patches touch: "
            + "; ".join(f"{name} is missing {symbols}" for name, symbols in sorted(missing.items()))
            + ". The patched translation unit was never compiled into that binary, so any measured "
            "effect would be noise; refusing to certify the bundle."
        )
    return report


def verify_instrumented_artifacts(
    binaries: dict[str, str | Path],
    resolved_targets: Any,
    *,
    file_root: str | Path,
) -> dict[str, Any]:
    """Verify every published binary *and* the instrumented sources they came from.

    The source check is the reason verification cannot pass on a stale cache entry:
    if the cached source trees do not actually carry ``*__seqalign_uninstrumented``
    the bundle was produced by code that instrumented nothing (the 2026-10-08
    failure), and no amount of ``-DP7_EVO_PROF`` in a manifest would be true.
    """
    file_root = Path(file_root)
    renamed = tuple(item[1] for item in (resolved_targets or ()))
    report: dict[str, Any] = {"binaries": {}, "source_bodies": {}}
    for name, path in sorted(binaries.items()):
        report["binaries"][name] = verify_instrumented_binary(path, renamed_functions=renamed)
    for item in resolved_targets or ():
        rel_path, function = item[0], item[1]
        source = file_root / rel_path
        if not source.is_file():
            raise InstrumentationError(
                f"cannot verify instrumentation: instrumented source missing from the bundle: {source}"
            )
        text = source.read_text(encoding="utf-8", errors="replace")
        body = f"{function}{RENAMED_SUFFIX}"
        report["source_bodies"][f"{rel_path}:{function}"] = {
            "file": str(source),
            "body_symbol": body,
            "found": body in text,
        }
        if body not in text:
            raise InstrumentationError(
                f"instrumented source does not define {body}: {source} is not an instrumented tree, "
                "so the bundle cannot be a valid profiling build"
            )
    return report
