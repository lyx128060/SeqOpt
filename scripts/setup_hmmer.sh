#!/usr/bin/env bash
# Fetch, patch and build HMMER 3.4 so the engine has a real source tree to patch.
#
# Why this is a script and not a vendored tree
# --------------------------------------------
# The engine edits real HMMER C sources and rebuilds them, so it needs a pristine
# HMMER 3.4 checkout at ./tools/hmmer-3.4/. That tree is ~98 MB and is HMMER's own
# copyrighted work under a BSD-3-Clause licence: it belongs with HMMER, not with
# this repository. Shipping it here would both bloat the clone and re-distribute a
# third-party project that already has a home.
#
# What this script does
# ---------------------
#   1. downloads the official HMMER 3.4 release tarball and checks its SHA-256;
#   2. unpacks it to ./tools/hmmer-3.4/;
#   3. applies patches/*.patch (the optimization-island exposure this project owns);
#   4. runs configure + make.
#
# After it succeeds, the tests marked `requires_hmmer_source` stop being skipped.
#
# Usage:
#   bash scripts/setup_hmmer.sh              # download + patch + build
#   bash scripts/setup_hmmer.sh --no-build   # download + patch only
#   bash scripts/setup_hmmer.sh --tarball /path/to/hmmer-3.4.tar.gz
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HMMER_DIR="$REPO_ROOT/tools/hmmer-3.4"
PATCH_DIR="$REPO_ROOT/patches"

# Official release: http://eddylab.org/software/hmmer/hmmer-3.4.tar.gz
URL="${HMMER_URL:-https://eddylab.org/software/hmmer/hmmer-3.4.tar.gz}"
# SHA-256 of the release tarball as distributed by eddylab.org.
EXPECTED_SHA256="ca70d94fd0cf271bd7063423aabb116d42de533117343a9b27a65c17ff06fbf3"

TARBALL=""
DO_BUILD=1
while [ $# -gt 0 ]; do
  case "$1" in
    --tarball) TARBALL="$2"; shift 2 ;;
    --no-build) DO_BUILD=0; shift ;;
    -h|--help) sed -n '2,26p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

if [ -d "$HMMER_DIR/src" ]; then
  echo "tools/hmmer-3.4/ already present; nothing to do."
  echo "Delete it first if you want a clean rebuild."
  exit 0
fi

mkdir -p "$REPO_ROOT/tools"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

if [ -n "$TARBALL" ]; then
  echo "==> using local tarball: $TARBALL"
  cp "$TARBALL" "$WORK/hmmer-3.4.tar.gz"
else
  echo "==> downloading $URL"
  if command -v curl >/dev/null 2>&1; then
    curl -fSL --retry 3 -o "$WORK/hmmer-3.4.tar.gz" "$URL"
  elif command -v wget >/dev/null 2>&1; then
    wget -O "$WORK/hmmer-3.4.tar.gz" "$URL"
  else
    echo "ERROR: neither curl nor wget is available" >&2
    exit 1
  fi
fi

echo "==> verifying SHA-256"
ACTUAL="$(sha256sum "$WORK/hmmer-3.4.tar.gz" | awk '{print $1}')"
if [ "$ACTUAL" != "$EXPECTED_SHA256" ]; then
  echo "ERROR: tarball checksum mismatch" >&2
  echo "  expected $EXPECTED_SHA256" >&2
  echo "  actual   $ACTUAL" >&2
  echo "Refusing to build an unverified HMMER tree." >&2
  exit 1
fi
echo "    ok ($ACTUAL)"

echo "==> unpacking"
tar xzf "$WORK/hmmer-3.4.tar.gz" -C "$WORK"
mv "$WORK/hmmer-3.4" "$HMMER_DIR"

if compgen -G "$PATCH_DIR/*.patch" > /dev/null; then
  echo "==> applying optimisation-island patches"
  for p in "$PATCH_DIR"/*.patch; do
    echo "    $(basename "$p")"
    ( cd "$HMMER_DIR" && patch -p1 --forward --silent < "$p" )
  done
fi

# The engine resolves a contract's ``c_target_file`` (for example
# ``impl_neon/fwdback.c``) against several candidate paths, so that a deployment
# which keeps its patchable sources one level deeper keeps working. Make that
# deeper view available here too: the profiling resolver reads the *generated*
# src/Makefile to locate the implementation directories, and the two views have
# to reconcile.
#
# ``src/src -> .`` mirrors the deployment tree: src/ is the build directory and
# src/src/ is where the patchable C sources appear to live.
ensure_nested_source_view() {
  local nested="$HMMER_DIR/src/src"
  if [ -e "$nested" ] || [ -L "$nested" ]; then
    echo "    nested source view already present"
    return
  fi
  ln -s . "$nested"
  echo "    created src/src -> . (nested source view used by contract resolution)"
}

if [ "$DO_BUILD" -eq 1 ]; then
  echo "==> configure"
  ( cd "$HMMER_DIR" && ./configure --enable-threads > /dev/null )
  echo "==> make (this takes a few minutes)"
  ( cd "$HMMER_DIR" && make -j"$(nproc)" > /dev/null )
  echo "==> exposing the nested source view"
  ensure_nested_source_view
  echo
  echo "Built binaries:"
  ls -1 "$HMMER_DIR/src/jackhmmer" "$HMMER_DIR/src/hmmsearch" 2>/dev/null | sed 's/^/  /'
else
  echo "==> build skipped; creating the nested source view anyway"
  ensure_nested_source_view
fi

echo
echo "Done. tools/hmmer-3.4/ is ready; the requires_hmmer_source tests will now run."
