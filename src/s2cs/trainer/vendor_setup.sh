#!/usr/bin/env bash
# Vendors a compatible verl source tree into a gitignored directory and applies
# the S3 integration patches. The vendored tree is importable as `verl`
# (via tests/trainer/conftest.py path injection in dev, or a full install
# after running this setup script).
set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
VENDOR="$REPO_ROOT/src/s2cs/trainer/vendor/verl_patched"
SRC="${VERL_SRC:-vendor/verl}"
PATCHES="$REPO_ROOT/src/s2cs/trainer/patches"

mkdir -p "$(dirname "$VENDOR")"
if [ ! -e "$VENDOR/verl/__init__.py" ]; then
  [ -d "$SRC" ] || { echo "verl source not found: $SRC (set VERL_SRC)"; exit 1; }
  rsync -a --exclude='.git' "$SRC/" "$VENDOR/"
  echo "vendored fork -> $VENDOR"
fi

# Apply SEAL graft patches idempotently (patches/ is empty until the SEAL task).
shopt -s nullglob
for p in "$PATCHES"/*.patch; do
  if patch -p1 -d "$VENDOR" --forward --reverse --dry-run -i "$p" >/dev/null 2>&1; then
    echo "already applied: $(basename "$p")"
  else
    patch -p1 -d "$VENDOR" --forward -i "$p" || { echo "patch failed: $p"; exit 1; }
    echo "applied: $(basename "$p")"
  fi
done

# Editable-install the vendored verl as a real package (--no-deps: the venv
# already satisfies torch/sglang/ray via the `trainer` group; we must NOT pull
# the fork's torch-2.6/vllm pins). This makes `verl` importable in Ray workers,
# not just via the test conftest path injection.
( cd "$REPO_ROOT" && uv pip install -e "$VENDOR" --no-deps -q )

# Neutralize sglang's non-CUDA vllm fallback (we always run on CUDA; vllm absent).
# Patches the installed sglang package, so re-run after any `uv sync`.
( cd "$REPO_ROOT" && uv run --no-sync python "$PATCHES/sglang_no_vllm_fallback.py" )

# Stop sglang force-defaulting compatible bf16 DeepSeek-architecture checkpoints
# to fp8 on Blackwell. Patches the
# installed sglang package, so re-run after any `uv sync`.
( cd "$REPO_ROOT" && uv run --no-sync python "$PATCHES/sglang_no_deepseek_fp8_default.py" )

echo "vendor ready at $VENDOR"
