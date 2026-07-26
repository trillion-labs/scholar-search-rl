"""Idempotently neutralize sglang's non-CUDA vllm fallback in activation.py.

We always run on CUDA, so the native SiluAndMul/GeluAndMul classes defined in
sglang's activation.py are sufficient. The module's non-CUDA fallback does
`from vllm.model_executor.layers.activation import GeluAndMul, SiluAndMul`,
which crashes when vllm is absent. That branch is reached in a GPU-less driver
process (where sglang's lru_cached is_cuda() is falsely False at import) — the
real rollout workers are on CUDA. Wrapping the import in try/except keeps the
native classes and unblocks the import.

This patches a pip-installed package, so re-run after any `uv sync` that
reinstalls sglang. Invoked by vendor_setup.sh.
"""

import os

import sglang

MARKER = "s2cs: vllm is not installed"
_OLD = """    from vllm.model_executor.layers.activation import (  # noqa: F401
        GeluAndMul,
        SiluAndMul,
    )"""
_NEW = """    try:
        from vllm.model_executor.layers.activation import (  # noqa: F401
            GeluAndMul,
            SiluAndMul,
        )
    except ImportError:
        # {marker}: native SiluAndMul/GeluAndMul (above) remain in effect.
        pass""".format(marker=MARKER)


def main() -> None:
    path = os.path.join(os.path.dirname(sglang.__file__), "srt", "layers", "activation.py")
    src = open(path).read()
    if MARKER in src:
        print(f"sglang activation already patched: {path}")
        return
    if _OLD not in src:
        raise SystemExit(f"sglang activation.py shape changed; update this patch ({path})")
    open(path, "w").write(src.replace(_OLD, _NEW))
    print(f"patched sglang activation.py: {path}")


if __name__ == "__main__":
    main()
