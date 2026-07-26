"""Idempotently stop sglang from force-defaulting DeepSeek-arch models to fp8.

sglang 0.5.7 `ServerArgs._handle_model_specific_adjustments` (server_args.py),
on a Blackwell/SM100 GPU, sets `self.quantization = "fp8"` for ANY
`DeepseekV3ForCausalLM` checkpoint that does not carry a `quantization_config`
-- on the assumption it is a native-fp8 DeepSeek-V3/R1 release.

Some DeepSeek-V3-architecture MoE checkpoints are shipped in **bf16** without
fp8 scales. Letting sglang force fp8 would online-quantize them at load, so an
sglang rollout policy could diverge from a bf16 FSDP actor -- a large
rollout-vs-train logprob gap that GRPO/TIS cannot absorb. There
is no `quantization` value we can pass to avoid this: `None` triggers this very
default, and any explicit string ("none"/"") is rejected by ModelConfig's
`supported_quantization` check. So we neutralize the default in place: keep the
detected `quant_method` (None => bf16).

This patches a pip-installed package, so re-run after any `uv sync` that
reinstalls sglang. Invoked by vendor_setup.sh.
"""

import os

import sglang

MARKER = "s2cs: do NOT default DeepSeek-arch to fp8"
_OLD = """                    if quant_method is None and model_arch in ["DeepseekV3ForCausalLM"]:
                        self.quantization = "fp8"
                        logger.info(
                            "Quantization not specified, default to fp8 for DeepSeek on sm100"
                        )
                    else:
                        self.quantization = quant_method"""
_NEW = """                    # {marker} on sm100. A bf16 DeepSeek-V3-architecture
                    # checkpoint may have no fp8 scales; forcing
                    # fp8 would online-quantize it and diverge the sglang rollout
                    # policy from the bf16 FSDP actor (breaks GRPO/TIS). Keep the
                    # detected quant_method (None => bf16).
                    self.quantization = quant_method""".format(marker=MARKER)


def main() -> None:
    path = os.path.join(os.path.dirname(sglang.__file__), "srt", "server_args.py")
    src = open(path).read()
    if MARKER in src:
        print(f"sglang fp8 default already neutralized: {path}")
        return
    if _OLD not in src:
        raise SystemExit(f"sglang server_args.py shape changed; update this patch ({path})")
    open(path, "w").write(src.replace(_OLD, _NEW))
    print(f"neutralized sglang DeepSeek fp8 default: {path}")


if __name__ == "__main__":
    main()
