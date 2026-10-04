# Qwen3.8 implementation provenance

The owned prototype is in `engine/qwen38_runner/`. It has no SGLang/vLLM
runtime imports. GPU leaf dependencies: Torch, Triton, FlashInfer.
The safetensors loader uses CPU mappings only at startup.

Equation/layout reference: SGLang `v0.5.20`, commit
`94602c9c2b7cbdb8efd5c52802dac6a1c180089e`, Apache-2.0:
license text: [QWEN38_APACHE_LICENSE](QWEN38_APACHE_LICENSE).

- `srt/models/qwen4_exp.py`: PLE history/hash/table partitioning, gated residual
  ordering, final hyperconnection mixer.
- `srt/models/qwen3_5.py`: GDN sharding and gates, QSA Q/gate head layout.
- `srt/layers/hyperconnection.py`: mix/combine and per-branch Gemma RMSNorm.
- `srt/layers/quantization/modelopt_quant.py`: NVFP4 input/global scales.
- `srt/layers/moe/moe_runner/flashinfer_cutlass.py`: leaf kernel calling convention.
- `python/sglang/kernels/ops/attention/fla/fused_recurrent.py`: BF16 gate
  rounding and packed-decode Q/K, FP32 state, output-reduction order.

These are mathematical/layout adaptations, not imports or copies of the
SGLang scheduler/model runner. Apache headers are retained in adapted files.
The prototype has passed initial eight-rank persistent-tensor residency checks
and graph capture/cleanup. After materializing the BF16 GDN gate and matching
the packed-decode recurrent output order, 64- and 128-token free generation
match the deterministic references on one prompt. Under teacher forcing,
64/64 and 128/128 top1 choices match, but the maximum selected-token logprob
differences remain 0.144327 and 0.274001 respectively.
This is not full correctness/serving acceptance.

Strata research reference: commit `99f3dbd0b21d1401b3769e0c0d963913607f380b`,
MIT. No Strata code is copied in this prototype. No CPU/SSD expert execution,
speculation, or host-resident PLE table is used.
