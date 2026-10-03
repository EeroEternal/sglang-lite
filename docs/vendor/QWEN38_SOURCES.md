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

These are mathematical/layout adaptations, not imports or copies of the
SGLang scheduler/model runner. Apache headers are retained in adapted files.
The prototype has passed initial eight-rank persistent-tensor residency checks
and graph capture/cleanup. Its first 43 generated tokens match the deterministic
reference; 64-token free generation diverges at zero-based token 43. Under
teacher forcing, 63/64 top1 choices match and the maximum selected-token
logprob difference is 0.279. This is not full correctness/serving acceptance.

Strata research reference: commit `99f3dbd0b21d1401b3769e0c0d963913607f380b`,
MIT. No Strata code is copied in this prototype. No CPU/SSD expert execution,
speculation, or host-resident PLE table is used.
