"""MoEModelRunner: batched HF forward with paged KV as attention source of truth."""

from __future__ import annotations

from collections import defaultdict
from types import SimpleNamespace
from typing import Dict, List, Optional

import torch

from .cuda_graph import (
    DecodeInputBuffer,
    HfStaticDecodeCudaGraph,
    PagedDecodeCudaGraph,
    experts_implementation_from_env,
    maybe_compile_model,
)
from .kernel_backend import PagedAttnContext, create_kernel_backend
from .kv_cache import PastKV, RadixCache, cache_is_hybrid
from .models import assert_moe_supported, is_fixture_model, register_verified
from .moe_hooks import maybe_attach_fused_moe
from .native_decode import native_decode_enabled, try_build_paged_native
from .sampling import make_generator, sample_logits
from .scheduler import Sequence


class MoERouter:
    """Placeholder router; real MoE routing is inside the HF model."""

    def route(self, input_ids: List[int]) -> List[int]:
        return [0] * len(input_ids)


class ModelRunner:
    def __init__(
        self,
        model_name: str = "stub",
        device: str = "cpu",
        max_batch: int = 4,
        allow_stub: bool = False,
    ):
        self.model_name = model_name
        self.device = device
        self.max_batch = max_batch
        self.allow_stub = allow_stub

        self.model = None
        self.tokenizer = None
        self._is_real = False
        self.vocab_size = 32000
        self.num_layers = 4
        self.num_kv_heads = 4
        self.head_dim = 64
        self.eos_token_id: Optional[int] = 2
        # Observability for tests: size of the last tensor forward (batch dim)
        self.last_model_forward_size = 0
        self.model_forward_count = 0
        self.paged_rebuild_count = 0
        self.use_paged_as_source = True
        self._v4_hybrid = False
        self._v4_prefix_cache = None
        self._v4_accel = None
        self._tp_plan = None
        self._kv_layout = None
        self._swa_layout = None
        self._decode_input_buf = DecodeInputBuffer(device)
        self._paged_decode_cg = PagedDecodeCudaGraph(device)
        self._hf_static_cg: Optional[HfStaticDecodeCudaGraph] = None
        self._paged_native = None  # lean decode body (no HF CausalLM.forward)
        self._paged_burst_toks: Optional[torch.Tensor] = None  # GPU token ring for thruput
        self._experts_impl: Optional[str] = None
        # Must match model compute dtype; fp16 pages + bf16 activations promote to fp32 in SDPA.
        self.torch_dtype = torch.float32 if self.device == "cpu" else torch.bfloat16
        self.kernel_backend = create_kernel_backend(self.device)

        if model_name == "stub":
            if not allow_stub:
                raise ValueError(
                    "model='stub' is only allowed with allow_stub=True "
                    "(explicit demo/test mode). Production must load a real MoE model."
                )
            self._init_tiny_stub_model()
        else:
            self._load_real(model_name)

    def _load_real(self, model_name: str) -> None:
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

        assert_moe_supported(model_name)

        if is_fixture_model(model_name):
            path = model_name.split(":", 1)[1]
            load_id = path
        else:
            load_id = model_name

        print(f"[sglang-lite] Loading MoE model: {load_id}")
        # Path/name may already identify deepseek_v4 before AutoConfig (older
        # transformers builds do not register model_type=deepseek_v4).
        fam_from_id = None
        try:
            fam_from_id = assert_moe_supported(model_name)
        except ValueError:
            fam_from_id = None

        config = None
        model_type = None
        if fam_from_id is None or fam_from_id.name != "deepseek_v4":
            try:
                config = AutoConfig.from_pretrained(load_id, trust_remote_code=True)
                model_type = getattr(config, "model_type", None)
            except ValueError:
                config = None
                model_type = None
        if config is None and (
            (fam_from_id and fam_from_id.name == "deepseek_v4")
            or "deepseek_v4" in model_name.lower()
            or "deepseek-v4" in model_name.lower()
            or "ds-v4" in model_name.lower()
        ):
            import json
            from pathlib import Path
            from types import SimpleNamespace

            raw = json.loads((Path(load_id) / "config.json").read_text(encoding="utf-8"))
            model_type = raw.get("model_type", "deepseek_v4")
            config = SimpleNamespace(**raw)

        fam = assert_moe_supported(model_name, model_type)

        self.tokenizer = AutoTokenizer.from_pretrained(
            load_id, trust_remote_code=True, use_fast=True
        )
        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        if fam.name == "deepseek_v4" or (model_type or "").lower() == "deepseek_v4":
            self._load_deepseek_v4_hybrid(model_name, load_id, config)
            return

        dtype = self.torch_dtype
        load_kwargs = {
            "dtype": dtype,
            "trust_remote_code": True,
            "low_cpu_mem_usage": True,
        }
        # MoE expert backend: batched_mm is ~2× grouped_mm and CUDA-graph safe.
        # Prefer env; when FORCE_HF_CACHE thruput path, default to batched_mm.
        import os as _os_exp

        exp_impl = experts_implementation_from_env()
        force_hf_early = _os_exp.environ.get("SGLANG_LITE_FORCE_HF_CACHE", "").lower() in (
            "1",
            "true",
            "yes",
        )
        if exp_impl is None and force_hf_early:
            exp_impl = "batched_mm"
        if exp_impl:
            load_kwargs["experts_implementation"] = exp_impl
            self._experts_impl = exp_impl
        use_device_map = False
        if self.device != "cpu":
            try:
                import accelerate  # noqa: F401

                load_kwargs["device_map"] = "auto"
                use_device_map = True
            except ImportError:
                use_device_map = False
        try:
            self.model = AutoModelForCausalLM.from_pretrained(load_id, **load_kwargs)
        except (TypeError, ValueError) as e:
            # Older TF or unsupported impl — retry without experts_implementation.
            if "experts_implementation" in load_kwargs:
                print(
                    f"[sglang-lite] experts_implementation={exp_impl!r} rejected "
                    f"({e}); falling back to library default"
                )
                load_kwargs.pop("experts_implementation", None)
                self._experts_impl = None
                self.model = AutoModelForCausalLM.from_pretrained(load_id, **load_kwargs)
            else:
                raise
        if not use_device_map:
            self.model = self.model.to(self.device)
        self.model.eval()
        self._is_real = True
        if self._experts_impl:
            print(f"[sglang-lite] MoE experts_implementation={self._experts_impl}")
        self.vocab_size = int(self.model.config.vocab_size)
        self.num_layers = int(self.model.config.num_hidden_layers)
        self.num_kv_heads = int(
            getattr(
                self.model.config,
                "num_key_value_heads",
                self.model.config.num_attention_heads,
            )
        )
        self.head_dim = int(
            getattr(self.model.config, "head_dim", None)
            or (self.model.config.hidden_size // self.model.config.num_attention_heads)
        )
        self.eos_token_id = getattr(self.tokenizer, "eos_token_id", None) or getattr(
            self.model.config, "eos_token_id", 2
        )

        # MLA (DeepSeek-V2/V3 style compressed KV) is incompatible with standard
        # FlashInfer paged MHA hooks. Keep HF DynamicCache path until MLA leaf lands.
        cfg_obj = self.model.config
        is_mla = (
            getattr(cfg_obj, "kv_lora_rank", None) is not None
            or getattr(cfg_obj, "q_lora_rank", None) is not None
            or (model_type or "").lower() in ("deepseek_v2", "deepseek_v3")
        )
        # FlashInfer paged decode rejects some GQA ratios (e.g. MiniMax 48/8 → 6).
        n_heads = int(self.model.config.num_attention_heads)
        n_kv = int(self.num_kv_heads) or n_heads
        gqa_group = max(1, n_heads // max(n_kv, 1))
        fi_gqa_ok = gqa_group in (1, 2, 4, 8, 16, 32, 64)
        mt_lower = (model_type or "").lower()
        is_minimax = mt_lower.startswith("minimax") or fam.name == "minimax_moe"
        text_cfg = cfg_obj
        getter = getattr(cfg_obj, "get_text_config", None)
        if callable(getter):
            try:
                text_cfg = getter(decoder=True)
            except Exception:
                text_cfg = cfg_obj
        layer_types = list(getattr(text_cfg, "layer_types", None) or [])
        is_hybrid_linear = any(
            t in ("linear_attention", "mamba", "conv") for t in layer_types
        )
        skip_paged = is_mla or is_minimax or not fi_gqa_ok or is_hybrid_linear
        print(f"[sglang-lite] Kernel backend: {self.kernel_backend.name}")
        import os as _os

        force_hf_cache = _os.environ.get("SGLANG_LITE_FORCE_HF_CACHE", "").lower() in (
            "1",
            "true",
            "yes",
        )
        # PRO6000 profile: FI paged hooks can be ~2× slower than native HF SDPA
        # for single-stream Qwen3-MoE (append+plan tax). Prefer HF cache when forced
        # or when env asks for thruput path.
        if force_hf_cache:
            self.use_paged_as_source = False
            print(
                "[sglang-lite] SGLANG_LITE_FORCE_HF_CACHE=1 — HF use_cache path "
                "(no FlashInfer paged hooks)"
            )
            # Experimental: StaticCache + CUDA graph. Qwen3-MoE StaticCache drifts
            # vs DynamicCache after ~14 tokens — off by default. batched_mm alone
            # already matches generate (~46 tok/s). Enable only for research:
            #   SGLANG_LITE_HF_STATIC_GRAPH=1
            if (
                str(self.device).startswith("cuda")
                and self._experts_impl == "batched_mm"
                and _os.environ.get("SGLANG_LITE_HF_STATIC_GRAPH", "").lower()
                in ("1", "true", "yes")
            ):
                max_len = int(
                    _os.environ.get("SGLANG_LITE_STATIC_CACHE_LEN", "4096")
                )
                self._hf_static_cg = HfStaticDecodeCudaGraph(
                    self.device, max_cache_len=max_len
                )
                print(
                    f"[sglang-lite] HF StaticCache CUDA-graph decode armed "
                    f"(experimental, max_cache_len={max_len})"
                )
        elif skip_paged:
            self.use_paged_as_source = False
            reason = (
                "hybrid linear attention"
                if is_hybrid_linear
                else (
                    "MLA"
                    if is_mla
                    else (
                        "MiniMax/custom"
                        if is_minimax
                        else f"GQA group_size={gqa_group} unsupported by FI paged"
                    )
                )
            )
            print(
                f"[sglang-lite] {reason} — skip standard paged MHA hooks "
                "(HF use_cache path)"
            )
        elif self.kernel_backend.supports_paged_attention:
            sm_scale = self.head_dim**-0.5
            n = self.kernel_backend.attach_to_model(
                self.model, num_qo_heads=n_heads, head_dim=self.head_dim, sm_scale=sm_scale
            )
            print(f"[sglang-lite] Paged attention hooks attached: {n} modules")

        # Fused MoE (FlashInfer cutlass) — before torch.compile so graph-breaks
        # land on the custom op boundary. Auto-on for FORCE_HF thruput path.
        self._fused_moe_n = maybe_attach_fused_moe(
            self.model, force_hf_cache=force_hf_cache
        )

        # Phase-A: optional torch.compile. Prefer on for HF batched_mm thruput
        # (~80 tok/s vs ~47 eager on Qwen3-MoE). Skip true multi-GPU shards only.
        dmap = getattr(self.model, "hf_device_map", None)
        multi_gpu_shard = False
        if isinstance(dmap, dict) and dmap:
            devices = {str(v) for v in dmap.values()}
            multi_gpu_shard = len(devices) > 1
        paged_hooks = bool(
            self.use_paged_as_source
            and getattr(self.kernel_backend, "supports_paged_attention", False)
        )
        prefer_compile = (
            not self.use_paged_as_source
            and self._experts_impl == "batched_mm"
            and str(self.device).startswith("cuda")
        )
        compiled = self.model
        if not multi_gpu_shard:
            compiled = maybe_compile_model(
                self.model,
                self.device,
                paged_hooks=paged_hooks,
                prefer_compile=prefer_compile,
            )
        elif prefer_compile:
            print(
                "[sglang-lite] torch.compile skipped (multi-GPU device_map shard)"
            )
        self._compile_preferred = compiled is not self.model
        self.model = compiled
        self._decode_input_buf = DecodeInputBuffer(self.device)
        # Lean paged decode: walk HF layers without CausalLM.forward / causal mask.
        # Build after hooks + compile so we reference the live modules.
        self._paged_native = None
        if (
            self.use_paged_as_source
            and paged_hooks
            and native_decode_enabled(default="1")
        ):
            self._paged_native = try_build_paged_native(self.model)
            if self._paged_native is not None:
                print(
                    "[sglang-lite] paged native decode enabled "
                    "(layer loop, no HF CausalLM.forward)"
                )
        # Growable ones-mask for HF path (avoid per-step host alloc of new tensors).
        self._attn_mask_buf: Optional[torch.Tensor] = None
        # Throughput: allow TF32 tensor cores on Ampere+ (safe for thruput probes).
        if str(self.device).startswith("cuda") and torch.cuda.is_available():
            try:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                torch.set_float32_matmul_precision("high")
                # Prefer Flash / mem-efficient SDPA kernels when available.
                torch.backends.cuda.enable_flash_sdp(True)
                torch.backends.cuda.enable_mem_efficient_sdp(True)
                torch.backends.cuda.enable_math_sdp(True)
            except Exception:
                pass

        # Torch.compile pays on first unique shapes — warm decode shapes at load
        # so the first user request is not a 100s+ cold compile (thruput path).
        if self._compile_preferred and str(self.device).startswith("cuda"):
            self._warmup_hf_compile_decode()

        register_verified(model_name)
        print(f"[sglang-lite] MoE model '{model_name}' ready on {self.device}")

    @torch.inference_mode()
    def _warmup_hf_compile_decode(self) -> None:
        """Run a few static-shape decode steps to trigger inductor kernels."""
        try:
            dev = self.device
            ids = torch.zeros((1, 1), dtype=torch.long, device=dev)
            # Use a non-pad token when available.
            if self.eos_token_id is not None:
                ids[0, 0] = int(self.eos_token_id) % max(self.vocab_size, 1)
            kwargs: Dict = {"input_ids": ids, "use_cache": True}
            if self._supports_logits_to_keep():
                kwargs["logits_to_keep"] = 1
            out = self.model(**kwargs)
            past = out.past_key_values
            pos = 1
            tok_buf = torch.zeros((1, 1), dtype=torch.long, device=dev)
            pos_buf = torch.zeros((1, 1), dtype=torch.long, device=dev)
            for _ in range(6):
                tok_buf[0, 0] = out.logits[0, -1].argmax()
                pos_buf[0, 0] = pos
                kw = {
                    "input_ids": tok_buf,
                    "past_key_values": past,
                    "use_cache": True,
                    "position_ids": pos_buf,
                    "cache_position": pos_buf.reshape(-1),
                }
                if self._supports_logits_to_keep():
                    kw["logits_to_keep"] = 1
                out = self.model(**kw)
                past = out.past_key_values
                pos += 1
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            print("[sglang-lite] torch.compile decode warmup done")
        except Exception as e:
            print(f"[sglang-lite] torch.compile warmup skipped: {e}")

    def _load_deepseek_v4_hybrid(self, model_name: str, load_id: str, config) -> None:
        """Hybrid: official/vendor Transformer + TP shard from convert.py.

        Graph load prefers in-tree ``vendor/deepseek_infer`` (v4_runner), never
        runtime ``import sglang`` / ``import vllm``. Product gate: V4-Flash only
        (escape ``SGLANG_LITE_V4_ONLY=0`` for legacy experiments).
        """
        import os

        from .kv_cache import KvLayout
        from .model_loader import resolve_v4_paths
        from .v4_runner import load_v4_flash

        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        paths = resolve_v4_paths(hf_ckpt=load_id)
        if paths.converted_ckpt is None:
            raise RuntimeError(
                "deepseek_v4 Hybrid load requires converted MP shards. "
                "Run scripts/v4_official_smoke.sh then set "
                "SGLANG_LITE_DSV4_CONVERTED=/tmp/ds-v4-mp8 (and torchrun for TP)."
            )
        if world_size < 2:
            raise RuntimeError(
                "deepseek_v4 requires tensor parallel (WORLD_SIZE>=2, typically 8). "
                "Use: torchrun --nproc-per-node=8 scripts/v4_lite_engine_gen.py "
                "or scripts/v4_lite_short_gen.py"
            )
        model, cfg, plan, graph_src = load_v4_flash(
            model_id=load_id,
            paths=paths,
            max_batch_size=self.max_batch,
        )
        print(
            f"[sglang-lite] v4 graph source kind={graph_src.kind} path={graph_src.path}"
        )
        self.model = model
        self.device = plan.local_device
        self.torch_dtype = torch.bfloat16
        self._is_real = True
        self._v4_hybrid = True
        from .v4_prefix_cache import V4PrefixCache
        from .v4_runner import V4DecodeAccelerator

        self._v4_prefix_cache = V4PrefixCache()
        self._v4_accel = V4DecodeAccelerator(
            model, device=plan.local_device, max_batch=self.max_batch
        )
        self._tp_plan = plan
        self.vocab_size = int(cfg.get("vocab_size", 129280))
        self.num_layers = int(cfg.get("n_layers") or cfg.get("num_hidden_layers", 43))
        self.num_kv_heads = int(cfg.get("num_key_value_heads", 1))
        self.head_dim = int(cfg.get("head_dim", 512))
        self.eos_token_id = getattr(self.tokenizer, "eos_token_id", None) or int(
            cfg.get("eos_token_id", 1)
        )
        # V4 uses MLA / sparse pools — standard MHA hooks do not apply.
        self.use_paged_as_source = False
        self._kv_layout = KvLayout.mla_compressed(
            ckv_dim=int(cfg.get("head_dim", 512)),
            kpe_dim=int(cfg.get("rope_head_dim") or cfg.get("qk_rope_head_dim", 64)),
        )
        self._swa_layout = KvLayout.dsv4_packed(584)
        kb = self.kernel_backend
        # Match official generate.py determinism for greedy numerical checks.
        torch.manual_seed(33377335)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(33377335)
        # Refresh capabilities after Hybrid import path may have put FI≥0.6.16
        # on sys.path (SGLANG_LITE_FI_PREFIX). Then arm sparse MLA hook.
        # Official-only when DISABLE_FI and no torch preference.
        # Default product path: SGLANG_LITE_V4_SPARSE=auto → torch sparse (faster
        # than FI pack on PRO6000). TileLang when SGLANG_LITE_V4_SPARSE=official.
        disable_fi = os.environ.get("SGLANG_LITE_V4_DISABLE_FI_SPARSE", "").lower() in (
            "1",
            "true",
            "yes",
        )
        sparse_pref = os.environ.get("SGLANG_LITE_V4_SPARSE", "auto").strip().lower()
        try:
            from .capability import probe_kernel_capabilities
            from .kernel_backend import FlashInferBackend
            from .v4_sparse_mla import attach_v4_sparse_mla

            from .capability import SparseMlaBackend

            caps = probe_kernel_capabilities(self.device)
            if isinstance(kb, FlashInferBackend):
                kb.capabilities = caps
                kb.supports_sparse_mla = caps.sparse_mla_backend in (
                    SparseMlaBackend.FLASHINFER_SPARSE_SM120,
                    SparseMlaBackend.FLASHINFER_SPARSE_SM100,
                )
            win = int(cfg.get("window_size", 128))
            # Torch path ignores DISABLE_FI (that flag only blocks FI leaf).
            force_official = sparse_pref in ("official", "tilelang")
            if force_official:
                print(
                    "[sglang-lite] v4 sparse: official TileLang "
                    f"(SGLANG_LITE_V4_SPARSE={sparse_pref})"
                )
                armed = False
            else:
                # Even with DISABLE_FI=1, auto/torch still arm torch sparse.
                if disable_fi and sparse_pref in ("fi", "flashinfer"):
                    print(
                        "[sglang-lite] v4 FI sparse disabled "
                        "(SGLANG_LITE_V4_DISABLE_FI_SPARSE=1)"
                    )
                    armed = False
                else:
                    armed = attach_v4_sparse_mla(kb, window_size=win)
            print(
                f"[sglang-lite] v4 sparse MLA hook armed={armed} "
                f"backend={kb.sparse_mla_backend.value} pref={sparse_pref}"
            )
        except Exception as e:
            print(f"[sglang-lite] v4 sparse MLA hook skipped: {e}")
        # MoE: activated-expert dispatch + fused FP4 act_quant (decode thruput).
        try:
            from .v4_moe_fast import attach_v4_moe_fast

            attach_v4_moe_fast(self.model)
        except Exception as e:
            print(f"[sglang-lite] v4 MoE fast skipped: {e}")
        # FP4 GEMM: DeepGEMM SM120 (official MXFP4 e8m0 1x32) — main thruput lever.
        try:
            from .v4_deep_gemm import attach_v4_deep_gemm

            attach_v4_deep_gemm(self.model)
        except Exception as e:
            print(f"[sglang-lite] v4 DeepGEMM skipped: {e}")
        # B12x: probe only by default (NVFP4 layout + no EP; not Hybrid-compatible).
        try:
            from .v4_b12x import attach_v4_b12x

            attach_v4_b12x(self.model, world_size=int(plan.world_size))
        except Exception as e:
            print(f"[sglang-lite] v4 B12x skipped: {e}")
        print(
            f"[sglang-lite] deepseek_v4 Hybrid rank={plan.rank}/{plan.world_size} "
            f"arch={kb.arch_family.value} sparse_mla={kb.sparse_mla_backend.value} "
            f"moe_gemm={kb.moe_gemm_backend.value}"
        )
        register_verified(model_name)
        print(f"[sglang-lite] MoE model '{model_name}' Hybrid ready on {plan.local_device}")

    def _init_tiny_stub_model(self):
        import torch.nn as nn

        class TinyLM(nn.Module):
            def __init__(self, vocab=32000, dim=256, layers=4, heads=4):
                super().__init__()
                self.embed = nn.Embedding(vocab, dim)
                self.layers = nn.ModuleList(
                    [
                        nn.TransformerEncoderLayer(dim, heads, dim * 4, batch_first=True)
                        for _ in range(layers)
                    ]
                )
                self.ln = nn.LayerNorm(dim)
                self.head = nn.Linear(dim, vocab, bias=False)
                self.vocab_size = vocab
                self.config = type(
                    "C",
                    (),
                    {
                        "vocab_size": vocab,
                        "num_hidden_layers": layers,
                        "num_attention_heads": heads,
                        "num_key_value_heads": heads,
                        "hidden_size": dim,
                        "eos_token_id": 2,
                    },
                )()

            def forward(
                self, input_ids, past_key_values=None, use_cache=False, attention_mask=None, **kwargs
            ):
                x = self.embed(input_ids)
                for layer in self.layers:
                    x = layer(x)
                x = self.ln(x)
                logits = self.head(x)
                dummy_past = None
                if use_cache:
                    dummy_past = [
                        (x[:, -1:, :].clone(), x[:, -1:, :].clone())
                        for _ in range(len(self.layers))
                    ]
                return type(
                    "Obj",
                    (),
                    {"logits": logits, "past_key_values": dummy_past},
                )()

        self.model = TinyLM(vocab=self.vocab_size).to(self.device)
        self.model.eval()
        self._is_real = False
        self.num_layers = 4
        self.num_kv_heads = 4
        self.head_dim = 64
        self.eos_token_id = 2
        self.use_paged_as_source = False  # stub past shapes are not real KV
        print("[sglang-lite] Using explicit tiny stub model (allow_stub=True).")

    def tokenize(self, text: str) -> List[int]:
        if self.tokenizer is not None:
            return self.tokenizer.encode(text, add_special_tokens=False)
        return [hash(c) % self.vocab_size for c in text[:128]]

    def apply_chat_template(self, messages: list) -> List[int]:
        if self._v4_hybrid:
            ids = self._v4_encode_messages(messages)
            if ids is not None:
                return ids
        if self.tokenizer is not None and hasattr(self.tokenizer, "apply_chat_template"):
            try:
                ids = self.tokenizer.apply_chat_template(
                    messages, tokenize=True, add_generation_prompt=True
                )
                return list(ids)
            except Exception:
                pass
        parts = []
        for m in messages:
            if isinstance(m, dict):
                role = m.get("role", "user")
                content = m.get("content") or ""
            else:
                role = getattr(m, "role", "user")
                content = getattr(m, "content", "") or ""
            parts.append(f"{role}: {content}")
        parts.append("assistant:")
        return self.tokenize("\n".join(parts))

    def _v4_encode_messages(self, messages: list) -> Optional[List[int]]:
        """Prefer vendored encoding_dsv4; fall back to HF tree encoding/."""
        import sys

        from .v4_runner import encode_chat_messages

        rendered = encode_chat_messages(messages, thinking_mode="chat")
        if rendered is None:
            try:
                from .model_loader import resolve_v4_paths

                paths = resolve_v4_paths()
            except Exception:
                return None
            encoding_dir = paths.hf_ckpt / "encoding"
            infer_dir = paths.hf_ckpt / "inference"
            if not encoding_dir.is_dir():
                return None
            for p in (str(encoding_dir), str(infer_dir)):
                if p not in sys.path:
                    sys.path.insert(0, p)
            try:
                from encoding_dsv4 import encode_messages

                norm = []
                for m in messages:
                    if isinstance(m, dict):
                        norm.append(
                            {
                                "role": m.get("role", "user"),
                                "content": m.get("content") or "",
                            }
                        )
                    else:
                        norm.append(
                            {
                                "role": getattr(m, "role", "user"),
                                "content": getattr(m, "content", "") or "",
                            }
                        )
                rendered = encode_messages(norm, thinking_mode="chat")
            except Exception:
                return None
        if rendered is None or self.tokenizer is None:
            return None
        try:
            return list(self.tokenizer.encode(rendered))
        except Exception:
            return None

    def detokenize(self, token_ids: List[int]) -> str:
        if self.tokenizer is not None:
            return self.tokenizer.decode(token_ids, skip_special_tokens=True)
        return "".join([chr(97 + (t % 26)) for t in token_ids[:20]])

    def detokenize_delta(self, token_ids: List[int], prev_text: str = "") -> str:
        """Incremental decode; never re-emit the whole string on tokenizer churn.

        Phase-A thruput: most steps use a **windowed** decode (last ≤16 tokens)
        instead of full-sequence decode every token (was O(n²) tokenizer cost).
        Full resync every 16 tokens and on empty/irregular deltas.
        """
        if not token_ids:
            return ""
        n = len(token_ids)
        # Windowed fast path: decode a short suffix and emit only the new piece.
        # Full resync periodically for stop-string / multi-byte correctness.
        use_window = n > 1 and (n % 16) != 0
        if use_window and self.tokenizer is not None:
            window = 12
            suffix_ids = token_ids[-window:]
            # Decode suffix; best-effort delta = last piece after re-decode.
            suffix = self.tokenizer.decode(suffix_ids, skip_special_tokens=True)
            # Heuristic: if prev ends with a prefix of suffix, emit the rest.
            # Fallback to full path when mismatch.
            for k in range(min(len(prev_text), len(suffix)), -1, -1):
                if k == 0:
                    break
                if prev_text.endswith(suffix[:k]):
                    delta = suffix[k:]
                    if delta == "\ufffd":
                        return ""
                    return delta
            # If prev is empty, emit full suffix once.
            if not prev_text:
                return suffix if suffix != "\ufffd" else ""

        full = self.detokenize(token_ids)
        if full.startswith(prev_text):
            delta = full[len(prev_text) :]
        elif prev_text.startswith(full):
            # Incomplete UTF-8 / multi-token glyph — hold until decode grows.
            return ""
        else:
            n_common = 0
            limit = min(len(prev_text), len(full))
            while n_common < limit and prev_text[n_common] == full[n_common]:
                n_common += 1
            if n_common == 0 and prev_text:
                return ""
            delta = full[n_common:]
        # Lone U+FFFD is an incomplete decode artifact — do not stream it.
        if delta == "\ufffd":
            return ""
        return delta

    @torch.inference_mode()
    def run_decode_burst(
        self, seq: Sequence, radix: RadixCache, n_steps: int
    ) -> List[int]:
        """Decode up to ``n_steps`` tokens for one sequence without scheduler hops.

        Appends to ``seq.output_ids`` and updates ``cached_len`` via ``_decode_one``.
        Caller must still run stop/finish logic.

        Must run under inference_mode: without it, DynamicCache updates build an
        autograd graph each step (~3× slower on Qwen3-30B-A3B decode).
        """
        n_steps = max(0, int(n_steps))
        if n_steps == 0:
            return []
        temp = float(getattr(seq, "temperature", 0.0) or 0.0)
        # V4 Hybrid: tight official-forward burst (skip dual stage/append tax).
        if self._v4_hybrid and temp <= 1e-5:
            return self._decode_burst_v4_greedy(seq, radix, n_steps)
        # Tight HF greedy path (no radix pages, no attention_mask) — matches
        # model.generate kwargs and keeps compile/inductor shapes stable.
        if (
            not self._v4_hybrid
            and not self.use_paged_as_source
            and not self._use_paged_attention()
            and temp <= 1e-5
            and seq.kv_state is not None
        ):
            return self._decode_burst_hf_greedy(seq, n_steps)
        # Tight paged greedy: fewer host branches than per-step _decode_one.
        if (
            not self._v4_hybrid
            and self._use_paged_attention()
            and temp <= 1e-5
        ):
            return self._decode_burst_paged_greedy(seq, radix, n_steps)

        out: List[int] = []
        eos = self.eos_token_id
        for _ in range(n_steps):
            if seq.finished:
                break
            tok = self._decode_one(seq, radix)
            out.append(int(tok))
            seq.output_ids.append(int(tok))
            seq.decode_tokens = getattr(seq, "decode_tokens", 0) + 1
            if eos is not None and int(tok) == int(eos):
                break
        return out

    def _decode_burst_v4_greedy(
        self, seq: Sequence, radix: RadixCache, n_steps: int
    ) -> List[int]:
        """Greedy V4 Hybrid decode burst — minimize dual-pool / host tax.

        Live KV stays in official Attention module buffers. Dual-pool stage is
        only done if ``_v4_need_stage`` (after prefix restore). Dual append is
        optional (``SGLANG_LITE_V4_DUAL_APPEND``, default on) but deferred to
        end of burst when off, or skipped when env disables.
        """
        out: List[int] = []
        eos = self.eos_token_id
        ignore_eos = bool(getattr(seq, "ignore_eos", False))
        # Prefill may have left last_logits for the first sample.
        if not seq.output_ids and getattr(seq, "last_logits", None) is not None:
            self._v4_ensure_restored(seq, batch_slot=0)
            seq.cached_len = len(seq.input_ids)
            tok = int(torch.argmax(seq.last_logits.to(device=self.device)).item())
            out.append(tok)
            seq.output_ids.append(tok)
            seq.decode_tokens = getattr(seq, "decode_tokens", 0) + 1
            n_steps -= 1
            if n_steps <= 0 or (
                not ignore_eos and eos is not None and tok == int(eos)
            ):
                return out

        # One restore + optional stage for the whole burst.
        self._v4_ensure_restored(seq, batch_slot=0)
        self._v4_stage_pages_before_forward(seq, batch_slot=0)

        tok_buf = self._decode_input_buf
        # GPU token ring; single D2H at end when ignore_eos (no mid-burst sync).
        if (
            self._paged_burst_toks is None
            or self._paged_burst_toks.device.type != torch.device(self.device).type
            or self._paged_burst_toks.numel() < n_steps
        ):
            self._paged_burst_toks = torch.zeros(
                max(n_steps, 128), dtype=torch.long, device=self.device
            )
        assert self._paged_burst_toks is not None
        last_t = torch.tensor(
            [seq.output_ids[-1] if seq.output_ids else seq.input_ids[-1]],
            dtype=torch.long,
            device=self.device,
        )
        # Fast path: ignore_eos thruput — zero host token reads until burst end.
        if ignore_eos or eos is None:
            for i in range(max(0, n_steps)):
                pos = seq.cached_len
                input_ids = tok_buf.set_token_tensor(last_t)
                logits = self._model_forward_v4(input_ids, start_pos=pos)
                nxt = torch.argmax(logits[0])
                self._paged_burst_toks[i] = nxt
                last_t = nxt
                seq.kv_state = None
                seq.cached_len = pos + 1
                seq.decode_tokens = getattr(seq, "decode_tokens", 0) + 1
            host = self._paged_burst_toks[:n_steps].tolist()
            seq.output_ids.extend(int(x) for x in host)
            return [int(x) for x in host]

        for i in range(max(0, n_steps)):
            pos = seq.cached_len
            input_ids = tok_buf.set_token_tensor(last_t)
            logits = self._model_forward_v4(input_ids, start_pos=pos)
            nxt = torch.argmax(logits[0])
            self._paged_burst_toks[i] = nxt
            tok = int(nxt.item())
            last_t = nxt
            seq.kv_state = None
            seq.cached_len = pos + 1
            out.append(tok)
            seq.output_ids.append(tok)
            seq.decode_tokens = getattr(seq, "decode_tokens", 0) + 1
            if tok == int(eos):
                break
        return out

    def _decode_burst_hf_greedy(self, seq: Sequence, n_steps: int) -> List[int]:
        """Greedy DynamicCache decode with minimal host/Python tax."""
        out: List[int] = []
        eos = self.eos_token_id
        ignore_eos = bool(getattr(seq, "ignore_eos", False))
        # First token may already sit in last_logits (prefill sampled later).
        if not seq.output_ids and getattr(seq, "last_logits", None) is not None:
            seq.cached_len = len(seq.input_ids)
            tok = int(torch.argmax(seq.last_logits.to(device=self.device)).item())
            out.append(tok)
            seq.output_ids.append(tok)
            seq.decode_tokens = getattr(seq, "decode_tokens", 0) + 1
            n_steps -= 1
            if n_steps <= 0 or (
                not ignore_eos and eos is not None and tok == int(eos)
            ):
                return out

        tok_buf = self._decode_input_buf
        past = seq.kv_state
        use_l2k = self._supports_logits_to_keep()
        for _ in range(max(0, n_steps)):
            last = seq.output_ids[-1] if seq.output_ids else seq.input_ids[-1]
            pos = seq.cached_len
            input_ids = tok_buf.set_token(int(last))
            position_ids = tok_buf.set_position(int(pos))
            kwargs = {
                "input_ids": input_ids,
                "past_key_values": past,
                "use_cache": True,
                "position_ids": position_ids,
                "cache_position": position_ids.reshape(-1),
            }
            if use_l2k:
                kwargs["logits_to_keep"] = 1
            self.model_forward_count += 1
            outputs = self.model(**kwargs)
            past = outputs.past_key_values
            seq.kv_state = past
            seq.cached_len = pos + 1
            tok = int(outputs.logits[0, -1].argmax().item())
            out.append(tok)
            seq.output_ids.append(tok)
            seq.decode_tokens = getattr(seq, "decode_tokens", 0) + 1
            if not ignore_eos and eos is not None and tok == int(eos):
                break
        return out

    def _decode_burst_paged_greedy(
        self, seq: Sequence, radix: RadixCache, n_steps: int
    ) -> List[int]:
        """Greedy paged decode burst — minimize host sync for thruput.

        Hot path (``ignore_eos`` thruput probes):
          * keep next-token on GPU (no per-step ``.item()``)
          * feed next input via device write into static token buffer
          * allocate / COW pages only on page boundaries
          * single host sync of the token ring at the end
        """
        out: List[int] = []
        eos = self.eos_token_id
        ignore_eos = bool(getattr(seq, "ignore_eos", False))
        if not seq.output_ids and getattr(seq, "last_logits", None) is not None:
            seq.cached_len = len(seq.input_ids)
            tok = int(torch.argmax(seq.last_logits.to(device=self.device)).item())
            out.append(tok)
            seq.output_ids.append(tok)
            seq.decode_tokens = getattr(seq, "decode_tokens", 0) + 1
            n_steps -= 1
            if n_steps <= 0 or (
                not ignore_eos and eos is not None and tok == int(eos)
            ):
                return out

        n_steps = max(0, int(n_steps))
        if n_steps == 0:
            return out

        tok_buf = self._decode_input_buf
        # Seed input buffer once from host (last emitted / prompt tail).
        last = seq.output_ids[-1] if seq.output_ids else seq.input_ids[-1]
        tok_buf.set_token(int(last))

        # GPU ring for completion tokens (one D2H at end when ignore_eos).
        if (
            self._paged_burst_toks is None
            or self._paged_burst_toks.device.type != torch.device(self.device).type
            or self._paged_burst_toks.numel() < n_steps
        ):
            self._paged_burst_toks = torch.zeros(
                max(n_steps, 128), dtype=torch.long, device=self.device
            )
        assert self._paged_burst_toks is not None
        page_size = int(radix.block_size)
        produced = 0

        for i in range(n_steps):
            pos = seq.cached_len
            # Page alloc + COW only when entering a new page slot.
            if (pos % page_size) == 0 or not seq.block_table:
                self._ensure_blocks(seq, radix, pos + 1)
                page_i = pos // page_size
                if page_i < len(seq.block_table):
                    seq.block_table[page_i] = radix.cow_block_if_shared(
                        seq.block_table[page_i]
                    )
            # Static [1,1] buffers — no per-step host tensor alloc.
            input_ids = tok_buf.token_buf
            tok_buf.set_position(pos)
            outputs = self._model_forward_paged(
                input_ids, [seq], radix, is_decode=True
            )
            seq.kv_state = None
            seq.cached_len = pos + 1
            # On-device greedy. Native body may already have written next token into
            # the static input buffer; re-argmax is cheap and keeps paths consistent.
            nxt = outputs.logits[0, -1].argmax()
            self._paged_burst_toks[i] = nxt
            tok_buf.set_token_tensor(nxt)
            produced += 1

            if not ignore_eos:
                # Need host value for EOS / streaming clients.
                tok = int(nxt.item())
                out.append(tok)
                seq.output_ids.append(tok)
                seq.decode_tokens = getattr(seq, "decode_tokens", 0) + 1
                if eos is not None and tok == int(eos):
                    return out

        if ignore_eos and produced > 0:
            # One D2H for the whole burst (probe thruput path).
            host = self._paged_burst_toks[:produced].tolist()
            out.extend(host)
            seq.output_ids.extend(host)
            seq.decode_tokens = getattr(seq, "decode_tokens", 0) + produced
        return out

    @torch.inference_mode()
    def run_batch(
        self,
        batch: List[Sequence],
        radix: RadixCache,
        is_prefill: List[bool],
    ) -> List[Optional[int]]:
        """Execute one scheduler step.

        Prefill/decode groups that share the same cached length are executed with a
        single batched model forward (tensor batch dim > 1 when possible).
        """
        if not batch:
            return []

        results: List[Optional[int]] = [None] * len(batch)
        self.last_model_forward_size = 0

        prefill_idxs: List[int] = []
        decode_idxs: List[int] = []

        for i, seq in enumerate(batch):
            if seq.finished:
                continue
            # Exact prefix hit: first token from stored logits (no forward)
            if not seq.output_ids and getattr(seq, "last_logits", None) is not None:
                if seq.cached_len >= len(seq.input_ids) or (
                    not is_prefill[i] and seq.cached_len == len(seq.input_ids)
                ):
                    if self._v4_hybrid:
                        self._v4_ensure_restored(seq, batch_slot=0)
                    results[i] = self._sample(seq.last_logits.to(device=self.device), seq)
                    seq.cached_len = len(seq.input_ids)
                    continue
            if is_prefill[i] and seq.cached_len < len(seq.input_ids):
                prefill_idxs.append(i)
            else:
                decode_idxs.append(i)

        if prefill_idxs:
            self._run_prefill_groups(batch, prefill_idxs, radix, results)
        if decode_idxs:
            self._run_decode_groups(batch, decode_idxs, radix, results)
        return results

    def _run_prefill_groups(
        self,
        batch: List[Sequence],
        idxs: List[int],
        radix: RadixCache,
        results: List[Optional[int]],
    ) -> None:
        groups: Dict[int, List[int]] = defaultdict(list)
        for i in idxs:
            groups[batch[i].cached_len].append(i)
        for _cached_len, group in groups.items():
            self._batch_prefill([batch[i] for i in group], group, radix, results)

    def _run_decode_groups(
        self,
        batch: List[Sequence],
        idxs: List[int],
        radix: RadixCache,
        results: List[Optional[int]],
    ) -> None:
        groups: Dict[int, List[int]] = defaultdict(list)
        for i in idxs:
            groups[batch[i].cached_len].append(i)
        for _cached_len, group in groups.items():
            self._batch_decode([batch[i] for i in group], group, radix, results)

    def _batch_prefill(
        self,
        seqs: List[Sequence],
        idxs: List[int],
        radix: RadixCache,
        results: List[Optional[int]],
    ) -> None:
        if not seqs:
            return
        news = [s.input_ids[s.cached_len :] for s in seqs]
        past_lens = [s.cached_len for s in seqs]
        new_lens = [len(n) for n in news]
        # Identical past + new lengths required for a correct single forward (no pad KV pollution)
        if len(set(past_lens)) > 1 or len(set(new_lens)) > 1:
            for s, i in zip(seqs, idxs):
                results[i] = self._prefill_one(s, radix)
            return

        B = len(seqs)
        max_new = new_lens[0]
        if max_new == 0:
            for b, (s, i) in enumerate(zip(seqs, idxs)):
                if s.last_logits is None:
                    raise RuntimeError("exact hit without last_logits")
                if self._v4_hybrid:
                    self._v4_ensure_restored(s, batch_slot=b)
                results[i] = self._sample(s.last_logits.to(self.device), s)
                s.cached_len = len(s.input_ids)
            return

        input_ids = torch.tensor(news, dtype=torch.long, device=self.device)
        max_past = past_lens[0]

        if self._v4_hybrid:
            # Official decode path assumes seqlen==1 when start_pos>0.
            if max_past > 0 and max_new > 1:
                for s, i in zip(seqs, idxs):
                    results[i] = self._prefill_one(s, radix)
                return
            for b, seq in enumerate(seqs):
                self._v4_ensure_restored(seq, batch_slot=b)
                if max_past > 0:
                    self._v4_stage_pages_before_forward(seq, batch_slot=b)
            logits = self._model_forward_v4(input_ids, start_pos=max_past)
            for b, (seq, i) in enumerate(zip(seqs, idxs)):
                nlen = new_lens[b]
                row = logits[b]
                seq.last_logits = row.detach().float().cpu().clone()
                seq.kv_state = None
                seq.cached_len = seq.cached_len + nlen
                seq.prefill_tokens = getattr(seq, "prefill_tokens", 0) + nlen
                results[i] = self._sample(row, seq)
                self._v4_maybe_save_prefix(seq, batch_slot=b, radix=radix)
            return

        if self.use_paged_as_source:
            for s in seqs:
                self._ensure_blocks(s, radix, s.cached_len + max_new)

        if self._use_paged_attention():
            outputs = self._model_forward_paged(input_ids, seqs, radix, is_decode=False)
            for b, (seq, i) in enumerate(zip(seqs, idxs)):
                nlen = new_lens[b]
                # logits_to_keep=1 → time dim is 1; always take last.
                logits = outputs.logits[b, -1, :]
                start = seq.cached_len
                seq.last_logits = logits.detach().float().cpu().clone()
                seq.kv_state = None
                seq.cached_len = start + nlen
                seq.prefill_tokens = getattr(seq, "prefill_tokens", 0) + nlen
                results[i] = self._sample(logits, seq)
            return

        pasts = [self._past_for_seq(s, radix) for s in seqs]
        batched_past = self._batch_caches(pasts) if past_lens[0] > 0 else None
        attn = torch.ones((B, max_past + max_new), dtype=torch.long, device=self.device)
        # Explicit position_ids for multi-token prefill (HF may omit when past=None).
        position_ids = torch.arange(
            max_past, max_past + max_new, device=self.device, dtype=torch.long
        ).unsqueeze(0).expand(B, -1)
        outputs = self._model_forward(
            input_ids, batched_past, attn, position_ids=position_ids
        )
        for b, (seq, i) in enumerate(zip(seqs, idxs)):
            nlen = new_lens[b]
            # logits_to_keep=1 → time dim is 1; always take last.
            logits = outputs.logits[b, -1, :]
            full_kv = self._split_batch_cache(outputs.past_key_values, b, B)
            start = seq.cached_len
            if self.use_paged_as_source:
                write_kv = self._slice_kv_tail(self._to_legacy_kv(full_kv), nlen)
                self._commit_pages(seq, radix, start, write_kv)
                seq.kv_state = None
            else:
                seq.kv_state = full_kv
            seq.last_logits = logits.detach().float().cpu().clone()
            seq.cached_len = start + nlen
            seq.prefill_tokens = getattr(seq, "prefill_tokens", 0) + nlen
            results[i] = self._sample(logits, seq)

    def _batch_decode(
        self,
        seqs: List[Sequence],
        idxs: List[int],
        radix: RadixCache,
        results: List[Optional[int]],
    ) -> None:
        if not seqs:
            return
        # All seqs in this group share cached_len
        pending_seqs: List[Sequence] = []
        pending_idxs: List[int] = []
        for b, (s, i) in enumerate(zip(seqs, idxs)):
            if not s.output_ids and getattr(s, "last_logits", None) is not None:
                if self._v4_hybrid:
                    self._v4_ensure_restored(s, batch_slot=b)
                results[i] = self._sample(s.last_logits.to(self.device), s)
                s.cached_len = len(s.input_ids)
                continue
            pending_seqs.append(s)
            pending_idxs.append(i)
        if not pending_seqs:
            return
        seqs, idxs = pending_seqs, pending_idxs
        B = len(seqs)
        lasts = [s.output_ids[-1] if s.output_ids else s.input_ids[-1] for s in seqs]
        input_ids = torch.tensor([[t] for t in lasts], dtype=torch.long, device=self.device)

        if self._v4_hybrid:
            for b, seq in enumerate(seqs):
                self._v4_ensure_restored(seq, batch_slot=b)
                # 0c-4: page-primary → stage official kv_cache from pages first.
                self._v4_stage_pages_before_forward(seq, batch_slot=b)
            start_pos = seqs[0].cached_len
            logits = self._model_forward_v4(input_ids, start_pos=start_pos)
            for b, (seq, i) in enumerate(zip(seqs, idxs)):
                row = logits[b]
                seq.kv_state = None
                seq.cached_len = seq.cached_len + 1
                results[i] = self._sample(row, seq)
                # Phase 0c: keep dual-pool pages in sync on decode (best-effort).
                self._v4_dual_append_decode(seq, batch_slot=b, radix=radix)
            return

        # COW last page before append (standard MHA page path only)
        if self.use_paged_as_source:
            for s in seqs:
                pos = s.cached_len
                self._ensure_blocks(s, radix, pos + 1)
                page_i = pos // radix.block_size
                if page_i < len(s.block_table):
                    s.block_table[page_i] = radix.cow_block_if_shared(
                        s.block_table[page_i]
                    )

        if self._use_paged_attention():
            outputs = self._model_forward_paged(input_ids, seqs, radix, is_decode=True)
            for b, (seq, i) in enumerate(zip(seqs, idxs)):
                logits = outputs.logits[b, -1, :]
                seq.kv_state = None
                seq.cached_len = seq.cached_len + 1
                results[i] = self._sample(logits, seq)
            return

        pasts = [self._past_for_seq(s, radix) for s in seqs]
        batched_past = self._batch_caches(pasts)
        past_len = seqs[0].cached_len
        attn = torch.ones((B, past_len + 1), dtype=torch.long, device=self.device)
        outputs = self._model_forward(input_ids, batched_past, attn)

        for b, (seq, i) in enumerate(zip(seqs, idxs)):
            logits = outputs.logits[b, -1, :]
            full_kv = self._split_batch_cache(outputs.past_key_values, b, B)
            pos = seq.cached_len
            if self.use_paged_as_source:
                write_kv = self._slice_kv_tail(self._to_legacy_kv(full_kv), 1)
                self._commit_pages(seq, radix, pos, write_kv)
                seq.kv_state = None
            else:
                seq.kv_state = full_kv
            seq.cached_len = pos + 1
            results[i] = self._sample(logits, seq)

    @torch.inference_mode()
    def _prefill_one(self, seq: Sequence, radix: RadixCache) -> int:
        """Serial prefill fallback (different past lengths in a group)."""
        prompt = seq.input_ids
        start = seq.cached_len
        new_tokens = prompt[start:]
        if not new_tokens:
            if seq.last_logits is None:
                raise RuntimeError("exact hit without last_logits")
            if self._v4_hybrid:
                self._v4_ensure_restored(seq, batch_slot=0)
            return self._sample(seq.last_logits.to(self.device), seq)
        input_ids = torch.tensor([new_tokens], dtype=torch.long, device=self.device)
        if self._v4_hybrid:
            self._v4_ensure_restored(seq, batch_slot=0)
            if start > 0 and len(new_tokens) > 1:
                row = None
                for t_i, tok in enumerate(new_tokens):
                    if t_i == 0 or getattr(seq, "_v4_page_primary", False):
                        self._v4_stage_pages_before_forward(seq, batch_slot=0)
                    step_ids = torch.tensor(
                        [[tok]], dtype=torch.long, device=self.device
                    )
                    logits = self._model_forward_v4(step_ids, start_pos=start + t_i)
                    row = logits[0]
                assert row is not None
            else:
                if start > 0:
                    self._v4_stage_pages_before_forward(seq, batch_slot=0)
                logits = self._model_forward_v4(input_ids, start_pos=start)
                row = logits[0]
            seq.last_logits = row.detach().float().cpu().clone()
            seq.kv_state = None
            seq.cached_len = len(prompt)
            seq.prefill_tokens = getattr(seq, "prefill_tokens", 0) + len(new_tokens)
            tok = self._sample(row, seq)
            self._v4_maybe_save_prefix(seq, batch_slot=0, radix=radix)
            return tok
        if self.use_paged_as_source:
            self._ensure_blocks(seq, radix, len(prompt))
        if self._use_paged_attention():
            outputs = self._model_forward_paged(input_ids, [seq], radix, is_decode=False)
            logits = outputs.logits[0, -1, :]
            # Keep on device for immediate sample; radix insert will CPU-clone if needed.
            seq.last_logits = logits.detach()
            seq.kv_state = None
            seq.cached_len = len(prompt)
            seq.prefill_tokens = getattr(seq, "prefill_tokens", 0) + len(new_tokens)
            return self._sample(logits, seq)
        # HF thruput path: prefill into StaticCache so decode can CUDA-graph.
        if self._hf_static_cg is not None and not self.use_paged_as_source:
            cache = self._hf_static_cg.alloc_cache(self.model)
            n_new = len(new_tokens)
            position_ids = torch.arange(
                start, start + n_new, device=self.device, dtype=torch.long
            ).unsqueeze(0)
            kwargs = {
                "input_ids": input_ids,
                "past_key_values": cache,
                "use_cache": True,
                "position_ids": position_ids,
                "cache_position": position_ids.reshape(-1),
            }
            if self._supports_logits_to_keep():
                kwargs["logits_to_keep"] = 1
            self.model_forward_count += 1
            outputs = self.model(**kwargs)
            logits = outputs.logits[0, -1, :]
            seq.kv_state = outputs.past_key_values
            seq.last_logits = logits.detach()
            seq.cached_len = len(prompt)
            seq.prefill_tokens = getattr(seq, "prefill_tokens", 0) + len(new_tokens)
            return self._sample(logits, seq)

        past = self._past_for_seq(seq, radix)
        attn = self._attention_mask(input_ids, past, past_len_hint=start)
        n_new = len(new_tokens)
        position_ids = torch.arange(
            start, start + n_new, device=self.device, dtype=torch.long
        ).unsqueeze(0)
        outputs = self._model_forward(
            input_ids, past, attn, position_ids=position_ids
        )
        logits = outputs.logits[0, -1, :]
        new_kv = outputs.past_key_values
        if self.use_paged_as_source:
            write_kv = self._slice_kv_tail(self._to_legacy_kv(new_kv), len(new_tokens))
            self._commit_pages(seq, radix, start, write_kv)
            seq.kv_state = None
        else:
            # MLA / non-standard KV: keep HF DynamicCache only (no MHA pages).
            seq.kv_state = new_kv
        seq.last_logits = logits.detach()
        seq.cached_len = len(prompt)
        seq.prefill_tokens = getattr(seq, "prefill_tokens", 0) + len(new_tokens)
        return self._sample(logits, seq)

    @torch.inference_mode()
    def _decode_one(self, seq: Sequence, radix: RadixCache) -> int:
        if not seq.output_ids and getattr(seq, "last_logits", None) is not None:
            if self._v4_hybrid:
                self._v4_ensure_restored(seq, batch_slot=0)
            seq.cached_len = len(seq.input_ids)
            return self._sample(seq.last_logits.to(device=self.device), seq)
        last_token = seq.output_ids[-1] if seq.output_ids else seq.input_ids[-1]
        # Reuse host-side alloc for single-token decode (Phase-A thruput).
        input_ids = self._decode_input_buf.set_token(int(last_token))
        pos = seq.cached_len
        if self._v4_hybrid:
            self._v4_ensure_restored(seq, batch_slot=0)
            self._v4_stage_pages_before_forward(seq, batch_slot=0)
            logits = self._model_forward_v4(input_ids, start_pos=pos)
            row = logits[0]
            seq.kv_state = None
            seq.cached_len = pos + 1
            self._v4_dual_append_decode(seq, batch_slot=0, radix=radix)
            return self._sample(row, seq)
        if self.use_paged_as_source:
            self._ensure_blocks(seq, radix, pos + 1)
            if seq.block_table:
                page_i = pos // radix.block_size
                if page_i < len(seq.block_table):
                    seq.block_table[page_i] = radix.cow_block_if_shared(
                        seq.block_table[page_i]
                    )
        if self._use_paged_attention():
            outputs = self._model_forward_paged(input_ids, [seq], radix, is_decode=True)
            logits = outputs.logits[0, -1, :]
            seq.kv_state = None
            seq.cached_len = pos + 1
            return self._sample(logits, seq)
        # HF StaticCache CUDA-graph decode (batched_mm thruput path).
        if (
            self._hf_static_cg is not None
            and not self.use_paged_as_source
            and seq.kv_state is not None
        ):
            if pos >= self._hf_static_cg.max_cache_len - 1:
                # Fall back to eager DynamicCache-style when static capacity exceeded.
                pass
            else:
                outputs = self._hf_static_cg.run_decode_step(
                    token_id=int(last_token),
                    pos=int(pos),
                    past=seq.kv_state,
                )
                self.model_forward_count += 1
                logits = outputs.logits[0, -1, :]
                seq.kv_state = outputs.past_key_values
                seq.cached_len = pos + 1
                return self._sample(logits, seq)
        past = self._past_for_seq(seq, radix)
        attn = self._attention_mask(input_ids, past, past_len_hint=pos)
        position_ids = self._decode_input_buf.set_position(pos)
        outputs = self._model_forward(
            input_ids, past, attn, position_ids=position_ids
        )
        logits = outputs.logits[0, -1, :]
        new_kv = outputs.past_key_values
        if self.use_paged_as_source:
            write_kv = self._slice_kv_tail(self._to_legacy_kv(new_kv), 1)
            self._commit_pages(seq, radix, pos, write_kv)
            seq.kv_state = None
        else:
            seq.kv_state = new_kv
        seq.cached_len = pos + 1
        return self._sample(logits, seq)

    def _use_paged_attention(self) -> bool:
        if self._v4_hybrid:
            return False
        # MLA / non-standard KV keeps use_paged_as_source=False and HF cache.
        if not self.use_paged_as_source:
            return False
        return bool(
            self._is_real
            and getattr(self.kernel_backend, "supports_paged_attention", False)
        )

    def _model_forward_v4(self, input_ids: torch.Tensor, start_pos: int) -> torch.Tensor:
        """Official/vendored Transformer → logits [B, vocab] via v4_runner."""
        self.model_forward_count += 1
        self.last_model_forward_size = int(input_ids.shape[0])
        accel = getattr(self, "_v4_accel", None)
        if accel is not None:
            return accel.decode_logits(input_ids, start_pos)
        from .v4_runner import model_forward_logits

        return model_forward_logits(self.model, input_ids, start_pos)

    def v4_match_prefix(self, token_ids: List[int]):
        """Match ``token_ids`` against the Hybrid V4 prefix snapshot store."""
        cache = self._v4_prefix_cache
        if cache is None:
            return 0, None
        return cache.match(token_ids)

    def _v4_ensure_restored(self, seq: Sequence, *, batch_slot: int = 0) -> None:
        """Restore prefix state, or clear the slot on a cold prefill.

        Phase 0c-3: prefer dual-pool bf16 page restore for ``kv_cache`` rows,
        then apply CPU snapshot for remaining buffers (``kv_state`` /
        ``score_state`` / fallback).
        """
        if getattr(seq, "_v4_kv_pending_restore", False):
            entry = getattr(seq, "_v4_prefix_entry", None)
            seq._v4_kv_pending_restore = False
            if entry is None or self.model is None:
                return
            from .v4_prefix_cache import restore_v4_kv

            skip_keys = set()
            handle = getattr(seq, "_v4_dual_handle", None)
            radix = getattr(seq, "_v4_dual_radix", None)
            if (
                handle is not None
                and radix is not None
                and getattr(radix, "restore_bf16_cache", None) is not None
                and handle.swa_blocks
            ):
                try:
                    from .v4_dual_pool import restore_dual_pool_to_model

                    n_dual, skip_keys = restore_dual_pool_to_model(
                        self.model,
                        radix,
                        handle,
                        batch_slot=batch_slot,
                        n_tokens=int(
                            getattr(entry, "dual_pool_tokens", 0)
                            or handle.n_tokens
                            or seq.cached_len
                        ),
                    )
                    if n_dual > 0:
                        seq._v4_dual_restored = True
                        # 0c-4: after page restore, pages own KV for subsequent decode.
                        if getattr(entry, "dual_primary", False) or n_dual > 0:
                            seq._v4_page_primary = True
                            # Stage once before the next forward; do NOT re-stage
                            # every decode step (that was destroying thruput).
                            seq._v4_need_stage = True
                except Exception as e:
                    print(f"[sglang-lite] v4 dual-pool restore skipped: {e}")
                    skip_keys = set()
            restore_v4_kv(
                self.model,
                entry.buffers,
                batch_slot=batch_slot,
                skip_keys=skip_keys,
            )
            seq._v4_slot_prepared = True
            return
        # Cold path: wipe stale compressor / indexer state before start_pos=0.
        if (
            seq.cached_len == 0
            and not getattr(seq, "_v4_slot_prepared", False)
            and self.model is not None
        ):
            from .v4_prefix_cache import clear_v4_kv_slot

            clear_v4_kv_slot(self.model, batch_slot=batch_slot)
            seq._v4_slot_prepared = True

    def v4_release_seq(self, seq: Sequence, *, batch_slot: int = 0) -> None:
        """Drop in-GPU slot ownership after a sequence finishes or cancels."""
        if not self._v4_hybrid or self.model is None:
            return
        from .v4_prefix_cache import clear_v4_kv_slot

        clear_v4_kv_slot(self.model, batch_slot=batch_slot)
        # Phase 0c: release only this sequence's dual-pool fork (cache keeps its ref).
        self._v4_release_dual_pool(seq)
        seq._v4_prefix_entry = None
        seq._v4_kv_pending_restore = False
        seq._v4_slot_prepared = False
        seq._v4_page_primary = False
        seq._v4_dual_restored = False

    def _v4_release_dual_pool(self, seq: Sequence) -> None:
        """Free dual-pool pages forked for this sequence (not the prefix-cache copy)."""
        handle = getattr(seq, "_v4_dual_handle", None)
        if handle is not None:
            try:
                from .v4_dual_pool import release_dual_pool_pages

                radix = getattr(seq, "_v4_dual_radix", None)
                if radix is not None:
                    release_dual_pool_pages(radix, handle)
            except Exception:
                pass
            seq._v4_dual_handle = None
            seq._v4_dual_radix = None
        seq.swa_block_table = []
        seq.comp_block_table = []
        seq._v4_page_primary = False

    def _v4_stage_pages_before_forward(
        self, seq: Sequence, *, batch_slot: int = 0
    ) -> bool:
        """Stage official ``kv_cache`` from dual-pool pages when required.

        Only after a **prefix-hit dual restore** (``_v4_need_stage``). Continuous
        decode already has live module buffers; re-staging every token was
        measured as thruput poison on PRO6000 (~official 7.5 → host tax).
        """
        if self.model is None:
            return False
        # One-shot after restore; also allow explicit force via page_primary+need_stage.
        if not getattr(seq, "_v4_need_stage", False):
            return False
        handle = getattr(seq, "_v4_dual_handle", None)
        radix = getattr(seq, "_v4_dual_radix", None)
        if handle is None or radix is None or not handle.swa_blocks:
            return False
        if getattr(radix, "restore_bf16_cache", None) is None:
            return False
        n = int(handle.n_tokens or seq.cached_len or 0)
        if n <= 0:
            return False
        try:
            from .v4_dual_pool import stage_official_kv_from_pages

            n_mod, _ = stage_official_kv_from_pages(
                self.model,
                radix,
                handle,
                batch_slot=batch_slot,
                n_tokens=n,
            )
            if n_mod > 0:
                seq._v4_need_stage = False
            return n_mod > 0
        except Exception as e:
            print(f"[sglang-lite] v4 page stage skipped: {e}")
            return False

    def v4_attach_dual_pool_from_entry(
        self, seq: Sequence, entry, radix: Optional[RadixCache]
    ) -> bool:
        """On prefix hit: fork dual-pool pages from the cache entry onto ``seq``."""
        cache = self._v4_prefix_cache
        if cache is None or radix is None or entry is None:
            return False
        if not getattr(entry, "swa_block_ids", None):
            return False
        # Prefer cache.fork so dual_hit_count is updated.
        if cache.radix is None:
            cache.bind_radix(radix)
        handle = cache.fork_dual_pool_for_hit(entry)
        if handle is None:
            return False
        seq._v4_dual_handle = handle
        seq._v4_dual_radix = radix
        seq.swa_block_table = list(handle.swa_blocks)
        seq.comp_block_table = list(handle.comp_blocks)
        # dual_primary entries will stage from pages after hit restore.
        if getattr(entry, "dual_primary", False):
            seq._v4_page_primary = True
        return True

    def _v4_maybe_save_prefix(
        self, seq: Sequence, *, batch_slot: int = 0, radix: Optional[RadixCache] = None
    ) -> None:
        """Snapshot official KV after a completed prompt prefill.

        Phase 0c: dual-write packed + bf16 restore pages into ``radix``. When
        dual-write covers ``kv_cache`` modules, those tensors are **slimed out**
        of the CPU snapshot (``dual_primary``) so pages become the primary
        restore path for KV; ``kv_state`` / ``score_state`` stay in the snapshot.
        """
        cache = self._v4_prefix_cache
        if cache is None or self.model is None:
            return
        if seq.cached_len != len(seq.input_ids):
            return
        from .v4_prefix_cache import snapshot_v4_kv

        buffers = snapshot_v4_kv(self.model, batch_slot=batch_slot)
        if not buffers:
            return

        if radix is not None and cache.radix is None:
            cache.bind_radix(radix)

        swa_ids: List[int] = []
        comp_ids: List[int] = []
        dual_tokens = 0
        dual_layers = 0
        dual_keys: List[str] = []
        dual_primary = False
        if radix is not None and (
            radix.packed_swa_cache is not None or radix.packed_kv_cache is not None
        ):
            try:
                from .v4_dual_pool import dual_write_from_model, slim_snapshot_buffers

                handle = dual_write_from_model(
                    self.model,
                    radix,
                    batch_slot=batch_slot,
                    n_tokens=int(seq.cached_len),
                )
                if handle is not None:
                    swa_ids = list(handle.swa_blocks)
                    comp_ids = list(handle.comp_blocks)
                    dual_tokens = int(handle.n_tokens)
                    dual_layers = int(handle.n_layers_written)
                    dual_keys = list(handle.layer_keys or [])
                    # Sequence keeps the allocate-ref; cache.insert will fork.
                    seq.swa_block_table = list(swa_ids)
                    seq.comp_block_table = list(comp_ids)
                    seq._v4_dual_handle = handle
                    seq._v4_dual_radix = radix
                    # Slim CPU snapshot when we have page-backed kv rows.
                    paged_keys = {k for k in dual_keys if k}
                    if paged_keys and radix.restore_bf16_cache is not None:
                        buffers = slim_snapshot_buffers(buffers, paged_keys)
                        dual_primary = True
                        # Pages are a durable backup; live decode keeps module KV.
                        # Do NOT set _v4_need_stage here (would re-stage every step).
                        seq._v4_page_primary = True
            except Exception as e:
                # Dual-write is best-effort; full CPU snapshot remains restore path.
                print(f"[sglang-lite] v4 dual-write skipped: {e}")

        cache.insert(
            seq.input_ids[: seq.cached_len],
            last_logits=seq.last_logits,
            buffers=buffers,
            swa_block_ids=swa_ids,
            comp_block_ids=comp_ids,
            dual_pool_tokens=dual_tokens,
            dual_pool_layers=dual_layers,
            dual_layer_keys=dual_keys,
            dual_primary=dual_primary,
        )

    def _v4_dual_append_decode(
        self, seq: Sequence, *, batch_slot: int = 0, radix: Optional[RadixCache] = None
    ) -> None:
        """Best-effort append the latest decode token into dual-pool pages.

        Disable hot-path append with ``SGLANG_LITE_V4_DUAL_APPEND=0`` (thruput).
        Default stays on so prefix pages stay coherent for later hits.
        """
        import os

        if os.environ.get("SGLANG_LITE_V4_DUAL_APPEND", "1").lower() in (
            "0",
            "false",
            "no",
            "off",
        ):
            return
        handle = getattr(seq, "_v4_dual_handle", None)
        radix = radix or getattr(seq, "_v4_dual_radix", None)
        if handle is None or radix is None or self.model is None:
            return
        if seq.cached_len <= 0:
            return
        pos = int(seq.cached_len) - 1
        try:
            from .v4_dual_pool import dual_append_from_model

            dual_append_from_model(
                self.model,
                radix,
                handle,
                batch_slot=batch_slot,
                pos=pos,
            )
            seq.swa_block_table = list(handle.swa_blocks)
            seq.comp_block_table = list(handle.comp_blocks)
        except Exception:
            pass

    def _model_forward_paged(
        self,
        input_ids: torch.Tensor,
        seqs: List[Sequence],
        radix: RadixCache,
        is_decode: bool,
    ):
        """Forward with FlashInfer paged attention; no DynamicCache rebuild."""
        self.model_forward_count += 1
        self.last_model_forward_size = int(input_ids.shape[0])
        B, q_len = input_ids.shape
        cached_lens = [s.cached_len for s in seqs]
        q_lens = [q_len] * B
        past_len = cached_lens[0]
        # Decode B=1: reuse static position buffer.
        if is_decode and B == 1 and q_len == 1:
            position_ids = self._decode_input_buf.set_position(past_len)
        else:
            position_ids = (
                torch.arange(
                    past_len, past_len + q_len, device=self.device, dtype=torch.long
                )
                .unsqueeze(0)
                .expand(B, -1)
            )
        # Paged FI path ignores mask for attention; decode uses None so CUDA graph
        # is not tied to a growing mask length.
        if is_decode and B == 1:
            attn = None
        else:
            attn = torch.ones(
                (B, past_len + q_len), dtype=torch.long, device=self.device
            )
        n_heads = int(self.model.config.num_attention_heads)
        sm_scale = float(self.head_dim**-0.5)
        # Decode thruput: pass block_table refs (no per-step list copy of page ids).
        # Prefill still snapshots to avoid concurrent mutation surprises.
        if is_decode and B == 1:
            btables = [seqs[0].block_table]
        else:
            btables = [list(s.block_table) for s in seqs]
        ctx = PagedAttnContext(
            radix=radix,
            block_tables=btables,
            cached_lens=cached_lens,
            q_lens=q_lens,
            is_decode=is_decode,
            num_qo_heads=n_heads,
            num_kv_heads=self.num_kv_heads,
            head_dim=self.head_dim,
            sm_scale=sm_scale,
        )

        use_native = (
            is_decode
            and B == 1
            and q_len == 1
            and self._paged_native is not None
        )

        # When True, decode body writes next greedy token into the static
        # input buffer (device-side) so the next step needs no host token int.
        write_next = bool(
            use_native
            and is_decode
            and B == 1
            and q_len == 1
            and getattr(self, "_paged_write_next_token", True)
        )

        def _body():
            if use_native:
                logits = self._paged_native.forward_logits(input_ids, position_ids)
                if write_next:
                    # Graph-friendly: next input token stays on device.
                    input_ids[0, 0] = logits[0, -1].argmax()
                return logits
            return self.model(
                input_ids=input_ids,
                position_ids=position_ids,
                past_key_values=None,
                use_cache=False,
                attention_mask=attn,
            )

        def _as_outputs(raw):
            if torch.is_tensor(raw):
                return SimpleNamespace(logits=raw)
            return raw

        # Radix-native CUDA graph: plan outside, capture/replay model body.
        # Opt-in only (SGLANG_LITE_CUDA_GRAPH_DECODE=1). Captures on Qwen3+batched_mm
        # but may accumulate more FI/SDPA drift than eager paged — validate before default-on.
        from .cuda_graph import cuda_graph_decode_enabled

        use_cg = (
            is_decode
            and B == 1
            and q_len == 1
            and str(self.device).startswith("cuda")
            and not self._v4_hybrid
            and cuda_graph_decode_enabled(default="0")
        )
        if use_cg:
            page_size = radix.block_size
            total_len = past_len + 1
            npages = (total_len + page_size - 1) // page_size
            self.kernel_backend.begin_forward(ctx)
            try:
                kb = self.kernel_backend
                max_pages = int(getattr(kb, "decode_static_max_pages", 0) or 0)
                buf_gen = int(getattr(kb, "decode_static_buf_gen", -1))
                out = self._paged_decode_cg.maybe_run(
                    model_fn=_body,
                    npages=npages,
                    max_pages=max_pages,
                    buf_gen=buf_gen,
                    enabled=True,
                )
                if out is None:
                    out = _body()
                return _as_outputs(out)
            finally:
                self.kernel_backend.end_forward()

        self.kernel_backend.begin_forward(ctx)
        try:
            return _as_outputs(_body())
        finally:
            self.kernel_backend.end_forward()

    def _model_forward(self, input_ids, past, attention_mask, position_ids=None):
        self.model_forward_count += 1
        self.last_model_forward_size = int(input_ids.shape[0])
        kwargs = {
            "input_ids": input_ids,
            "past_key_values": self._as_model_cache(past),
            "use_cache": True,
        }
        # Match HF generate(): skip pure-ones decode masks (extra work each step).
        if attention_mask is not None:
            skip_ones = False
            try:
                if (
                    past is not None
                    and input_ids.shape[-1] == 1
                    and input_ids.shape[0] == 1
                    and attention_mask.numel() > 0
                    and bool(torch.all(attention_mask == 1).item())
                ):
                    skip_ones = True
            except Exception:
                skip_ones = False
            if not skip_ones:
                kwargs["attention_mask"] = attention_mask
        if position_ids is not None:
            kwargs["position_ids"] = position_ids
            # TF5 / Qwen3: cache_position aligns rope + cache updates.
            if position_ids.numel() > 0:
                kwargs["cache_position"] = position_ids.reshape(-1)
        # Match HF generate(): only materialize last-token logits (big prefill win).
        if self._supports_logits_to_keep():
            kwargs["logits_to_keep"] = 1
        return self.model(**kwargs)

    def _supports_logits_to_keep(self) -> bool:
        flag = getattr(self, "_logits_to_keep_ok", None)
        if flag is not None:
            return flag
        ok = False
        try:
            import inspect

            fwd = getattr(self.model, "forward", None)
            if fwd is not None:
                ok = "logits_to_keep" in inspect.signature(fwd).parameters
        except Exception:
            ok = False
        self._logits_to_keep_ok = ok
        return ok

    def _cache_is_hybrid(self, past) -> bool:
        return cache_is_hybrid(past)

    def _past_for_seq(self, seq: Sequence, radix: RadixCache):
        """CPU/stub path: rebuild HF cache from paged KV when enabled."""
        # Paged pages only store full-attention K/V. A live hybrid cache is the
        # source of truth for Gated DeltaNet conv/recurrent state.
        if self._cache_is_hybrid(getattr(seq, "kv_state", None)):
            return seq.kv_state
        if self.use_paged_as_source and seq.block_table and seq.cached_len > 0:
            self.paged_rebuild_count += 1
            return radix.build_cache(seq.block_table, seq.cached_len)
        return self._as_model_cache(seq.kv_state)

    def _commit_pages(
        self, seq: Sequence, radix: RadixCache, start: int, write_kv: PastKV
    ) -> None:
        if not write_kv:
            raise RuntimeError(
                "empty KV write: past_key_values could not be converted to legacy "
                "(check Transformers DynamicCache / Cache API compatibility)"
            )
        append_len = write_kv[0][0].shape[-2] if write_kv[0][0].dim() == 4 else write_kv[0][0].shape[0]
        self._ensure_blocks(seq, radix, start + append_len)
        self.kernel_backend.append_paged_kv(radix, seq.block_table, start, write_kv)

    def _batch_caches(self, pasts: List):
        """Stack per-seq caches into one batched DynamicCache/legacy list."""
        if not pasts or all(p is None for p in pasts):
            return None
        if any(self._cache_is_hybrid(p) for p in pasts):
            if len(pasts) != 1:
                raise RuntimeError(
                    "hybrid linear-attention cache supports one sequence per forward"
                )
            return pasts[0]
        legacies = []
        for p in pasts:
            if p is None:
                raise RuntimeError("cannot batch None past with non-None peers")
            leg = self._to_legacy_kv(p)
            if not leg:
                raise RuntimeError("empty KV in batch_caches")
            legacies.append(leg)
        # All same seq length (caller guarantees)
        n_layers = len(legacies[0])
        batched = []
        for layer in range(n_layers):
            ks = torch.cat([leg[layer][0] for leg in legacies], dim=0)
            vs = torch.cat([leg[layer][1] for leg in legacies], dim=0)
            batched.append((ks, vs))
        return self._as_model_cache(batched)

    def _split_batch_cache(self, past, index: int, batch_size: int):
        if self._cache_is_hybrid(past):
            if batch_size != 1:
                raise RuntimeError(
                    "hybrid linear-attention cache supports one sequence per forward"
                )
            return past
        legacy = self._to_legacy_kv(past)
        if legacy is None:
            return None
        sliced = []
        for k, v in legacy:
            sliced.append((k[index : index + 1].contiguous(), v[index : index + 1].contiguous()))
        return self._as_model_cache(sliced)

    def _sample(self, logits: torch.Tensor, seq: Sequence) -> int:
        temperature = float(getattr(seq, "temperature", 0.0) or 0.0)
        top_p = float(getattr(seq, "top_p", 1.0) or 1.0)
        top_k = getattr(seq, "top_k", None)
        seed = getattr(seq, "seed", None)
        step = len(seq.output_ids)
        gen = make_generator(self.device, int(seed) + step) if seed is not None else None
        return sample_logits(
            logits, temperature=temperature, top_p=top_p, top_k=top_k, generator=gen
        )

    def _ensure_blocks(self, seq: Sequence, radix: RadixCache, total_len: int) -> None:
        needed = (total_len + radix.block_size - 1) // radix.block_size
        while len(seq.block_table) < needed:
            seq.block_table.extend(radix.allocate_blocks(1))

    def _slice_kv_tail(self, kv: Optional[PastKV], n: int) -> PastKV:
        if not kv:
            return []
        out = []
        for k, v in kv:
            if k.dim() == 4:
                out.append((k[:, :, -n:, :].contiguous(), v[:, :, -n:, :].contiguous()))
            elif k.dim() == 3:
                out.append((k[-n:, :, :].contiguous(), v[-n:, :, :].contiguous()))
            else:
                out.append((k, v))
        return out

    def _attention_mask(
        self, input_ids: torch.Tensor, past, past_len_hint: Optional[int] = None
    ) -> torch.Tensor:
        past_len = 0
        if past_len_hint is not None:
            past_len = past_len_hint
        elif past is not None and hasattr(past, "get_seq_length"):
            past_len = int(past.get_seq_length())
        elif isinstance(past, (list, tuple)) and past:
            k0 = past[0][0]
            past_len = int(k0.shape[-2]) if k0.dim() >= 3 else 0
        total = past_len + input_ids.shape[-1]
        bsz = int(input_ids.shape[0])
        buf = self._attn_mask_buf
        if (
            buf is None
            or buf.device != input_ids.device
            or buf.shape[0] < bsz
            or buf.shape[1] < total
        ):
            # Cap growth in powers-of-two-ish chunks to limit reallocs.
            cols = max(total, 64)
            if buf is not None and buf.shape[1] > 0:
                cols = max(cols, int(buf.shape[1]) * 2)
            self._attn_mask_buf = torch.ones(
                (max(bsz, 1), cols), dtype=torch.long, device=input_ids.device
            )
            buf = self._attn_mask_buf
        return buf[:bsz, :total]

    def _cast_legacy_kv(self, past: PastKV) -> PastKV:
        """Ensure K/V match model dtype (avoids bf16×fp16 → fp32 SDPA promotion)."""
        dt = self.torch_dtype
        out: PastKV = []
        for k, v in past:
            if k.dtype != dt:
                k = k.to(dtype=dt)
            if v.dtype != dt:
                v = v.to(dtype=dt)
            out.append((k, v))
        return out

    def _as_model_cache(self, past):
        if past is None:
            return None
        if isinstance(past, (list, tuple)):
            past = self._cast_legacy_kv(list(past))
            try:
                from transformers import DynamicCache

                # TF 4.x
                if hasattr(DynamicCache, "from_legacy_cache"):
                    return DynamicCache.from_legacy_cache(past)
                # TF 5.x: constructor takes ddp_cache_data=Iterable[(k,v), ...]
                try:
                    return DynamicCache(ddp_cache_data=list(past))
                except TypeError:
                    pass
                cache = DynamicCache()
                for layer_idx, (k, v) in enumerate(past):
                    cache.update(k, v, layer_idx)
                return cache
            except Exception:
                # Last resort: pass legacy list; some remote codes still accept it.
                return past
        # Existing Cache / DynamicCache: pass through. Never materialize all layers
        # to legacy just to check dtypes — that was a multi-ms/step tax on decode.
        # Hybrid linear-attention caches must not be rebuilt from (K, V) either.
        if self._cache_is_hybrid(past):
            return past
        if hasattr(past, "get_seq_length") or hasattr(past, "layers"):
            return past
        return past

    def _to_legacy_kv(self, past) -> Optional[PastKV]:
        """Normalize HF cache → list[(k,v)] for page commit.

        Transformers 4.x: DynamicCache.to_legacy_cache().
        Transformers 5.x: to_legacy_cache may be absent; use layers[i].keys/values
        or Cache.__getitem__ instead.
        """
        if past is None:
            return None
        if isinstance(past, (list, tuple)):
            if not past:
                return None
            if isinstance(past[0], (list, tuple)) and len(past[0]) >= 2:
                return list(past)
            return None
        # Prefer official conversion when present and non-empty.
        if hasattr(past, "to_legacy_cache"):
            try:
                legacy = past.to_legacy_cache()
                if legacy:
                    return list(legacy)
            except Exception:
                pass
        # Transformers 5+ / Cache API: per-layer .keys / .values
        layers = getattr(past, "layers", None)
        if layers is not None:
            out: PastKV = []
            for layer in layers:
                k = getattr(layer, "keys", None)
                v = getattr(layer, "values", None)
                if k is None or v is None:
                    continue
                out.append((k, v))
            if out:
                return out
        # Cache protocol: past[i] -> (k, v)
        try:
            n = len(past)
        except Exception:
            n = 0
        if n > 0:
            out = []
            for i in range(n):
                try:
                    pair = past[i]
                except Exception:
                    break
                if isinstance(pair, (list, tuple)) and len(pair) >= 2:
                    out.append((pair[0], pair[1]))
            if out:
                return out
        return None

    def sample_next(
        self,
        logits: torch.Tensor,
        temperature: float = 0.7,
        top_p: float = 1.0,
        top_k: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> int:
        gen = make_generator(self.device, seed)
        return sample_logits(
            logits, temperature=temperature, top_p=top_p, top_k=top_k, generator=gen
        )


MoEModelRunner = ModelRunner
