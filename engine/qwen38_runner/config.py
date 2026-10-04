"""Fail closed on model shapes/equations outside the dedicated prototype."""


def attention_layout(rank, world, attention_tp):
    if (
        world != 8
        or type(attention_tp) is not int
        or attention_tp not in (2, 4, 8)
        or not 0 <= rank < world
    ):
        raise ValueError("attention mesh requires eight GPUs and attention TP2/TP4/TP8")
    start = rank // attention_tp * attention_tp
    return tuple(range(start, start + attention_tp)), rank - start


def validate_config(config, world, capacity):
    if world != 8 or not 4 <= capacity <= 4096 or capacity % 4:
        raise ValueError("prototype requires TP8/EP8 and page-aligned context <=4096")
    if config.get("model_type") != "qwen4_exp":
        raise ValueError("expected qwen4_exp")
    quant = config.get("quantization_config", {})
    if quant.get("quant_algo") != "NVFP4":
        raise ValueError("expected NVFP4")
    group = quant.get("config_groups", {}).get("group_0", {})
    for name in ("weights", "input_activations"):
        if any(
            group.get(name, {}).get(k) != v
            for k, v in {"group_size": 16, "num_bits": 4, "dynamic": False, "type": "float"}.items()
        ):
            raise ValueError(f"unsupported NVFP4 {name}")
    text = config["text_config"]
    expected = {
        "hidden_size": 2560,
        "num_hidden_layers": 48,
        "hc_count": 4,
        "hc_lowrank": 320,
        "num_attention_heads": 24,
        "num_key_value_heads": 2,
        "head_dim": 256,
        "linear_num_key_heads": 16,
        "linear_num_value_heads": 48,
        "linear_key_head_dim": 128,
        "linear_value_head_dim": 128,
        "linear_conv_kernel_dim": 4,
        "num_experts": 512,
        "num_experts_per_tok": 10,
        "moe_intermediate_size": 640,
        "shared_expert_intermediate_size": 640,
        "output_gate_type": "sigmoid",
        "indexer_budget": 2048,
        "indexer_compress_ratio": 4,
        "indexer_head_dim": 128,
        "indexer_kv_heads": 1,
        "indexer_n_heads": 4,
        "ngram_size": 3,
        "heads_per_ngram": 8,
        "ple_conv_kernel_size": 4,
        "ple_embed_dim": 2560,
        "ple_layer_ids": [2],
        "split_ngram_parts": 128,
        "ple_embedding_dtype": "float8_e4m3fn",
        "make_ngram_vocab_size_divisible_by": 128,
        "bos_token_id": 248044,
        "eos_token_id": 248044,
        "rms_norm_eps": 1e-6,
        "vocab_size": 248320,
        "hidden_act": "silu",
        "attention_bias": False,
        "mamba_ssm_dtype": "float32",
    }
    for key, value in expected.items():
        if text.get(key) != value:
            raise ValueError(f"unsupported Qwen3.8 {key}: {text.get(key)!r}")
    if text.get("layer_types") != (["linear_attention"] * 3 + ["full_attention"]) * 12:
        raise ValueError("unsupported attention layer order")
    rope = text.get("rope_parameters", {})
    if any(
        rope.get(k) != v
        for k, v in {
            "rope_theta": 10000000,
            "partial_rotary_factor": 0.25,
            "rope_type": "default",
        }.items()
    ):
        raise ValueError("unsupported RoPE")
    return text
