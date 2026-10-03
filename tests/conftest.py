"""Shared fixtures for sglang-lite tests."""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def tiny_mixtral_path(tmp_path_factory) -> str:
    """Build a tiny Mixtral-style model + tokenizer on disk for CPU tests."""
    path = tmp_path_factory.mktemp("tiny_mixtral")
    _build_tiny_mixtral(Path(path))
    return str(path)


@pytest.fixture(scope="session")
def tiny_mixtral_id(tiny_mixtral_path) -> str:
    return f"fixture:{tiny_mixtral_path}"


def _build_offline_tokenizer(path: Path, vocab_size: int = 256):
    """Build a tiny BPE tokenizer without network access."""
    from tokenizers import Tokenizer
    from tokenizers.models import BPE
    from tokenizers.pre_tokenizers import Whitespace
    from tokenizers.trainers import BpeTrainer
    from transformers import PreTrainedTokenizerFast

    special = ["<unk>", "<pad>", "<eos>", "<bos>"]
    tok_model = Tokenizer(BPE(unk_token="<unk>"))
    tok_model.pre_tokenizer = Whitespace()
    trainer = BpeTrainer(vocab_size=vocab_size, special_tokens=special)
    corpus = [f"hello world {i} test token cache prefix" for i in range(64)]
    tok_model.train_from_iterator(corpus, trainer=trainer)
    tok_path = path / "tokenizer.json"
    path.mkdir(parents=True, exist_ok=True)
    tok_model.save(str(tok_path))
    tok = PreTrainedTokenizerFast(
        tokenizer_file=str(tok_path),
        unk_token="<unk>",
        pad_token="<pad>",
        eos_token="<eos>",
        bos_token="<bos>",
    )
    return tok


def _build_tiny_mixtral(path: Path) -> None:
    import torch
    from transformers import MixtralConfig, MixtralForCausalLM

    path.mkdir(parents=True, exist_ok=True)
    try:
        from transformers import GPT2TokenizerFast

        tok = GPT2TokenizerFast.from_pretrained("gpt2", local_files_only=True)
        if not tok.encode("hello world"):
            raise ValueError("cached GPT2 tokenizer produced no fixture tokens")
    except Exception:
        tok = _build_offline_tokenizer(path)

    # head_dim must be FlashInfer-supported (64/128/…); 16 is rejected by paged kernels.
    config = MixtralConfig(
        vocab_size=len(tok),
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_local_experts=4,
        num_experts_per_tok=2,
        max_position_embeddings=256,
        rms_norm_eps=1e-5,
        rope_theta=10000.0,
        router_aux_loss_coef=0.0,
        bos_token_id=tok.bos_token_id or tok.eos_token_id,
        eos_token_id=tok.eos_token_id,
        pad_token_id=tok.eos_token_id,
    )
    torch.manual_seed(0)
    model = MixtralForCausalLM(config)
    model.eval()
    model.save_pretrained(path)
    tok.save_pretrained(path)
    (path / "sglang_lite_moe_family").write_text("mixtral\n")
