from typing import Dict


def _base_config(vocab_size: int, pad_token_id: int, eos_token_id: int) -> Dict:
    config = {
        "attention_head_size": 64,
        "hidden_size": 256,
        "feed_forward_size": 512,
        "max_position_embeddings": 4096,
        "num_attention_heads": 2,
        "pad_token_id": pad_token_id,
        "vocab_size": vocab_size,
        "eos_token_id": eos_token_id,
        "axial_pos_embds": False,
        "attn_layers": ["local", "lsh", "local", "lsh", "local", "lsh"],
    }

    return config


def _small_config(vocab_size: int, pad_token_id: int, eos_token_id: int) -> Dict:
    config = {
        "attention_head_size": 4,
        "hidden_size": 64,
        "feed_forward_size": 64,
        "max_position_embeddings": 4096,
        "num_attention_heads": 2,
        "pad_token_id": pad_token_id,
        "vocab_size": vocab_size,
        "eos_token_id": eos_token_id,
        "axial_pos_embds": False,
        "attn_layers": ["local", "lsh", "local", "lsh"],
    }

    return config


def _medium_config(vocab_size: int, pad_token_id: int, eos_token_id: int) -> Dict:
    config = {
        "attention_head_size": 8,
        "hidden_size": 128,
        "feed_forward_size": 256,
        "max_position_embeddings": 4096,
        "num_attention_heads": 2,
        "pad_token_id": pad_token_id,
        "vocab_size": vocab_size,
        "eos_token_id": eos_token_id,
        "axial_pos_embds": False,
        "attn_layers": ["local", "lsh", "local", "lsh", "local", "lsh"],
    }
    return config
