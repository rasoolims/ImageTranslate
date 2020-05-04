from typing import Dict


def _base_config(vocab_size: int, pad_token_id: int, bos_token_id: int, eos_token_id: int) -> Dict:
    config = {
        "attention_probs_dropout_prob": 0.1,
        "hidden_act": "gelu_new",
        "hidden_dropout_prob": 0.1,
        "embedding_size": 128,
        "hidden_size": 4096,
        "initializer_range": 0.02,
        "intermediate_size": 16384,
        "max_position_embeddings": 512,
        "num_attention_heads": 64,  # smaller than usual
        "num_hidden_layers": 12,  # smaller than usual
        "num_hidden_groups": 1,
        "net_structure_type": 0,
        "gap_size": 0,
        "num_memory_blocks": 0,
        "inner_group_num": 1,
        "down_scale_factor": 1,
        "type_vocab_size": 2,
        "vocab_size": vocab_size,
        "pad_token_id": pad_token_id,
        "bos_token_id": bos_token_id,
        "eos_token_id": eos_token_id,
    }

    return config


def _medium_config(vocab_size: int, pad_token_id: int, bos_token_id: int, eos_token_id: int) -> Dict:
    config = {
        "attention_probs_dropout_prob": 0.1,
        "hidden_act": "gelu_new",
        "hidden_dropout_prob": 0.1,
        "embedding_size": 128,
        "hidden_size": 1024,
        "initializer_range": 0.02,
        "intermediate_size": 4096,
        "max_position_embeddings": 512,
        "num_attention_heads": 8,  # smaller than usual
        "num_hidden_layers": 3,  # smaller than usual
        "num_hidden_groups": 1,
        "net_structure_type": 0,
        "gap_size": 0,
        "num_memory_blocks": 0,
        "inner_group_num": 1,
        "down_scale_factor": 1,
        "type_vocab_size": 2,
        "vocab_size": vocab_size,
        "pad_token_id": pad_token_id,
        "bos_token_id": bos_token_id,
        "eos_token_id": eos_token_id,
    }

    return config


def _small_config(vocab_size: int, pad_token_id: int, bos_token_id: int, eos_token_id: int) -> Dict:
    config = {
        "attention_probs_dropout_prob": 0.1,
        "hidden_act": "gelu_new",
        "hidden_dropout_prob": 0.1,
        "embedding_size": 64,
        "hidden_size": 256,
        "initializer_range": 0.02,
        "intermediate_size": 1024,
        "max_position_embeddings": 512,
        "num_attention_heads": 2,  # smaller than usual
        "num_hidden_layers": 2,  # smaller than usual
        "num_hidden_groups": 1,
        "net_structure_type": 0,
        "gap_size": 0,
        "num_memory_blocks": 0,
        "inner_group_num": 1,
        "down_scale_factor": 1,
        "type_vocab_size": 2,
        "vocab_size": vocab_size,
        "pad_token_id": pad_token_id,
        "bos_token_id": bos_token_id,
        "eos_token_id": eos_token_id,
    }

    return config


def _toy_config(vocab_size: int, pad_token_id: int, bos_token_id: int, eos_token_id: int) -> Dict:
    config = {
        "attention_probs_dropout_prob": 0.1,
        "hidden_act": "gelu_new",
        "hidden_dropout_prob": 0.1,
        "embedding_size": 16,
        "hidden_size": 64,
        "initializer_range": 0.02,
        "intermediate_size": 128,
        "max_position_embeddings": 512,
        "num_attention_heads": 2,  # smaller than usual
        "num_hidden_layers": 1,  # smaller than usual
        "num_hidden_groups": 1,
        "net_structure_type": 0,
        "gap_size": 0,
        "num_memory_blocks": 0,
        "inner_group_num": 1,
        "down_scale_factor": 1,
        "type_vocab_size": 2,
        "vocab_size": vocab_size,
        "pad_token_id": pad_token_id,
        "bos_token_id": bos_token_id,
        "eos_token_id": eos_token_id,
    }

    return config
