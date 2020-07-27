from typing import Dict


def _bert_config(vocab_size: int, pad_token_id: int, bos_token_id: int, eos_token_id: int) -> Dict:
    config = {
        "attention_probs_dropout_prob": 0.1,
        "hidden_act": "gelu",
        "hidden_dropout_prob": 0.1,
        "hidden_size": 768,
        "initializer_range": 0.02,
        "intermediate_size": 3072,
        "max_position_embeddings": 512,
        "num_attention_heads": 12,
        "num_hidden_layers": 6,
        "vocab_size": vocab_size,
        "pad_token_id": pad_token_id,
        "bos_token_id": bos_token_id,
        "eos_token_id": eos_token_id,
    }
    return config


def get_config(vocab_size: int, pad_token_id: int, bos_token_id: int, eos_token_id: int, enc_layer: int = 6,
               embed_dim: int = 768, intermediate_dim: int = 3072):
    config = _bert_config(vocab_size=vocab_size, pad_token_id=pad_token_id, bos_token_id=bos_token_id,
                          eos_token_id=eos_token_id)
    config["num_hidden_layers"] = enc_layer
    config["intermediate_size"] = intermediate_dim
    config["hidden_size"] = embed_dim
    return config
