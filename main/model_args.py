from dataclasses import dataclass
from typing import Optional

@dataclass
class ModelArgs:
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32  # num heads for Queries
    n_heads_kv: Optional[int] = None  # num heads for K, V
    vocab_size: int = -1  # will be set while loading tokenizer

    # hidden dim of feedforward layer
    # useful for grouped query attention
    multiple_of: int = 256
    ffn_dim_multiplier: Optional[float] = None

    norm_eps: float = 1e-5

    # for KV cache
    max_batch_size: int = 32
    max_seq_len: int = 2048

    device: Optional[str] = None
