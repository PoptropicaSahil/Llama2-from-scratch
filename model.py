from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

# import torch.nn.functional as F
# import math


@dataclass
class ModelArgs:
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32  # num heads for Queries
    n_kv_heads: Optional[int] = None  # num heads for K, V
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


def precompute_theta_pos_frequencies(
    head_dim: int, seq_len: int, device: str, theta_constant: float = 10000.0
):
    # Check images/rope-complex-freqs.png

    # theta param of 10,000 comes from the paper
    assert head_dim % 2 == 0, "Dimension must be even"

    # Build the theta parameters
    # Formula is theta_i = 10000 ^ (-2(i-1)/dim) for i = [1, 2, ... , dim/2]
    # Shape: (head_dim / 2) since paper says i from 1 to d/2
    theta_numerator = torch.arange(
        start=0, end=head_dim, step=2
    ).float()  # EQUIVALENT because it is 2*(...)
    theta = theta_constant ** (-theta_numerator / head_dim)
    theta.to(device)

    # so far theta shape is (head_dim/2)

    # Build the m parameter
    # Shape: (seq_len)
    # m is position of token in the sentence
    # max positions we have is seq_len (passed max_seq_len*2 for safer side)
    m = torch.arange(seq_len, device=device)

    # Multiply each theta by each position using outer product
    # Shape: (seq_len) outer product (head_dim/2) -> (seq_len, head_dim/2)
    freqs = torch.outer(m, theta).float()

    # COMPLEX NUMBERS FORMS:
    # z = x + i * y
    # <==> z = r(cos theta + i * sin theta)
    # <==> z = r * exp(i * theta)

    # Compute complex numbers in the polar form
    # c = R * exp(i * m * theta)
    # Shape: (seq_len, head_dim/2) -> (seq_len, head_dim/2)
    freqs_complex = torch.polar(abs=torch.ones_like(freqs), angle=freqs)
    return freqs_complex

def apply_rotary_embeddings(x: torch.Tensor, freqs_complex: torch.Tensor, device: str):
    """

    Args:
        x (torch.Tensor): token to which we want to apply rotary embeddings
        freqs_complex (torch.Tensor): output form compute_theta_pos_frequencies function, only for this token's positions
        device (str): _description_
    """
    # Check images/rope-complex-freqs-2.png

    # Remember dim = n_heads * head_dim
    # x will be passed to multi-head attention

    # Transformation 1
    # Shape: (B, seq_len, n_heads, head_dim) -> (B, seq_len, n_heads, head_dim/2)
    # head_dim/2 because we are pairing conscecutive 
    x = x.float().reshape(*x.shape[:-1], -1, 2)
    
    # Transformation 2
    x_complex = torch.view_as_complex(x) 

    # Shape: (seq_len, head_dim/2) -> (1, seq_len, head_dim/2) -> (1, seq_len, 1, head_dim/2)
    # Equivalent to B = 1, n_heads = 1
    # Required so that element-wise multiplication can be done
    freqs_complex = freqs_complex.unsqueeze(0).unsqueeze(2)

    # Shape: (B, seq_len, n_heads, head_dim/2) * (1, seq_len, 1, head_dim/2) -> (B, seq_len, n_heads, head_dim/2)
    x_rotated = x_complex * freqs_complex

    # Transformation 3
    # Shape: (B, seq_len, n_heads, head_dim/2) -> (B, seq_len, n_heads, head_dim/2, 2)
    x_out = torch.view_as_real(x_rotated)

    # Transformation 4
    # Shape: (B, seq_len, n_heads, head_dim / 2, 2) -> (B, seq_len, n_heads, head_dim)
    # Flatten x back to its original shape
    x_out = x_out.reshape(*x.shape)

    return x_out.type_as(x).to(device)

    


class Transformer(nn.Module):
    def __init__(self, args: ModelArgs) -> None:
        super().__init__()

        assert args.vocab_size != -1, "vocab_size must be set while loading tokenizer"

        self.args = args

        # self.vocab_size = self.args.vocab_size
        # self.n_layers = self.args.n_layerss

        self.tok_embeddings = nn.Embedding(self.args.vocab_size, self.args.dim)

        self.layers = nn.ModuleList()
        for _ in range(self.args.n_layers):
            self.layers.append(EncoderBlock(args))

        self.norm = RMSNorm(self.args.dim, eps=self.args.norm_eps)  # normalisation
        self.output = nn.Linear(self.args.dim, self.args.vocab_size, bias=False)

        self.freqs_complex = precompute_theta_pos_frequencies(
            head_dim=self.args.dim // self.args.n_heads,
            seq_len=self.args.max_seq_len * 2,
            device=self.args.device,
        )

    def forward(self, tokens: torch.Tensor, start_idx: int):
        # tokens shape = (batch_size, seq_len)
        batch_size, seq_len = tokens.shape
        assert (
            seq_len == 1
        ), "seq_len must be 1, since only 1 token at a time can be processed"

        # each token in seq_len gets its embedding
        # (B, seq_len) -> (B, seq_len, dim)
        h = self.tok_embeddings(tokens)

        # Retrieve the pairs (m, theta) corresponding to the positions [start_pos, start_pos + seq_len]
        freqs_complex = self.freqs_complex[start_idx : start_idx + seq_len]

        # Consecutively apply all the encoder layers
        for layer in self.layers:
            h = layer(h, start_idx, freqs_complex)
        h = self.norm(h)
        h = self.output(h).float()

        # will apply softmax during inference
        return h
