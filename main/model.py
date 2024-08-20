import torch
import torch.nn as nn

from main.model_args import ModelArgs

import torch.nn.functional as F
import math


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
    # Shape: (B, seq_len, n_heads, head_dim) -> (B, seq_len, n_heads, head_dim/2, 2)
    # head_dim/2 because we are pairing conscecutive
    
    # Suppose initial shape is (32, 128, 16, 64)
    # After the reshape:
    # *x.shape[:-1] gives us (32, 128, 16) i.e. unpacking upto last dim
    # The last dimension (64) is reshaped into (-1, 2)
    # 64 into (-1, 2) gives (32, 2)
    # Final result is (32, 128, 16 ,32, 2)
    x = x.float().reshape(*x.shape[:-1], -1, 2)

    # Transformation 2
    # Shape: (B, seq_len, n_heads, head_dim/2, 2) -> (B, seq_len, n_heads, head_dim/2)
    # Real part and imaginary part
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


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch_size, seq_len, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    else:
        return (
            # (B, seq, N_kv_heads, 1, head_dim)
            x[:, :, :, None, :] # We add a dimension
            .expand(batch_size, seq_len, n_kv_heads, n_rep, head_dim) # expand it n_rep times
            .reshape(batch_size, seq_len, n_kv_heads * n_rep, head_dim) # flatten that dimension
        )



class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

        # Gamma parameter - learnable ofcourse!!
        self.gamma = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor):
        # Shape: (B, seq_len, dim) * (B, seq_len, 1) -> (B, seq_len, dim)
        # rsqrt is reciprocal of square root
        # summation and 1/n in formula is by .mean only!
        
        # x.pow(2) shape: (B, seq_len, dim)
        # x.pow(2).mean(-1) shape (B, seq_len)
        # x.pow(2).mean(-1, keepdim = True) shape (B, seq_len, 1) i.e. KEEPS the last dim

        # (B, seq_len, dim) * (B, seq_len, 1) 
        # --> broadcasting for second term! 
        # --> (B, seq_len, dim) * (B, seq_len, dim)
        # --> (B, seq_len, dim)
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x

    def forward(self, x: torch.Tensor):
        # Shape: (dim) * (B, seq_len, dim) -> (B, seq_len, dim)
        return self.gamma * self._norm(x.float()).type_as(x)


class SelfAttention(nn.Module):
    """
    This code does not take care of GPU parallelisation
    The code is only for inferencing!
    """

    def __init__(self, args: ModelArgs):
        super().__init__()
        # defaults to multi-head attention if n_heads_kv is not set
        # refer to images/grouped-mqa.png

        # Shows number of heads for keys and values
        self.n_heads_kv = (
            args.n_heads_kv if args.n_heads_kv is not None else args.n_heads
        )
        # Shows number of heads for queries
        self.n_heads_q = args.n_heads
        # Shows how many times the HEADS of keys and values should be repeated to match the number of heads in queries
        self.n_rep = self.n_heads_q // self.n_heads_kv
        # Shows dimension of each head
        self.head_dim = args.dim // args.n_heads

        # The Weight matrices
        # Shape almost equivalent to (dim, dim)
        # but to account for multiple heads we write as (dim, head_dim*n_heads)
        self.wq = nn.Linear(args.dim, self.head_dim * self.n_heads_q, bias=False)
        self.wk = nn.Linear(args.dim, self.head_dim * self.n_heads_kv, bias=False)
        self.wv = nn.Linear(args.dim, self.head_dim * self.n_heads_kv, bias=False)
        self.wo = nn.Linear(self.head_dim * args.n_heads, args.dim, bias=False)

        # Initialise KV caches
        # Just have the batch size dimention extra here, otherwise usual only
        # QUESTION: Should it not be (B, seq_len, n_heads_kv, dim/n_heads_kv)
        self.cache_k = torch.zeros(
            (args.max_batch_size, args.max_seq_len, self.n_heads_kv, self.head_dim)
        )
        self.cache_v = torch.zeros(
            (args.max_batch_size, args.max_seq_len, self.n_heads_kv, self.head_dim)
        )

    def forward(self, x: torch.Tensor, start_pos: int, freqs_complex: torch.Tensor):
        batch_size, seq_len, _ = x.shape  # (B, 1, dim)
        # x is only the latest token from previous pass
        # Output of this forward is also (B, 1, dim)

        # Apply Wq, Wk and Wv matrices to queries, keys and values
        # No change in shape here
        # (B, 1, dim) -> (B, 1, n_heads_q * head_dim)
        xq = self.wq(x)

        # May result in shape change since n_heads_kv may be smaller than n_heads_q
        # (B, 1, dim) -> (B, 1, n_heads_kv * head_dim)
        xk = self.wk(x)
        xv = self.wv(x)

        # Explicitly divide last dimension into n_heads_(q/k/v) and head_dim
        # (B, 1, n_heads_q * head_dim) -> (B, 1, n_heads_q, head_dim)
        xq = xq.view(batch_size, seq_len, self.n_heads_q, self.head_dim)
        # (B, 1, n_heads_kv * head_dim) -> (B, 1, n_heads_kv, head_dim)
        xk = xk.view(batch_size, seq_len, self.n_heads_kv, self.head_dim)
        xv = xv.view(batch_size, seq_len, self.n_heads_kv, self.head_dim)

        # Apply rotary position encodings only to xq and xk
        # Does not change the size of tensor since x_out = x_out.reshape(*x.shape)
        # device = x.device!!
        # RoPE not applied to the values
        xq = apply_rotary_embeddings(xq, freqs_complex=freqs_complex, device=x.device)
        xk = apply_rotary_embeddings(xk, freqs_complex=freqs_complex, device=x.device)

        # Apply KV cache
        # Query will be single token (x)

        # Replace the entry in the cache with THIS token (for every batch)
        # Like putting in the cache (remember it is initialised with zeros)
        self.cache_k[:batch_size, start_pos : start_pos + seq_len] = xk
        self.cache_v[:batch_size, start_pos : start_pos + seq_len] = xv

        # Retreive all the cached keys and values so far
        # remember we need everything upto this point from the cache (matrix) for upcoming multiplications
        # (B, seq_len_kv, n_heads_kv, head_dim)
        keys = self.cache_k[:batch_size, : start_pos + seq_len]
        values = self.cache_v[:batch_size, : start_pos + seq_len]

        # Repeat the heads of the K and V to match number of heads of Q
        keys = repeat_kv(keys, self.n_rep)
        values = repeat_kv(values, self.n_rep)

        # We are going to split the embedding over the multiple heads
        # (B, 1, n_heads_q, head_dim) -> (B, n_heads_q, 1, head_dim)
        xq = xq.transpose(1,2)
        keys = keys.transpose(1,2)
        values = values.transpose(1,2)

        # attention
        # (B, n_heads_q, 1, head_dim) @ (B, n_heads_q, head_dim, seq_len_kv) -> (B, n_heads_q, 1, seq_len_kv)
        scores = torch.matmul(xq, keys.transpose(2,3)) / math.sqrt(self.head_dim)
        scores = F.softmax(scores.float(), dim=1).type_as(xq)

        # (B, n_heads_q, 1, seq_len_kv) @ (B, n_heads_q, seq_len_kv, head_dim) -> (B, n_heads_q, 1, head_dim)
        output = torch.matmul(scores, values)

        # split attention across heads
        # (B, n_heads_q, 1, head_dim) -> (B, 1, n_heads_q, head_dim) -> (B, 1, dim)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        
        # (B, 1, dim) -> (B, 1, dim)
        output = self.wo(output)

        return output


class FeedForward(nn.Module):

    def __init__(self, args: ModelArgs):
        super().__init__()

        hidden_dim = 4 * args.dim # convention
        hidden_dim = int(2 * hidden_dim / 3) # convention
        if args.ffn_dim_multiplier is not None:
            hidden_dim = int(args.ffn_dim_multiplier * hidden_dim)

        # Round the hidden_dim to nearest multiple of multiple_of parameter (model param)
        hidden_dim = args.multiple_of * ((hidden_dim + args.multiple_of - 1) // args.multiple_of)

        self.w1 = nn.Linear(args.dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, args.dim, bias=False)
        self.w3 = nn.Linear(args.dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor):
        swish = F.silu(self.w1(x))
        x_V = self.w3(x)
        x = swish * x_V
        x = self.w2(x)
        return x





class EncoderBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = self.dim // self.n_heads

        self.attention = SelfAttention(args)
        self.feed_forward = FeedForward(args)

        # RMSNorm before self attention
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        # RMSNorm before feedforward (after self attention)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(self, x: torch.Tensor, start_pos: int, freqs_complex: torch.Tensor):
        """
        Args:
            x (torch.Tensor): _description_
            start_pos (int): We deal only with one token at a time, start_pos is its position
            freqs_complex (torch.Tensor): _description_
        """
        # Note assert seq_len == 1 in forward pass of Transformers
        # Refer to architecture.png
        # Shape: (B, seq_len, dim) + (B, seq_len, dim) -> (B, seq_len, dim)
        h = x + self.attention.forward(self.attention_norm(x), start_pos, freqs_complex)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out


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
