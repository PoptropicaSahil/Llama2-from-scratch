# Llama2 from scratch

This is a follow-along activity from Umar Jamil's excellent youtube video on the same
> https://www.youtube.com/watch?v=oM4VmoabDAI

![alt text](images/architecture.png)

![alt text](images/encodings-meme.png)

## ROPE
See nicely `x` and `d` and $\theta$ - all are predefined. The first image below is the more code-able form. Next image with the big matrix is same, but messy to code
![alt text](images/rope-1.png)
![alt text](images/rope-2.png)


![alt text](images/rope-complex-freqs.png)
![alt text](images/rope-complex-freqs-2.png)

## RMSNorm
Earlier frameworks did either batch norm or layernorm. **But RMSNorm paper says recentering is not necessary (computing mean), only scaling is necessary.**
![alt text](images/rms-norm-1.png)


A new statistic (RMS statistic) is introduced, that does not require recentering by mean (does not calculate mean and therefore variance also since variance depends on mean)
![alt text](images/rms-norm-2.png)


## Attention
![alt text](images/multihead-attention-1.png)
![alt text](images/multihead-attention-2.png)

### Issues with Attention
![alt text](images/kv-cache-intro.png) 

Say at T=3, the usual computation gives 3 attention vectors i.e. it will **recompute the attention of previously calculated tokens**! Note that new token prediction is one-by-one only (seq2seq)

At T=4, again all **prev attention scores** are computed
![alt text](images/prev-attn-1.png) 
![alt text](images/prev-attn-2.png)

### KV Cache
We directly use the last token in the `Query` only. The Query needs access to previous `Keys` and `Values` so we keep appending to them.
![alt text](images/kv-cache-1.png)
![alt text](images/kv-cache-2.png)
![alt text](images/kv-cache-3.png)
> $QK^{T}$ has only the useful row

> multiplying with `V` gives only the useful attention


### Grouped Query Attention
Issue: GPUs are very fast at doing computations than doing computations/transfers across memory - new bottleneck is about memory transfers

Solution A: Optimise memory access, but loose some performace - **Multi query attention** with KV cache --> Reduce number of heads for key and value (reason for less performance)

Solution B: **Grouped Multi Query Attention**
Multi head is best performance, Multi query is fastest
![alt text](images/grouped-mqa.png)
