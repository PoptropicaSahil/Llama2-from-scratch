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