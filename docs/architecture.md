# HARP-Core+ v1 contract

HARP-Core+ is intentionally checkpoint-incompatible with RIFT-SVC V4.

The frozen v1 design consists of four changes only:

1. sampler-exposure frequency rotation and clipped partial whitening;
2. variance-aware analytic velocity skip and residual prediction;
3. normalized content, pitch, harmonic, and energy conditioning;
4. explicit multiplicative time-speaker low-rank AdaLN.

The temporal backbone remains 1024 wide and 16 blocks deep. The first quality
run uses BF16 model compute, FP32 flow arithmetic and ODE state, cuDNN SDPA,
and `torch.compile(mode="max-autotune")`.

The transform must be fitted before training. Its validation covariance must
meet both `offdiag_ratio <= 0.10` and `max_abs_corr <= 0.30`. Failure is fatal;
training must not silently substitute per-bin normalization.
