# RIFT-HARP

RIFT-HARP is the clean implementation of the incompatible HARP-Core+ model
family. It intentionally does not load or resume RIFT-SVC V4 checkpoints.

The first implementation milestone contains:

- a sampler-exposure frequency transform contract;
- DCT or full-PCA rotation with clipped partial whitening;
- variance-aware analytic flow parameterization;
- normalized frame conditioning and harmonic coordinates;
- multiplicative time-speaker low-rank AdaLN;
- semantic initialization and role-aware AdamW groups.

The production path is BF16 model compute with FP32 flow construction, loss,
and ODE state. FP8 and legacy experiment branches are intentionally absent.

