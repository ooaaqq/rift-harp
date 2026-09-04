# HARP-Core+ v1 contract

HARP-Core+ is intentionally incompatible with RIFT-SVC V4.

## Data and flow

Raw log-mel is centered, rotated by the frozen full-PCA basis, and scaled by
clipped partial whitening. The transformed target is

`y1 = gain * basis * (x1 - mean)`.

Training constructs `yt = (1 - t) z + t y1` in FP32. Per-mode coefficients use
the persisted transformed variance `lambda`:

```text
q      = t^2 lambda + (1 - t)^2
c_in   = q^-1/2
c_skip = (t lambda - (1 - t)) / q
c_out  = sqrt(lambda / q)
v_hat  = c_skip yt + c_out F_theta
```

The model learns only the standardized residual target. CFG combines
conditional and null `F_theta`; the analytic skip is applied exactly once.
Sampling integrates the transformed state in FP32 and applies the inverse
transform only at the endpoint.

## Conditioning

The feature artifact persists the exact float32 Slaney filter centers produced
by the mel grid, their SHA256, corrected harmonic statistics, and standardized
`log(rms + 1e-5)` statistics. Model startup compares every geometry and RMS
field against the resolved config.

Harmonic occupancy uses a per-frame harmonic limit based on
`min(mel_fmax, 0.95 * Nyquist)`, `n^-0.5` weighting, and per-frame maximum
normalization. Standardization is followed by the voiced mask so unvoiced
features remain exactly zero.

Content, pitch, harmonic, and energy branches are normalized separately.
Magnitude-preserving concatenation makes their trainable mixing parameters
describe energy shares rather than allowing branch width to dominate. Initial
shares are approximately 76.0%, 10.8%, 10.8%, and 2.4%.

## Backbone

The temporal backbone remains 1024 wide, 16 blocks deep, with 16 heads,
head-dim 64, QK norm, RoPE, a 2816-channel gated Conv-FFN, and depthwise kernel
31. Harmonic coordinates are injected through zero-initialized linear adapters
before blocks 4, 8, and 12. The main residual head always predicts all 128
transformed modes.

Each block uses low-rank multiplicative time-speaker modulation:

```text
t_low = W_t(time_code)
s_low = W_s(speaker_code)
mixed = SiLU(W_mix([t_low, s_low, t_low * s_low]))
mod   = W_out(mixed)
```

`W_out`, harmonic adapters, and the full-band output are zero initialized.

## Training time and recovery

`seen_valid_frames` is the authoritative exposure clock. The frozen warmup
target is 163,072,000 valid frames. EMA decay for a batch containing `N` valid
frames is `0.9999 ** (N / 16307.2)`; logs report the half-life in valid frames.

Sampler batches are pure functions of seed, epoch, and step. DataLoader worker
seeding uses an independent generator, so rebuilding a loader on resume cannot
shift model noise or timestep RNG. Full checkpoints contain model, EMA,
optimizer, progress counters, sampler position, CPU/CUDA/Python/NumPy RNG,
resolved config, and both numerical contracts. Checkpoint writes are atomic and
the index is append-only.

## Audits

The independent flow audit must pass:

- validation off-diagonal ratio at most 0.10;
- validation maximum absolute correlation at most 0.30;
- transformed variance P95/P05 at most 16;
- transformed variance max/min at most 64;
- median transformed variance between 0.8 and 1.25;
- no lambda or q floor hits;
- FP32 roundtrip maximum error at most `1e-4`;
- a 4x gain cap relative to median raw gain.

Local-field metrics first invert velocity to raw log-mel coordinates without a
mean term, then report historical DCT bands across fixed timesteps and voiced,
unvoiced, F0-quartile, stable-F0, and rapid-F0 strata. Endpoint comparisons,
not raw same-t local NMSE ratios against V3/V4, remain the fair cross-architecture
quality measure.
