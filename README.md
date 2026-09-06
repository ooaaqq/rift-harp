# RIFT-HARP

RIFT-HARP is the checkpoint-incompatible HARP-Core+ foundation training stack.
It keeps the 1024 x 16 temporal backbone while replacing the old mel geometry,
velocity output, conditioning stem, and full-rank AdaLN implementation.

The production contract is:

- FP32 parameters, Adam states, EMA, flow arithmetic, loss, and ODE state;
- rowwise FP8 for QKV, attention output, FF up, and FF down GEMMs;
- BF16 for residual, conditioning, convolution, and cuDNN SDPA compute;
- canonical `96 x 256`, `64 x 384`, and `48 x 512` batches;
- selective rematerialization that retains heavy GEMMs, SDPA, DWConv, and SiLU;
- warmup, EMA, checkpoints, and audit milestones measured in valid frames;
- no V4 checkpoint resume, high-band loss, or auxiliary 2D refiner.

## Required artifacts

All commands consume the same resolved config and manifest. Run them from the
repository root. Existing precomputed `.mel.pt`, `.f0.pt`, `.rms.pt`, and
ContentVec tensors must use the exact Slaney frontend described by the config.

The numerical PCA artifact is already frozen at
`artifacts/flow_transform_v1.npz`. Do not refit it merely because batch sizes
changed: batch size belongs to `batch_runtime_hash`, not
`exposure_semantics_hash`.

Build corrected harmonic and log-RMS statistics from 3M to 5M valid frames:

```bash
harp-build-feature-contract \
  --manifest /path/to/training.content.jsonl \
  --output artifacts/feature_contract_v1.npz \
  --frames 3000000
```

Audit the frozen PCA on an independent 0.5M to 1M-frame sampler stream:

```bash
harp-audit-flow-transform \
  --manifest /path/to/training.content.jsonl \
  --transform artifacts/flow_transform_v1.npz \
  --output artifacts/flow_transform_v1.audit.json \
  --frames 750000 \
  --seed 2027
```

The audit rejects correlation, variance-spread, floor-hit, scale, or roundtrip
contract failures and reports conditional residual-target energy for voiced,
unvoiced, and F0-quartile strata.

Freeze train and song-disjoint shadow panels before training:

```bash
harp-build-audit-panels \
  --manifest /path/to/training.content.jsonl \
  --output artifacts/fixed_panels_v1.json
```

## Preflight and training

On the RTX 5090, run the complete three-shape preflight. It materializes FP32
Adam states and EMA, compiles all canonical shapes with `fullgraph=True` and
`max-autotune`, rotates them for 30 steps, and fails on extra compiled programs.
All static shapes live under one CUDA Graph Trees manager so their captures use
the same device memory pool:

```bash
export TORCHINDUCTOR_CACHE_DIR=/some/persistent/rift-harp-sm120-cache
export TORCHINDUCTOR_FX_GRAPH_CACHE=1
export TORCHINDUCTOR_AUTOGRAD_CACHE=1
uv run python scripts/gpu_preflight.py \
  --rotation-steps 30 \
  --save-cache-artifact artifacts/compiler-cache-sm120.bin
```

Validate config and manifest resolution without starting optimization:

```bash
harp-train \
  --manifest /path/to/training.content.jsonl \
  --output /path/to/new-run
```

Formal training additionally requires a clean Git worktree and a new output
directory:

```bash
harp-train \
  --manifest /path/to/training.content.jsonl \
  --output /path/to/new-run \
  --compiler-cache-artifact artifacts/compiler-cache-sm120.bin \
  --execute-training
```

Resume only from a full checkpoint in the same run directory. Config, feature,
transform, audit, exposure, batch-runtime, sampler-audit, and parameter-role
identities must match; runtime version drift is recorded as a resume event but
is not confused with training semantics.

## Evaluation

Frame milestones produce audit checkpoints and append immutable requests to
`audit_requests.jsonl`. A separate evaluation process consumes those requests,
so endpoint solvers and external metrics cannot perturb training RNG or CUDA
graphs. Run local-field and endpoint panels against the requested checkpoints:

```bash
harp-audit-local-field \
  --manifest /path/to/training.content.jsonl \
  --panels artifacts/fixed_panels_v1.json \
  --checkpoint /path/to/audit.pt \
  --output /path/to/local-field.json

harp-audit-endpoint \
  --manifest /path/to/training.content.jsonl \
  --panels artifacts/fixed_panels_v1.json \
  --checkpoint /path/to/audit.pt \
  --output /path/to/endpoint.json

harp-render-full-panel \
  --manifest /path/to/training.content.jsonl \
  --panels artifacts/fixed_panels_v1.json \
  --checkpoint /path/to/full.pt \
  --output /path/to/full-panel \
  --pc-nsf-checkout /path/to/SingingVocoders \
  --pc-nsf-lock /path/to/pc_nsf_hifigan.lock.json \
  --vocoder-checkpoint /path/to/pc_nsf_hifigan.ckpt

harp-audit-speaker-progress \
  --manifest /path/to/training.content.jsonl \
  --pairs /path/to/pairs.lock.json \
  --calibration /path/to/speaker-calibration.lock.json \
  --anchors /path/to/speaker-calibration-anchors.json \
  --checkpoint /path/to/full.pt \
  --output /path/to/speaker-progress \
  --pc-nsf-checkout /path/to/SingingVocoders \
  --pc-nsf-lock /path/to/pc_nsf_hifigan.lock.json \
  --vocoder-checkpoint /path/to/pc_nsf_hifigan.ckpt
```

The endpoint command covers raw and EMA weights with correct, null, and wrong
speaker reconstruction, both pooled and split by requested context length. The
full-panel command renders fixed raw/EMA PC-NSF audio and records pitch and
waveform-tail diagnostics. The speaker-progress command renders the locked A-to-B
conversion panel and measures normalized progress with the pinned WavLM speaker
encoder and historical source/target anchors.

## Singer Adaptation

Singer adaptation keeps the foundation EMA and all shared model weights frozen.
Stage A learns one 256-dimensional target speaker code. Stage B starts from the
EMA adapter from A and learns only the 17 target-specific AdaLN offsets. The
adapter checkpoint is bound to the exact parent foundation checkpoint.

Run stage A on a target-singer manifest (accepted `train` entries):

```bash
harp-adapt-singer \
  --config configs/foundation.json \
  --manifest /path/to/singer-manifest.jsonl \
  --parent /path/to/foundation-full-or-audit.pt \
  --output /path/to/singer-adapt-a \
  --stage a \
  --valid-frames 12000000
```

Run stage B from the selected stage-A adapter; it loads the stage-A EMA by
default:

```bash
harp-adapt-singer \
  --config configs/foundation.json \
  --manifest /path/to/singer-manifest.jsonl \
  --parent /path/to/foundation-full-or-audit.pt \
  --init-adapter /path/to/singer-adapt-a/adapter-a-step-*.pt \
  --output /path/to/singer-adapt-b \
  --stage b \
  --valid-frames 36000000
```

Use a song/recording-disjoint `dev` split and the fixed Euler32 external
conversion panel to select checkpoints. The adaptation CLI deliberately does
not alter the foundation transform, frontend, null row, or optimizer state.
