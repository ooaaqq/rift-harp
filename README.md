# RIFT-HARP

RIFT-HARP is a singing voice conversion model, foundation training stack, and
target-singer finetuning workflow. The released foundation weights and acoustic
artifacts are available on [Hugging Face](https://huggingface.co/ooaaqq/RIFT-HARP).

## Documentation

- [Model architecture and flow objective](docs/architecture.md)
- [Training datasets and third-party components](docs/datasets.md)

## Development

The project targets Python 3.14. With direnv enabled, approve the project once;
subsequent entries load the flake automatically. If direnv is unavailable, run
`nix develop` first. Then synchronize the development dependencies and run the
CPU test suite:

```bash
direnv allow
uv sync --extra dev
uv run pytest
ruff check .
```

All regular CLIs default to `--device auto`, selecting CUDA when available and
CPU otherwise. A short local training or fine-tuning smoke run can also add
`--no-compile`; explicit `--device cpu` remains available for deterministic CPU
runs.

## Foundation training

Training consumes a song-disjoint manifest with cached mel, F0, RMS, and
ContentVec features. The resolved config, feature contract, and flow transform
must describe the same frontend.

```bash
uv sync --extra dev
uv run harp-train \
  --manifest /path/to/foundation.jsonl \
  --output /path/to/run \
  --execute-training
```

Exposure, warmup, EMA, evaluation requests, and checkpoints are measured in
valid target frames. A full checkpoint stores the model, EMA, optimizer,
sampler position, RNG state, and numerical contracts required for exact resume.

## Target-singer finetuning

The target workflow pairs real target mel, F0, and RMS with two kinds of content
input:

$$
C_T \rightarrow M_T, \qquad
\widetilde C_{T\rightarrow j} \rightarrow M_T.
$$

The second content tensor is re-extracted after the frozen foundation converts
the target recording to a carrier singer $j$. Teacher audio is an input
perturbation only; the target remains the original recording.

Build and review the pseudo-content bank:

```bash
uv run harp-build-pseudo-bank \
  --manifest /path/to/target.jsonl \
  --parent /path/to/foundation.pt \
  --output /path/to/pseudo-bank \
  --student-target custom:target \
  --carrier-speaker OpenSinger:female-35 \
  --carrier-speaker OpenSinger:male-08 \
  --content-model /path/to/content-vec-best \
  --pc-nsf-checkout /path/to/SingingVocoders \
  --pc-nsf-lock /path/to/pc_nsf_hifigan.lock.json \
  --vocoder-checkpoint /path/to/pc_nsf_hifigan.ckpt
```

Only accepted variants are sampled. Finetuning selects the target song and crop
before choosing original content or a pseudo variant. The default mixture is
30% original and 70% pseudo content. The acoustic model and target singer code
are optimized together; the original speaker table and four branch-mixing
scalars remain frozen.

```bash
uv run harp-adapt-singer \
  --manifest /path/to/target.jsonl \
  --pseudo-bank /path/to/pseudo-bank/bank.json \
  --parent /path/to/foundation.pt \
  --output /path/to/finetune-run \
  --valid-frames 100000000
```

## Inference

Target-singer inference loads the EMA acoustic model and matching EMA singer
code from one compact checkpoint. Source audio is converted to ContentVec, F0,
RMS, and harmonic features; Euler sampling predicts the target log-mel, and the
pinned PC-NSF vocoder synthesizes the waveform with continuous excitation and
waveform overlap-add.

```bash
uv sync --extra evaluation
uv run harp-convert-singer \
  --config configs/foundation.json \
  --finetune /path/to/singer-inference.pt \
  --input /path/to/vocals.wav \
  --output /path/to/output \
  --content-model /path/to/content-vec-best \
  --pc-nsf-checkout /path/to/SingingVocoders \
  --pc-nsf-lock /path/to/pc_nsf_hifigan.lock.json \
  --vocoder-checkpoint /path/to/pc_nsf_hifigan.ckpt \
  --states ema \
  --steps 32 \
  --guidance 1.0
```

## License

Project-authored material is licensed under
[CC BY-NC-SA 4.0](LICENSE). Third-party datasets, code, and checkpoints retain
their original licenses and attribution requirements.

## Acknowledgements

This project builds on
[RIFT-SVC](https://github.com/Pur1zumu/RIFT-SVC),
[ContentVec](https://github.com/auspicious3000/contentvec),
[torchfcpe](https://github.com/CNChTu/FCPE), and
[OpenVPI SingingVocoders](https://github.com/openvpi/SingingVocoders). The
foundation training datasets and their authors are listed in
[docs/datasets.md](docs/datasets.md).

Questions, bug reports, and reproducible failure cases are welcome in
[GitHub Issues](https://github.com/ooaaqq/rift-harp/issues).
