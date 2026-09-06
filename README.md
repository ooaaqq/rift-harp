# RIFT-HARP

RIFT-HARP is a singing voice conversion model and training archive. The
foundation checkpoint and its inference artifacts are available from:

- [RIFT-HARP foundation](https://huggingface.co/ooaaqq/RIFT-HARP)

## Inference

Install the project and provide the pinned ContentVec and PC-NSF assets listed
in the foundation model repository.

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

Inference is eager PyTorch. It uses BF16 and cuDNN SDPA on SM80+ GPUs, or FP16
and PyTorch SDPA fallback on Turing/T4. It does not require `torch.compile` or
FP8.

## License

Project-authored material is released under CC BY-NC-SA 4.0. Third-party
datasets, model components, and checkpoints retain their original licenses and
attribution requirements. See the model cards and artifact lock files for the
pinned third-party sources.

The inference pipeline uses the PC-NSF vocoder implementation from
[openvpi/SingingVocoders](https://github.com/openvpi/SingingVocoders).
