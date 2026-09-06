# Datasets and components

The foundation model used the following singing datasets. Sampling first
selects a dataset, then a physical singer, then a song; singer and song priors
use bounded square-root duration weighting.

| Dataset | Sampling | Role | Source and terms |
| --- | ---: | --- | --- |
| OpenSinger | 43.5% | Real singer diversity | [Project](https://github.com/Multi-Singer/Multi-Singer.github.io) |
| GTSinger | 32% | Multilingual real singing and techniques | [Dataset](https://huggingface.co/datasets/AaronZ345/GTSinger), [license](https://github.com/AaronZ345/GTSinger/blob/master/dataset_license.md) |
| M4Singer | 20% | Real Mandarin singing | [Project and license](https://github.com/M4Singer/M4Singer) |
| ACE-Opencpop | 2% | Synthetic timbre augmentation | [Dataset](https://huggingface.co/datasets/espnet/ace-opencpop-segments), [paper](https://arxiv.org/abs/2401.17619) |
| Opencpop | 1.5% | Real Mandarin anchor | [Project](https://wenet-e2e.github.io/opencpop/), [dataset](https://modelscope.cn/datasets/wenet/opencpop) |
| Kiritan | 1% | Low-weight real singing | [Official distribution](https://zunko.jp/kiridev/login.php), [corpus paper](https://www.jstage.jst.go.jp/article/ast/42/3/42_E2074/_pdf) |

Opencpop and ACE-Opencpop share one source-song family capped at 3.5% so a real
song and its synthetic derivatives cannot cross the train/validation split.
Synthetic ACE audio is excluded from vocoder training.

The released inference path also uses:

| Component | Purpose | Source |
| --- | --- | --- |
| ContentVec | Source content representation | [auspicious3000/contentvec](https://github.com/auspicious3000/contentvec) |
| torchfcpe | F0 extraction | [CNChTu/FCPE](https://github.com/CNChTu/FCPE) |
| PC-NSF | Mel-to-waveform synthesis | [openvpi/SingingVocoders](https://github.com/openvpi/SingingVocoders) |

Dataset and component names here are attribution, not relicensing. Each source
retains its own license, access conditions, and attribution requirements.
