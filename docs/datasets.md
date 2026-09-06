# Datasets and components

The foundation model used the following singing datasets.

| Dataset | Recordings | Hours | Singers | Songs | Sampling | Source and terms |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| OpenSinger | 42,947 | 51.900 | 76 | 1,127 | 43.5% | [Project](https://github.com/Multi-Singer/Multi-Singer.github.io) |
| GTSinger | 26,711 | 75.291 | 20 | 610 | 32% | [Dataset](https://huggingface.co/datasets/AaronZ345/GTSinger), [license](https://github.com/AaronZ345/GTSinger/blob/master/dataset_license.md) |
| M4Singer | 20,889 | 29.689 | 20 | 419 | 20% | [Project](https://github.com/M4Singer/M4Singer), [license](https://github.com/M4Singer/M4Singer/blob/master/dataset_license.md) |
| ACE-Opencpop | 105,209 | 128.195 | 30 | 100 | 2% | [Dataset](https://huggingface.co/datasets/espnet/ace-opencpop-segments), [paper](https://arxiv.org/abs/2401.17619) |
| Opencpop | 3,752 | 5.218 | 1 | 100 | 1.5% | [Project](https://wenet-e2e.github.io/opencpop/), [dataset](https://modelscope.cn/datasets/wenet/opencpop) |
| Kiritan | 43 | 3.036 | 1 | 43 | 1% | [Official distribution](https://zunko.jp/kiridev/login.php), [corpus paper](https://www.jstage.jst.go.jp/article/ast/42/3/42_E2074/_pdf) |
| **Total** | **199,551** | **293.329** | **148** | **-** | **100%** | |

## Sampling

Each training crop is selected in four steps:

1. Choose a dataset with the fixed probability in the table.
2. Choose a physical singer within that dataset using square-root duration
   weights, bounded between 0.5 and 2 times uniform probability.
3. Choose one of that singer's songs with the same bounded square-root rule.
4. Choose a recording within the song in proportion to its valid frame count.

The crop length is then selected from 256, 384, and 512 frames with probabilities
20%, 30%, and 50%. Batch sizes are 96, 64, and 48 respectively, keeping 24,576
requested frames per foundation update.

Opencpop and ACE-Opencpop share one source-song family capped at 3.5% so a real
song and its synthetic derivatives cannot cross the train/validation split.
Synthetic ACE audio is excluded from vocoder training.

## Summary

The accepted corpus contains 199,551 recordings, 293.329 hours of audio, and
148 namespaced singers. The song-disjoint split contains 179,067 training,
10,449 validation, and 10,035 test recordings. After feature-length
reconciliation, the training split contains 81,348,986 valid mel frames.

Song counts are not summed because ACE-Opencpop and Opencpop share the same 100
source-song identities.

The released inference path also uses:

| Component | Purpose | Source |
| --- | --- | --- |
| ContentVec | Source content representation | [auspicious3000/contentvec](https://github.com/auspicious3000/contentvec) |
| torchfcpe | F0 extraction | [CNChTu/FCPE](https://github.com/CNChTu/FCPE) |
| PC-NSF | Mel-to-waveform synthesis | [openvpi/SingingVocoders](https://github.com/openvpi/SingingVocoders) |

Dataset and component names here are attribution, not relicensing. Each source
retains its own license, access conditions, and attribution requirements.
