<h1 align="center">Multimodal Flow</h1>

<h3 align="center">
  Unified Flow Modeling of Language and Vision in Embedding Spaces
</h3>

<p align="center">
  A fully continuous framework for multimodal understanding, generation, and
  the flexible sequences in between.
</p>

<p align="center">
  <a href="https://github.com/Hongyuan-Tao">Hongyuan Tao</a><sup>1</sup>&nbsp;&nbsp;
  <a href="https://xwcv.github.io/">Xinggang Wang</a><sup>1</sup><sup>*</sup>&nbsp;&nbsp;
  <a href="https://lh-zhu.github.io/index.html">Lianghui Zhu</a><sup>1</sup>&nbsp;&nbsp;
  <a href="https://scholar.google.com/citations?user=F7hPv1QAAAAJ&amp;hl=en">Yongkang Li</a><sup>1</sup>&nbsp;&nbsp;
  <a href="https://weiyc.github.io/">Yunchao Wei</a><sup>2</sup>&nbsp;&nbsp;
  <a href="https://scholar.google.com/citations?user=nRc8u6gAAAAJ&amp;hl=zh-CN">Bin Feng</a><sup>1</sup><br>
  <a href="https://scholar.google.com/citations?user=PIeNN2gAAAAJ&amp;hl=en">Shaoyu Chen</a><sup>3</sup>&nbsp;&nbsp;
  <a href="https://scholar.google.com/citations?user=pCY-bikAAAAJ&amp;hl=zh-CN">Qian Zhang</a><sup>3</sup>&nbsp;&nbsp;
  <a href="https://scholar.google.com/citations?user=IyyEKyIAAAAJ&amp;hl=en">Chang Huang</a><sup>3</sup>&nbsp;&nbsp;
  <a href="https://scholar.google.com/citations?user=y5zkBeMAAAAJ&amp;hl=en">Kai Yu</a><sup>3</sup>
</p>

<p align="center">
  <sup>1</sup>Huazhong University of Science and Technology
  &nbsp;&nbsp;
  <sup>2</sup>Beijing Jiaotong University
  &nbsp;&nbsp;
  <sup>3</sup>Horizon Robotics
  <br>
  <sub><sup>*</sup>Corresponding author:
  <a href="mailto:xgwang@hust.edu.cn">Xinggang Wang</a></sub>
</p>

<p align="center">
  <a href="https://huggingface.co/hustvl/Multimodal-Flow"><img src="https://img.shields.io/badge/Model-Hugging%20Face-f39c12" alt="Hugging Face model"></a>
  <a href="https://arxiv.org/abs/2609.40362"><img src="https://img.shields.io/badge/arXiv-2609.40362-b31b1b" alt="arXiv paper"></a>
  <a href="#citation"><img src="https://img.shields.io/badge/Cite-BibTeX-8e44ad" alt="Citation"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache--2.0-f39c12" alt="Apache-2.0 license"></a>
</p>

## News

- **Oct. 1, 2026:** 🎉 Our [paper](https://arxiv.org/abs/2609.40362) is now available on arXiv! Check it out to learn more about Multimodal Flow.
- **Oct. 1, 2026:** We have released the first version of the Multimodal Flow code. Welcome to try it and build on it!

## Table of Contents

- [Introduction](#introduction)
- [Getting Started](#getting-started)
- [Documentation](#documentation)
- [License](#license)

## Introduction

Multimodal Flow explores a general way to model heterogeneous information in a
fully continuous space. Rather than forcing every modality into a shared
discrete token vocabulary or giving each task its own model, MF represents
different modalities as continuous hyperchunks and places them in one ordered,
causal stream. Language, images, and other structured signals can remain
distinct while sharing the same context, transformations, and generative
dynamics.

<p align="center">
  <font color="#2f80ed"><strong>🌊 One continuous space</strong></font>
  &nbsp;&nbsp;·&nbsp;&nbsp;
  <font color="#8e44ad"><strong>🧩 Flexible sequences</strong></font>
  &nbsp;&nbsp;·&nbsp;&nbsp;
  <font color="#27ae60"><strong>⚙️ Configurable recipes</strong></font>
</p>
<p align="center">
  <img src="docs/assets/multimodal-flow-architecture.png"
       alt="Multimodal Flow architecture" width="100%">
</p>


<p align="center"><em>Different modalities meet in one continuous,
chunk-causal flow and return to their native spaces through their decoders.</em></p>
During training, an ordered multimodal context is encoded into continuous
hyperchunks and target chunks are predicted in parallel. During inference,
completed chunks are generated sequentially and appended to the context. The
same interface therefore supports understanding, generation, and sequences
that move between them.

<p align="center">
  <img src="docs/assets/mixed_multimodal_pretraining.png"
       alt="Multimodal Flow training and inference" width="100%">
</p>


<p align="center"><em>Parallel target prediction during training and
sequential continuation during inference.</em></p>

The open-source implementation turns this idea into a flexible continuous
multimodal training and inference framework. A model is built from configurable
modality codecs, semantic sequences, task definitions, physical layouts,
objectives, and generation paths, so the same backbone can be shaped into a
concrete research recipe without rewriting the core system. Text, images, and
mixed sequences share one continuous interface, while their native
representations, temporal structure, supervision, and decoders remain explicit.
This makes the framework a natural foundation for language modeling, visual
understanding, image generation, editing, video sequences, and future
modalities that can be expressed as ordered continuous chunks. Training and
inference follow the same sequence contract: targets are predicted in parallel
during training and unfolded step by step at inference time.

## Getting Started

🚀 <font color="#2f80ed"><strong>Start locally, configure once, and run.</strong></font>
MF is a local-first, configuration-driven framework. The shortest path is:

~~~bash
git clone https://github.com/hustvl/Multimodal-Flow.git
cd Multimodal-Flow
conda env create -f environment.yaml
conda activate multimodal-flow
~~~

The released 1.6B pretraining and SFT models are available on
[Hugging Face](https://huggingface.co/hustvl/Multimodal-Flow). Use the
[Inference guide](docs/INFERENCE.md) to run them or a checkpoint produced by
your own training run.

🧭 <font color="#27ae60"><strong>Follow this path for a first run:</strong></font>

| Goal | Guide |
| --- | --- |
| 1. Prepare datasets and model assets | [Data preparation](docs/DATA.md) |
| 2. Run pretraining or SFT | [Training](docs/TRAINING.md) |
| 3. Use the resulting checkpoint | [Inference](docs/INFERENCE.md) |
| Reference | [Architecture and extensions](docs/ARCHITECTURE.md) |

The shortest complete path is:

~~~bash
torchrun --standalone --nproc-per-node=4 -m mf train --config configs/quickstart-1.6b.yaml
torchrun --standalone --nproc-per-node=4 -m mf sft --config configs/sft.yaml --init-from <PRETRAIN_CHECKPOINT>
mf infer --checkpoint <CHECKPOINT> text --prompt "A short continuation"
~~~

## Documentation

| Guide | Contents |
| --- | --- |
| [Data](docs/DATA.md) | Public datasets, model assets, local layout, and data adapters |
| [Training](docs/TRAINING.md) | Pretraining, SFT, configs, launch, resume, and checkpoints |
| [Inference](docs/INFERENCE.md) | Text, captioning, image generation, and checkpoint loading |
| [Architecture](docs/ARCHITECTURE.md) | Dataflow, sequence contracts, registries, and extension points |

## Citation

If you use Multimodal Flow in your research, please cite the project and follow
the licensing terms of all external assets.

```bibtex
@misc{tao2026multimodalflow,
  title        = {Multimodal Flow},
  author       = {Tao, Hongyuan and Wang, Xinggang and Zhu, Lianghui and
                  Li, Yongkang and Wei, Yunchao and Feng, Bin and
                  Chen, Shaoyu and Zhang, Qian and Huang, Chang and Yu, Kai},
  year         = {2026},
  howpublished = {\url{https://github.com/hustvl/Multimodal-Flow}},
  note         = {Paper and arXiv information to be added}
}
```

## Acknowledgements

We thank the authors and contributors of
[ELF](https://github.com/lillian039/ELF),
[minit2i](https://github.com/Hope7Happiness/minit2i-torch), and
[RAE](https://github.com/bytetriper/RAE), along with other high-quality
open-source projects that helped make this work possible.

## License

The Multimodal Flow source code is released under the Apache License 2.0; see
[LICENSE](LICENSE). Third-party components retain the licenses and notices
specified in `THIRD_PARTY_NOTICES.md`. Model weights, datasets, and benchmark
scorers are external assets and are not redistributed here.
