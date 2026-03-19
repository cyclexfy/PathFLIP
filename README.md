# PathFLIP: Fine-grained Language-Image Pretraining for Versatile Computational Pathology

[![Paper](https://img.shields.io/badge/arXiv-Paper-b31b1b.svg?style=flat-square)](https://arxiv.org/abs/2512.17621)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg?style=flat-square)](LICENSE)
![alt text](https://img.shields.io/badge/AAAI-2026-blue)

> **PathFLIP** is a novel vision-language framework for holistic Whole Slide Image (WSI) interpretation. By decomposing slide-level captions into region-level sub-captions and leveraging Large Language Models (LLMs), PathFLIP achieves precise visual-language grounding and instruction-aware WSI interpretation.

## 🌟 Key Features

While Vision-Language Models (VLMs) have achieved notable progress in computational pathology, the gigapixel scale and spatial heterogeneity of WSIs continue to pose challenges. **PathFLIP** addresses these issues with the following capabilities:

* 🧩 **Fine-grained Visual-Language Grounding:** Decomposes slide-level captions into region-level sub-captions and generates text-conditioned region embeddings, capturing fine-grained correspondences across thousands of patches.
* 🤖 **LLM-Powered Instruction Following:** Seamlessly follows diverse clinical instructions and adapts to varied diagnostic contexts by harnessing the reasoning power of LLMs.
* 🎯 **Versatile Task Adaptation:** Efficiently handles multiple paradigms, including slide-level classification, WSI-text retrieval, fine-grained lesion localization, and instruction following.
* ⚡ **High Efficiency:** Outperforms existing large-scale pathological VLMs on four representative benchmarks while requiring significantly less training data.

---

## 🏗️ Architecture Overview

PathFLIP proposes a region-aware pretraining strategy to bridge the gap between massive gigapixel visual contexts and textual diagnostic descriptions. 

![PathFLIP Framework](docs/overview.png)
*(Brief description of the figure: The overall pipeline of PathFLIP, illustrating the decomposition of slide-level captions and the text-conditioned region embedding generation.)*

---

## ✏️ Citation

If you find PathFLIP useful in your research, please consider citing our paper:

```bibtex
@article{liu2025pathflip,
  title={Pathflip: Fine-grained language-image pretraining for versatile computational pathology},
  author={Liu, Fengchun and Jiang, Songhan and Cai, Linghan and Wang, Ziyue and Zhang, Yongbing},
  journal={arXiv preprint arXiv:2512.17621},
  year={2025}
}
```

## 🙏 Acknowledgement
We would like to thank the open-source community for their invaluable contributions, specifically the repositories of [CLAM](https://github.com/mahmoodlab/CLAM), [CONCH](https://github.com/mahmoodlab/CONCH) and [BLIP2](https://github.com/salesforce/LAVIS).