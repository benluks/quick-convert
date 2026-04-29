# Emotion Compensation

From the [emotion compensation](https://github.com/xiaoxiaomiao323/emotion-compensation) pipeline, based on the paper by Miao et al. (2025).

## Steps

1. Install deps:

```bash
uv sync --extra emotion-compensation
```

2. Download pre-trained models

From the [English demo inference script](https://github.com/xiaoxiaomiao323/emotion-compensation/blob/877a1c5b3a9cdfc1e33d7b02494573e0a852fe26/gen/scripts/engl_scripts/01_demo.sh#L19)

Make sure you're in the correct donor directory:

```bash
pwd
# path/to/quick-convert/quick_convert/components/donors/emotion-compensation
```

```bash
wget https://zenodo.org/record/6529898/files/pretrained_models_anon_xv.tar.gz
tar -xzvf pretrained_models_anon_xv.tar.gz
cd pretrained_models_anon_xv/
wget https://dl.fbaipublicfiles.com/hubert/hubert_base_ls960.pt
```


```BibTex
@article{10.1016/j.csl.2025.101810,
author = {Miao, Xiaoxiao and Zhang, Yuxiang and Wang, Xin and Tomashenko, Natalia and Soh, Donny Cheng Lock and Mcloughlin, Ian},
title = {Adapting general disentanglement-based speaker anonymization for enhanced emotion preservation},
year = {2025},
issue_date = {Nov 2025},
publisher = {Academic Press Ltd.},
address = {GBR},
volume = {94},
number = {C},
issn = {0885-2308},
url = {https://doi.org/10.1016/j.csl.2025.101810},
doi = {10.1016/j.csl.2025.101810},
journal = {Comput. Speech Lang.},
month = nov,
numpages = {15},
keywords = {Disentanglement-based speaker anonymization, Orthogonal householder neural network, Emotion encoder, Emotion compensation}
}
```