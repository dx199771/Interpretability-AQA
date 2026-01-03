# UIL-AQA: Uncertainty-aware, Interpretable Long-term Action Quality Assessment

This is the official PyTorch implementation of the paper **"UIL-AQA: Uncertainty-aware, Interpretable Long-term Action Quality Assessment"**. 

This work is a substantial extension of our **BMVC 2024 (Oral)** paper: *"Interpretable Long-term Action Quality Assessment"*.

---

### 📄 Publications & Links

[![IJCV](https://img.shields.io/badge/Journal-IJCV_2025-blue.svg?style=for-the-badge)](https://link.springer.com/article/10.1007/s11263-025-02638-6)
[![BMVC Oral](https://img.shields.io/badge/BMVC_2024-Oral-orange.svg?style=for-the-badge)](https://arxiv.org/abs/2408.11687)
[![Project Page](https://img.shields.io/badge/Project-Page-green.svg?style=for-the-badge)](https://andrewjohngilbert.github.io/InterpretAQA/)

---

## 🚀 Key Features
* **Interpretable**: Uses Temporal Decoding Networks and Attention Loss to provide granular feedback on action clips.
* **Long-term Analysis**: Specialized for evaluating complex, long-duration athletic performances.
* **Uncertainty-aware**: Incorporates uncertainty estimation for more robust AQA scoring.

---

## 🖼️ Framework & Method Overview

<table border="0">
  <tr>
    <td width="45%">
      <img src="https://media.springernature.com/full/springer-static/image/art%3A10.1007%2Fs11263-025-02638-6/MediaObjects/11263_2025_2638_Fig1_HTML.png" width="100%">
    </td>
    <td width="55%">
      <img src="https://media.springernature.com/full/springer-static/image/art%3A10.1007%2Fs11263-025-02638-6/MediaObjects/11263_2025_2638_Fig4_HTML.png" width="100%">
    </td>
  </tr>
</table>

---



## 📂 Datasets

We evaluate our method on three benchmark datasets. We highly recommend using the **pre-extracted features** provided below for efficient training.

### 1. [LOng-form GrOup (LOGO)](https://github.com/shiyi-zh0408/LOGO)
- **Video Frames**: [Google Drive](https://drive.google.com/file/d/1-MpOQSo72TZhoTzr8bqviDezi-ge7o6V/view?usp=sharing) | [Baidu Drive](https://pan.baidu.com/s/1GNi_ZcbSq6oi2SEX_iuFwA?pwd=v329) (Code: `v329`)
- **Annotations & Split**: [Google Drive](https://drive.google.com/drive/folders/1i4lG1_iwP0lHMCvyYlqS8h7YRQCSRFyA?usp=drive_link) | [Baidu Drive](https://pan.baidu.com/s/1UwlGzCeq_UjY0GbOnaHXxw?pwd=ojgf) (Code: `ojgf`)
- **Video Swin Transformer (VST) Features**: [Baidu Drive](https://pan.baidu.com/s/1zFZgyJ1CCVd67ZfQZyYC6g) (Code: `9ojl`)

### 2. [Figure Skating Video (Fis-V)](https://github.com/chmxu/MS_LSTM)
- **Raw Videos**: [Download Link](https://drive.google.com/file/d/1FQ0-H3gkdlcoNiCe8RtAoZ3n7H1psVCI/view?usp=sharing)
- **Features & Labels**: [OneDrive](https://1drv.ms/u/s!AqXkt0Mw7p9llWEihc533CB87U5P?e=EadhCo) (via [GDLT](https://github.com/xuangch/CVPR22_GDLT) repo)

### 3. [Rhythmic Gymnastics (RG)](https://github.com/qinghuannn/ACTION-NET)
- **Dataset**: [Download Link](https://1drv.ms/u/s!ApyE_Lf3PFl2issDbaK99shfZRKchg?e=fdd2eO)
- **Features & Labels**: [OneDrive](https://1drv.ms/u/s!AqXkt0Mw7p9llVaV2oV1mwmdAICG) (via [GDLT](https://github.com/xuangch/CVPR22_GDLT) repo)

### Data Structure
Organize the data in your `$DATASET_ROOT` as follows:
```text
$DATASET_ROOT
├── LOGO
│   ├── logo_feats/
│   │   └── WorldChampionship2019_free_final/
│   ├── LOGO Anno&Split/
│   │   └── anno_dict.pkl
│   └── Video_result/
│       └── WorldChampionship2019_free_final/
│           └── 0/
│               └── 00000.jpg
├── GDLT_data (RG)
│   ├── swintx_avg_fps25_clip32/
│   │   └── Ball_084.npy
│   ├── test.txt
│   └── train.txt
└── GDLT_data (Fis-V)
    ├── swintx_avg_fps25_clip32/
    │   └── 100.npy
    ├── test.txt
    └── train.txt
```

## 🛠️ Installation

### Step 1: Create and activate conda environment ###

```bash
conda create -n interaqa python=3.8 -y
conda activate interaqa
```
### Step 2: Install dependencies ###

```bash
pip install -r requirements.txt
```
### Step 3: Training ###
To train the model on different datasets, run:
```bash
# Train on LOGO
python3 main.py --config configs/train_logo.py

# Train on RG
python3 main.py --config configs/train_rg.py

# Train on Fis-V
python3 main.py --config configs/train_fisv.py
```
---
## ✍️ Citation

If you find our work or code helpful for your research, please consider citing both the IJCV journal paper and the original BMVC conference paper:

**IJCV 2026 (Journal Extension)**
```bibtex
@article{dong2026uilaqa,
  title={UIL-AQA: Uncertainty-Aware Clip-Level Interpretable Action Quality Assessment},
  author={Dong, Xu and Liu, Xinran and Li, Wanqing and Adeyemi-Ejeye, Anthony and Gilbert, Andrew},
  journal={International Journal of Computer Vision},
  volume={134},
  number={24},
  year={2026},
  publisher={Springer},
  doi={10.1007/s11263-025-02638-6},
  url={[https://doi.org/10.1007/s11263-025-02638-6](https://doi.org/10.1007/s11263-025-02638-6)}
}
```
**BMVC 2024 (Oral)**
```bibtex
@inproceedings{dong2024interpretable,
  title={Interpretable Long-term Action Quality Assessment},
  author={Dong, Xu and Liu, Xinran and Li, Wanqing and Adeyemi-Ejeye, Anthony and Gilbert, Andrew},
  booktitle={Proceedings of the British Machine Vision Conference (BMVC)},
  year={2024},
  eprint={2408.11687},
  archivePrefix={arXiv},
  url={[https://arxiv.org/abs/2408.11687](https://arxiv.org/abs/2408.11687)}
}
```


