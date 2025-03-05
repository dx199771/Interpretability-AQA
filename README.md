# Interpretable Long-term Action Quality Assessment
The official PyTorch implementation of the paper "Interpretable Long-term Action Quality Assessment".

[![arXiv](https://img.shields.io/badge/arXiv-2408.11687-red.svg)](https://arxiv.org/abs/2408.11687)



## Datasets
We use three datasets for training and testing
Please download the dataset form these link:
### [LOng-form GrOup (LOGO)](https://github.com/shiyi-zh0408/LOGO)
- Video_Frames:  [Google Drive](https://drive.google.com/file/d/1-MpOQSo72TZhoTzr8bqviDezi-ge7o6V/view?usp=sharing) or [baidu Drive](https://pan.baidu.com/s/1GNi_ZcbSq6oi2SEX_iuFwA?pwd=v329) (extract number: v329) 
- Annotations and Split: [Google Drive](https://drive.google.com/drive/folders/1i4lG1_iwP0lHMCvyYlqS8h7YRQCSRFyA?usp=drive_link) or [Baidu Drive](https://pan.baidu.com/s/1UwlGzCeq_UjY0GbOnaHXxw?pwd=ojgf) (extract number: ojgf)
- We also provide Video Swin Transformer (VST) feature for LOGO dataset [Baidu_Drive](https://pan.baidu.com/s/1zFZgyJ1CCVd67ZfQZyYC6g) (extract number: 9ojl)
### [Figure Skating Video (Fis-V)](https://github.com/chmxu/MS_LSTM)
- The raw videos have been uploaded. You can download from here [Fis-V](https://drive.google.com/file/d/1FQ0-H3gkdlcoNiCe8RtAoZ3n7H1psVCI/view?usp=sharing).
- The features and label files of Fis-V dataset can be download from here [Fis-V](https://1drv.ms/u/s!AqXkt0Mw7p9llWEihc533CB87U5P?e=EadhCo) from [GDLT](https://github.com/xuangch/CVPR22_GDLT) repository.
### [Rhythmic Gymnastics (RG)](https://github.com/qinghuannn/ACTION-NET)
- The Rhythmic Gymnastics Dataset can be downloaded from here [RG](https://1drv.ms/u/s!ApyE_Lf3PFl2issDbaK99shfZRKchg?e=fdd2eO).
- The features and label files of RG dataset can be download from here [RG](https://1drv.ms/u/s!AqXkt0Mw7p9llVaV2oV1mwmdAICG) from [GDLT](https://github.com/xuangch/CVPR22_GDLT) repository.

We recommend using pre-extracted features for above three datasets.

The data structure should be like:
```
$DATASET_ROOT
├── LOGO
|  ├── logo_feats
|    ├── WorldChampionship2019_free_final
|  ├── LOGO Anno&Split
|    ├── anno_dict.pkl
|  ├── Video_result
|    ├── WorldChampionship2019_free_final
|       ├── 0
|          ├── 00000.jpg
├── GDLT_data
|  ├── swintx_avg_fps25_clip32
|    ├── Ball_084.npy
|  ├── test.txt
|  ├── train.txt
├── GDLT_data
|  ├── swintx_avg_fps25_clip32
|    ├── 100.npy
|  ├── test.txt
|  ├── train.txt
└──
```

## Getting started
This code was tested on xx and requires:

### Setup environment

#### a. Create a conda virtual environment and activate it.
```
conda create -n interaqa python=3.8 -y
conda activate interaqa
```
#### b. Install PyTorch and required packages
```
pip install -r requirements.txt
```

### Train a Model

**Train LOGO dataset**
```
python3 main.py --config configs/train_logo.py
```
**Train RG dataset**
```
python3 main.py --config configs/train_rg.py
```
**Train Fis-V dataset**
```
python3 main.py --config configs/train_fisv.py
```
## Citation
If our project is helpful for your research, please consider citing:
```
@misc{dong2024interpretablelongtermactionquality,
      title={Interpretable Long-term Action Quality Assessment}, 
      author={Xu Dong and Xinran Liu and Wanqing Li and Anthony Adeyemi-Ejeye and Andrew Gilbert},
      year={2024},
      eprint={2408.11687},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2408.11687}, 
}
```

## Acknowledgement

