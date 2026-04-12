<div align="center">
<img src="assets/meanfuser.png" width="800">
<h1>[🎉CVPR 2026] MeanFuser</h1>
<h4>Fast One-Step Multi-Modal Trajectory Generation and Adaptive Reconstruction via MeanFlow for End-to-End Autonomous Driving</h4>

[![Paper](https://img.shields.io/badge/ArXiv-A42C25?style=for-the-badge&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2602.20060)
[![License](https://img.shields.io/badge/Apache--2.0-019B8F?style=for-the-badge&logo=apache)](https://github.com/wjl2244/MeanFuser/blob/main/LICENSE) 

<h4 align="center"><em><a href="https://github.com/wjl2244">Junli Wang</a>, 
<a href="https://github.com/Liuxueyi">Xueyi Liu</a>,
<a href="https://github.com/ZhengYinan-AIR">Yinan Zheng</a>, 
<a href="https://github.com/ZebinX">Zebing Xing</a>, 
<a href="https://github.com/Philipflyg">Pengfei Li</a>, 
<a href="https://scholar.google.com/citations?user=McEfO8UAAAAJ&hl=en">Guang Li</a>, 
  
<a href="https://openreview.net/profile?id=~Kun_Ma3">Kun Ma</a>, 
<a href="https://openreview.net/profile?id=%7EGuang_Chen1">Guang Chen</a>, 
<a href="https://scholar.google.com/citations?user=68tXhe8AAAAJ&hl=en">Hangjun Ye</a>, 
<a href="https://openreview.net/profile?id=~Zhongpu_Xia1">Zhongpu Xia</a>, 
<a href="https://long.ooo/">Long Chen</a>, 
<a href="https://scholar.google.com/citations?user=snkECPAAAAAJ&hl=en">Qichao Zhang</a>📧</em>
</h4>

<h4 align="center">
<br>📧 indicates corresponding authors.<br>
<b > SKL-MAIS, CASIA &nbsp; | &nbsp; Xiaomi EV  &nbsp; | &nbsp; AIR, Tsinghua University </b>
</h4>
</div>

---

## 📢 News
- **`[2026/4/12]`** We released [NAVSIMv2 code](https://github.com/wjl2244/MeanFuser/tree/NAVSIMv2).
- **`[2026/3/20]`** We released code and [checkpoints](https://arxiv.org/abs/2602.20060).
- **`[2026/2/25]`** We released our [paper](https://arxiv.org/abs/2602.20060) on arXiv. 
- **`[2026/2/21]`** 🎉 Accepted to CVPR 2026.


## 📌 Table of Contents
- 📋 [TODO List](#-todo-list)
- 🏛️ [Model Zoo](#%EF%B8%8F-model-zoo)
- 🎯 [Getting Started](#-getting-started)
- 📦 [Data Preparation](#-data-preparation)
  - [Download Dataset](#1-download-dataset)
  - [Set Up Configuration](#2-set-up-configuration)
  - [Cache the Dataset](#3-cache-the-dataset)
- ⚙️ [Evaluation](#%EF%B8%8F-evaluation)
  - [Evaluation](#1-evaluation)
- ❤️ [Acknowledgements](#%EF%B8%8F-acknowledgements)

## 📋 TODO List
- [ ] HUGSIM code release (Apr. 2026).
- [x] NAVSIMv2 navtest code release (Apr. 2026).
- [x] Checkpoints release (Mar. 2026).
- [x] Code release (Mar. 2026).
- [x] Paper release (Feb. 2026).

## 🏛️ Model Zoo

| Method | Backbone | Benchmark | PDMS | Weight Download |
| :---: | :---: | :---:  | :---:  | :---: |
| MeanFuser | [ResNet-34](https://drive.google.com/file/d/1-6mtwHsrZt4TyH4lfFEJTT8_dnnkejAI/view?usp=drive_link) | NAVSIM | 89.0 | [Google Drive](https://drive.google.com/file/d/16989kIYhM3wQgxjSKvRFfK9cdKZfuU2P/view?usp=drive_link) |
| MeanFuser + [BeyondDrive](https://github.com/wjl2244/BeyondDrive) | [ResNet-34](https://drive.google.com/file/d/1-6mtwHsrZt4TyH4lfFEJTT8_dnnkejAI/view?usp=drive_link) | NAVSIM | 90.3 | [Google Drive](https://drive.google.com/file/d/1ztGNrSHNuxQBQf9HVzyV9lwCHb9B0aKn/view?usp=drive_link) |
| MeanFuser | [ResNet-34](https://drive.google.com/file/d/1-6mtwHsrZt4TyH4lfFEJTT8_dnnkejAI/view?usp=drive_link) | HUGSIM | - | [Google Drive](https://drive.google.com/file/d/1e0BdvpJHriai4zxKSXGsK7yagcme5QeS/view?usp=drive_link) |


## 🎯 Getting Started

### 1. Clone MeanFuser Repo

```bash
git clone -b NAVSIMv2 https://github.com/wjl2244/MeanFuser.git
cd MeanFuser
```

### 2. Create Environment

```bash
conda create -n meanfuser python=3.9 -y
conda activate meanfuser
pip install -e .
```

## 📦 Data Preparation
**NOTE: Please review and agree to the [LICENSE file](https://motional-nuplan.s3-ap-northeast-1.amazonaws.com/LICENSE) file before downloading the data.**

### 1. Download Dataset

#### a. Download via NAVSIM offical installation.
Follow the instructions in the [NAVSIM installation guide](https://github.com/autonomousvision/navsim/blob/main/docs/install.md#2-download-the-dataset) to download the dataset.


#### b. Download via Hugging Face
Alternatively, you can download the dataset using Hugging Face with the following commands:
```bash
export HF_ENDPOINT="https://huggingface.co"
# export HF_ENDPOINT="http://hf-mirror.com"  # Uncomment this line if you are in China

# Install the huggingface_hub tool
pip install -U "huggingface_hub"

# Download the OpenScene dataset
hf download --repo-type dataset OpenDriveLab/OpenScene --local-dir ./navsim_dataset/ --include "openscene-v1.1/*"

# Download the map data
cd download && ./download_maps.sh
```

### 2. Set Up Configuration
Move the download data to create the following structure.

```angular2html
navsim_workspace/
├── MeanFuser/
├── dataset/
│    ├── maps/
│    ├── navsim_logs/
│    │   ├── test/
│    │   ├── trainval/
│    ├── sensor_blobs/
│    │   ├── test/
│    │   ├── trainval/
└── cache/
     └── navtest_v2_metric_cache/
```

### 3. Cache the Dataset
We provide a script to cache the dataset and metrics.
```bash
cd MeanFuser

# Cache the metric.
bash scripts/evaluation/run_metric_cache.sh
```

## ⚙️ Evaluation

### 1. Evaluation
Please download the pre-trained checkpoints from [here](https://drive.google.com/drive/folders/1VGzTzvoJkd65aGLn5bp64r86QLrcPxI3?usp=sharing) and place them in the `navsim_workspace/MeanFuser/exp/` directory.

```bash
cd MeanFuser

bash scripts/evaluation/run_meanfuser_evaluation_one_stage.sh
```

## ❤️ Acknowledgements

We acknowledge all the open-source contributors for the following projects to make this work possible:

- [MeanFlow](https://github.com/zhuyu-cs/MeanFlow) | [NAVSIM](https://github.com/autonomousvision/navsim) | [HUGSIM](https://github.com/hyzhou404/NAVSIM) | [GTRS](https://github.com/NVlabs/GTRS)
