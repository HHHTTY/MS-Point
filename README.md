# MS-Point: Multi-Modal Spectral Learning for Self-Supervised 3D Point Cloud Understanding

<p align="center">
  <b>Hierarchical-Spectral Structured Cross-Modal Pretraining for Transferable 3D Point Cloud Representation Learning</b>
</p>


<p align="center">
  <img src="https://img.shields.io/badge/Task-Self--Supervised%203D%20Pretraining-blue" alt="Task">
  <img src="https://img.shields.io/badge/Modality-2D--3D%20Cross--Modal-green" alt="Modality">
  <img src="https://img.shields.io/badge/Backbone-DGCNN%20%7C%20PointNet-orange" alt="Backbone">
  <img src="https://img.shields.io/badge/Framework-PyTorch-ee4c2c" alt="PyTorch">
  <img src="https://img.shields.io/badge/Status-Anonymous%20Submission-lightgrey" alt="Anonymous Submission">
</p>


<p align="center">
  <a href="#overview">Overview</a> |
  <a href="#method">Method</a> |
  <a href="#installation">Installation</a> |
  <a href="#datasets">Datasets</a> |
  <a href="#training-and-evaluation">Training & Evaluation</a> |
  <a href="#results">Results</a> |
  <a href="#citation">Citation</a>
</p>


---

## Overview

<p align="center">
  <img src="pipeline.png" width="92%" alt="MS-Pointframework overview"/>
</p>

MS-Point investigates a complementary question:

> Can cross-modal supervision exploit the internal geometric-semantic hierarchy and spectral structure already present in point cloud backbones, rather than constraining only a single global embedding?

Our answer is a **hierarchical-spectral structured cross-modal learning framework**. MS-Point explicitly extracts intermediate representations from multiple point-backbone stages, decomposes selected features into low- and high-frequency branches through graph-filter-inspired operators, matches these structured 3D branches with hierarchical coarse/fine image-side supervision, and adaptively fuses the resulting representations into a unified downstream descriptor.

The codebase supports:

- self-supervised 2D--3D pretraining on paired point clouds and rendered images;
- CrossPoint-compatible global-only baseline reproduction;
- hierarchical-only and full hierarchical-spectral MS-Point variants;
- linear SVM evaluation on ModelNet40, ModelNet10, and ScanObjectNN;
- few-shot classification evaluation;
- part segmentation fine-tuning on ShapeNetPart;
- semantic segmentation fine-tuning on S3DIS;
- component ablations, hyperparameter sensitivity analysis, and efficiency profiling.

> **Double-blind review note.** This README is written for an anonymous submission repository. Author names, final paper URLs, non-anonymous tracking services, and permanent checkpoint links should be added only when permitted by the venue policy.

## Motivation

Point cloud understanding is central to 3D vision, robotics, autonomous driving, and embodied perception. Self-supervised learning is particularly attractive for point clouds because dense annotations are expensive and difficult to obtain. Cross-modal learning further strengthens self-supervision: rendered images offer semantic information that can complement sparse and irregular 3D geometry.

The original CrossPoint paradigm learns a global point representation using two coupled objectives:

1. **intra-modal consistency** between two augmented point cloud views;
2. **cross-modal consistency** between a point cloud prototype and its rendered image representation.

This global alignment is effective, but it does not explicitly exploit two forms of structure that are naturally present in a point encoder.

### Hierarchical semantic structure

For DGCNN-like point encoders, intermediate layers provide distinct geometric-semantic abstraction levels:

- shallow layers respond to local neighborhoods, edges, curvature, and fine geometric motifs;
- intermediate layers aggregate regional patterns and part-level structures;
- deep layers encode increasingly object-semantic information;
- the final pooled feature captures global identity.

When supervision is attached only to the final descriptor, intermediate features remain indirect computational states rather than explicitly trained, transferable representations.

### Spectral frequency structure

Point features can also be viewed as signals defined over a neighborhood graph. Under a graph signal processing perspective:

- **low-frequency features** capture smooth dominant shape, stable contours, and neighborhood-consistent structure;
- **high-frequency features** capture residual variation, boundaries, corners, and fine-grained geometric detail.

The two branches are complementary. Therefore, MS-Point models multi-scale structure along two axes: **semantic depth** and **frequency bandwidth**.

---

## Installation

### Upstream-Compatible Environment

The official CrossPoint implementation lists the following central dependencies:

```text
pytorch 1.9.0
 torchvision 0.10.0
 wandb 0.12.1
 pillow 8.4.0
 numpy 1.21.0
 h5py 3.3.0
 lightly 1.1.21
 gdown 3.13.0
```

For direct comparability with the upstream baseline, use a conservative environment:

```bash
conda create -n mspoint python=3.8 -y
conda activate mspoint

# Select a CUDA toolkit compatible with the available hardware.
conda install pytorch=1.9.0 torchvision=0.10.0 cudatoolkit=11.1 -c pytorch -c nvidia

pip install wandb==0.12.1 pillow==8.4.0 numpy==1.21.0 \
    h5py==3.3.0 lightly==1.1.21 gdown==3.13.0 \
    scikit-learn pyyaml tqdm tensorboard
```

### Installation from the Released Repository

```bash
git clone <ANONYMOUS_REPOSITORY_URL>
cd MS-Point

conda create -n mspoint python=3.8 -y
conda activate mspoint
pip install -r requirements.txt
```

### Check the Environment

```bash
python - <<'PY'
import torch
import torchvision
print("PyTorch:", torch.__version__)
print("Torchvision:", torchvision.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
PY
```

### Hardware Recommendations

| Experiment                      | Minimum Practical Setup      | Recommended Setup              |
| ------------------------------- | ---------------------------- | ------------------------------ |
| Feature extraction / linear SVM | 1 GPU or CPU                 | 1 GPU                          |
| Few-shot classification         | 1 GPU or CPU                 | 1 GPU with cached features     |
| ShapeNet pretraining            | 1 GPU with sufficient memory | 2--4 GPUs with AMP             |
| ShapeNetPart fine-tuning        | 1 GPU                        | 1--2 GPUs                      |
| S3DIS fine-tuning               | 1 high-memory GPU            | Multi-GPU                      |
| Full ablation suite             | Multiple sequential runs     | GPU cluster with fixed configs |

---

## Datasets

### Datasets Used in the Paper

| Dataset                   | Role                     | Task                             | Metric                 |
| ------------------------- | ------------------------ | -------------------------------- | ---------------------- |
| ShapeNet / ShapeNetRender | Pretraining              | Paired 2D--3D self-supervision   | Contrastive objectives |
| ModelNet40                | Downstream               | Linear object classification     | Accuracy               |
| ModelNet10                | Downstream               | Linear object classification     | Accuracy               |
| ScanObjectNN              | Downstream and ablations | Real-world object classification | Accuracy               |
| ShapeNetPart              | Downstream               | Part segmentation fine-tuning    | OA / mIoU              |
| S3DIS                     | Downstream               | Indoor semantic segmentation     | OA / mIoU              |

### Download Data for CrossPoint-Compatible Tasks

The upstream CrossPoint project supplies a data download script covering ShapeNetRender, ModelNet40, ScanObjectNN, and ShapeNetPart. To retain the original dataset protocol:

```bash
cd data
bash download_data.sh
cd ..
```

If using the unmodified upstream script:

```bash
cd data
source download_data.sh
cd ..
```

The upstream script retrieves and extracts archives corresponding to:

```text
shapenet.tar.gz
shapenet_render.tar.gz
scanobjectnn.tar.gz
modelnet40.tar.gz
shapenet_part.tar.gz
```

### Prepare S3DIS

S3DIS is not part of the minimal upstream CrossPoint data script. Download it under the official terms and prepare it using the repository preprocessing utility:

```bash
python tools/prepare_s3dis.py \
    --raw_root /path/to/raw/S3DIS \
    --output_root data/s3dis
```

### Expected Data Directory

```text
data/
├── shapenet/
├── shapenet_render/
├── modelnet40/
├── scanobjectnn/
├── shapenet_part/
└── s3dis/
```

### Protocol and Licensing Notes

MS-Point does not redistribute original datasets. Researchers must download datasets through their official or authorized distribution channels and comply with the corresponding usage terms.

| Dataset                   | Protocol Information to Report                           | Terms/License Note                                           |
| ------------------------- | -------------------------------------------------------- | ------------------------------------------------------------ |
| ShapeNet / ShapeNetRender | categories, render-view selection, pretraining split     | Follow official ShapeNet terms; use for permitted academic research purposes |
| ModelNet40 / ModelNet10   | number of points, sampling, voting, SVM settings         | Follow official benchmark usage terms                        |
| ScanObjectNN              | **exact variant** (`OBJ_ONLY`, `OBJ_BG`, or `PB_T50_RS`) | Follow the source/repository terms                           |
| ShapeNetPart              | sampling, category handling, metric definition           | Follow ShapeNet/ShapeNetPart terms                           |
| S3DIS                     | Area split, preprocessing, block sampling                | Follow official Stanford data terms                          |

> **Required before release:** Replace `<SCANOBJECTNN_VARIANT>` and any other protocol placeholders in the commands and configuration files with the exact setting used to produce the paper tables.
