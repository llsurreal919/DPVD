# DPVD: A Dataset and Benchmark for Dermoscopy-based Pediatric Vitiligo Diagnosis

> **Paper:** DPVD: A Dataset and Benchmark for Dermoscopy-based Pediatric Vitiligo Diagnosis  
> **Authors:** Yue Huang, Zhilin Chen, Shijuan Yu, Kai Zhou, Gang Wang, Xiaoyan Luo, Hua Wang, Yunpeng Xiao, Lei Luo  
> **Repository:** [github.com/llsurreal919/DPVD](https://github.com/llsurreal919/DPVD)

## Overview

DPVD is the first fine-grained, age-stratified dermoscopic image dataset for **pediatric vitiligo diagnosis** (ages 5–12). This repository provides a comprehensive deep learning benchmark covering the complete clinical workflow:

| Task | Model | Performance |
|------|-------|-------------|
| **Classification** (vitiligo vs. non-vitiligo) | Swin Transformer | Accuracy 98.9% |
| **Detection** (lesion localization) | YOLOv8 | Precision 90.7% |
| **Segmentation** (lesion boundary delineation) | U-Net (VGG/ResNet) | DICE 94% |

The benchmark forms a progressive diagnostic pipeline: **Classification → Localization → Quantification**.

## Repository Structure

```
DPVD/
├── unet/                  # U-Net segmentation model
│   ├── nets/              # Model architectures (VGG, ResNet backbones)
│   ├── tools/             # Dataset preparation (VOC annotation)
│   ├── utils/             # Training utilities, data loaders
│   ├── train.py           # Training script
│   ├── predict.py         # Inference script
│   └── test.py            # Evaluation script
├── swin-transformer/      # Swin Transformer classification model
│   ├── model.py           # Swin Transformer architecture
│   ├── train.py           # Training script
│   ├── predict.py         # Inference script
│   └── utils.py           # Utility functions
├── ultralytics/           # YOLOv8 detection model
│   ├── ultralytics/       # Core library
│   ├── examples/          # Usage examples
│   └── train.py           # Training script
├── data/                  # Dataset placeholder (see below)
└── README.md
```

## Dataset & Model Checkpoints

Due to file size constraints, the DPVD dataset and pre-trained model checkpoints are hosted separately.

📥 **Download from Baidu Netdisk:**
> Link: *(see data/README.md after download link is available)*

The download package includes:
- **DPVD Dataset:** Dermoscopic images with expert annotations (classification, detection, segmentation labels)
- **Model Checkpoints:** Pre-trained weights for all three tasks
- **Training Logs:** TensorBoard logs and training history

After downloading, place the `data/` folder at the repository root, and model checkpoints in their respective `model_data/` or `weights/` directories.

## Installation

```bash
# Clone the repository
git clone https://github.com/llsurreal919/DPVD.git
cd DPVD

# Install dependencies
pip install -r unet/requirements.txt
pip install -r ultralytics/requirements.txt
```

### Requirements
- Python >= 3.8
- PyTorch >= 1.10
- torchvision >= 0.11
- For full requirements, see each submodule's `requirements.txt`

## Usage

### 1. Classification (Swin Transformer)

```bash
cd swin-transformer
# Train
python train.py --data-path ../data --epochs 100
# Predict
python predict.py --weights path/to/weights.pth --image path/to/image.jpg
```

### 2. Detection (YOLOv8)

```bash
cd ultralytics
# Train
python train.py --data data.yaml --epochs 100
# Predict
yolo detect predict model=best.pt source=image.jpg
```

### 3. Segmentation (U-Net)

```bash
cd unet
# Train
python train.py
# Predict
python predict.py
```

## Clinical Validation

The benchmark has been validated at the **National Clinical Research Center for Child Health and Disorders**, achieving:
- Average expert rating > 4.0/5 across all tasks (Likert scale)
- ~91% overall diagnostic consistency rate with dermatology specialists

## Citation

If you use DPVD in your research, please cite:

```bibtex
@article{huang2026dpvd,
  title={DPVD: A Dataset and Benchmark for Dermoscopy-based Pediatric Vitiligo Diagnosis},
  author={Huang, Yue and Chen, Zhilin and Yu, Shijuan and Zhou, Kai and 
          Wang, Gang and Luo, Xiaoyan and Wang, Hua and Xiao, Yunpeng and Luo, Lei},
  journal={Electronics},
  year={2026},
  publisher={MDPI}
}
```

## License

This project is licensed under the terms specified in the LICENSE file. The ultralytics submodule follows its original AGPL-3.0 license.

## Acknowledgments

This work was conducted in collaboration with the National Clinical Research Center for Child Health and Disorders. The UNet codebase is adapted from [bubbliiiing/unet-pytorch](https://github.com/bubbliiiing/unet-pytorch), and the YOLOv8 implementation from [ultralytics/ultralytics](https://github.com/ultralytics/ultralytics).
