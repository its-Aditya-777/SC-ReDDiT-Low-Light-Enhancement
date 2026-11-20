# SC-ReDDiT
## Inference-Time Stochastic Consensus for Diffusion-Based Low-Light Image Enhancement using ReDDiT-4

SC-ReDDiT is an inference-time consensus approach built on the ReDDiT-4 low-light image enhancement model. It generates four independent stochastic predictions from the same pretrained model and averages them pixel-wise. No additional model training is required.

**This repository contains the ReDDiT source implementation used by SC-ReDDiT**, including the model, diffusion, data loading, configuration, and utility code, together with the SC-ReDDiT evaluation scripts and recorded results. You do not need to clone a separate ReDDiT source-code repository.

> **External assets are still required:** the LOLv2-Real dataset and compatible pretrained ReDDiT-4 checkpoint are not included in Git because of their size and distribution restrictions.

---

## 1. Method

Four independent ReDDiT-4 inference runs are performed with the following random seeds:

| Run | Seed |
|---:|---:|
| 1 | 5660 |
| 2 | 459 |
| 3 | 6429 |
| 4 | 2705 |

For predictions \(I_1, I_2, I_3, I_4\), the consensus is:

\[
I_{SC} = \frac{1}{4}\sum_{k=1}^{4} I_k
\]

The four predictions are averaged pixel-wise, and the final image is clipped to the valid 8-bit range.

## 2. Experimental Results

Evaluation was performed on the 100 paired images in the LOLv2-Real test set.

### Sample-count ablation

| Samples | PSNR (dB) | SSIM | LPIPS |
|---:|---:|---:|---:|
| 1 | 31.2293 | 0.9036 | 0.0366 |
| 2 | 31.3621 | 0.9097 | 0.0352 |
| 3 | 31.4086 | 0.9118 | 0.0348 |
| 4 | **31.4315** | **0.9128** | **0.0346** |

The four-sample result is +0.2023 dB PSNR and +0.0092 SSIM relative to the mean single-sample result in this experiment.

The four-sample consensus achieves higher per-image PSNR than the mean of the individual samples on 96 of the 100 test images.

### Important evaluation qualification

The adjusted metrics above follow the brightness adjustment used by the original ReDDiT evaluation code: the predicted image is scaled using the mean brightness of its paired ground-truth image.

Because this operation uses ground-truth information, these adjusted metrics are **not a strictly inference-only or deployment-time evaluation**. They should be interpreted as results under that benchmark protocol, not as proof of a ground-truth-independent improvement.

For transparency, the four-sample consensus before ground-truth-dependent brightness adjustment achieved:

- **Raw PSNR:** 28.0527 dB
- **Raw SSIM:** 0.9054

The raw and adjusted metrics represent different evaluation settings and should not be compared as though they were calculated on identical predictions.

Recorded results and per-image data are available in `SC_ReDDiT_Final/`.

---

## 3. Repository Contents

- `model/`, `core/`, `data/`, `options/`, `utils/`: ReDDiT model and supporting implementation.
- `config/lolv2_real.yml`: LOLv2 dataset configuration.
- `config/lolv2_real_test.json`: ReDDiT-4 inference configuration.
- `test.py`: Inference script, modified to save raw predictions before ground-truth-dependent brightness adjustment.
- `evaluate_sc_reddit_final.py`: Four-run consensus and 1/2/3/4-sample evaluation.
- `eval_reddit_ensemble4.py`: Additional four-run consensus utility.
- `SC_ReDDiT_Final/`: Recorded result summary and CSV files.
- `ensemble4_results/`: Saved consensus result images.

---

## 4. Environment and Installation

The inference implementation requires an NVIDIA GPU and a CUDA-enabled PyTorch installation.

### Step 1: Clone this repository

```bash
git clone https://github.com/its-Aditya-777/SC-ReDDiT-Low-Light-Enhancement.git
cd SC-ReDDiT-Low-Light-Enhancement
```

### Step 2: Create a Python environment

Python 3.8 was specified by the original ReDDiT project.

```bash
conda create -n sc-reddit python=3.8
conda activate sc-reddit
```

### Step 3: Install PyTorch

Install a PyTorch and torchvision build compatible with your NVIDIA driver and CUDA environment.

Use the official PyTorch installation selector:

https://pytorch.org/get-started/locally/

### Step 4: Install dependencies

```bash
pip install -r requirements.txt
```

The first LPIPS evaluation may download the pretrained VGG weights used by the LPIPS package.

---

## 5. Dataset Preparation

Download LOLv2 from the dataset provider:

https://github.com/flyywh/CVPR-2020-Semi-Low-Light

For this experiment, place the **LOLv2-Real test split** in the following exact structure:

```text
dataset/
└── LOL-v2/
    └── Real_captured/
        └── Test/
            ├── Low/
            └── Normal/
```

The `Low` and `Normal` folders must contain the corresponding 100 test images with matching filenames.

The dataset is not included in this repository.

---

## 6. Pretrained Checkpoint

Download the ReDDiT pretrained checkpoint release:

https://drive.google.com/file/d/13_XM8nFxJc2IfUotC2_lJo9ATt0rcIyg/view?usp=sharing

The LOLv2-Real four-step checkpoint expected by the supplied configuration is:

```text
checkpoint/
└── lolv2_real_4step_gen.pth
```

The checkpoint must be compatible with `config/lolv2_real_test.json`.

Checkpoints are not committed to Git.

---

## 7. Reproducing the Four Inference Runs

Run all commands from the repository root.

The configuration file `config/lolv2_real_test.json` must contain a top-level seed entry. Before each run, set its value to the corresponding seed below. For example, add `"seed": 5660` alongside the `"name"` and `"phase"` entries.

For each seed, execute:

```bash
python test.py --dataset ./config/lolv2_real.yml --config config/lolv2_real_test.json --phase val --gpu_ids 0 --w_str 0.9 --w_snr 0.2 --w_gt 0.2 --w_lpips 0.1
```

Repeat the inference command once for each seed, in this order:

```text
5660
459
6429
2705
```

Each run should generate 100 raw predictions under its own timestamped experiment directory:

```text
experiments/lolv2_test_real/<run-name>/results/raw/
```

The raw files are named `*_normal_raw.png`.

Do not delete the four completed run directories before evaluation.

---

## 8. Generate the SC-ReDDiT Consensus and Metrics

After all four inference runs have completed successfully, execute:

```bash
python evaluate_sc_reddit_final.py
```

The evaluator locates the four most recently modified completed raw runs under `experiments/lolv2_test_real/`, creates the sample-count ablation, and writes the consensus images and metric CSV files under:

```text
SC_ReDDiT_Final/
```

**Important:** The current evaluator consumes four completed inference runs. It does not launch those four inference runs automatically. The inference stage and evaluation stage are therefore separate commands.

---

## 9. Citation and Acknowledgement

SC-ReDDiT builds on the ReDDiT method and implementation. Please cite the original paper when using the underlying model:

**Efficient Diffusion as Low Light Enhancer**, CVPR 2025.

Paper: https://arxiv.org/abs/2410.12346

Project page: https://mqleet.github.io/ReDDiT_Project/

```bibtex
@InProceedings{Lan_2025_CVPR,
    author    = {Lan, Guanzhou and Ma, Qianli and Yang, Yuqi and Wang, Zhigang and Wang, Dong and Li, Xuelong and Zhao, Bin},
    title     = {Efficient Diffusion as Low Light Enhancer},
    booktitle = {Proceedings of the Computer Vision and Pattern Recognition Conference (CVPR)},
    month     = {June},
    year      = {2025},
    pages     = {21277-21286}
}
```

The original ReDDiT work acknowledges SR3:

https://github.com/Janspiry/Image-Super-Resolution-via-Iterative-Refinement

Please also review and comply with the original project's licensing and attribution terms when redistributing or using its code.
