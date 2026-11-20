from pathlib import Path
import csv
import cv2
import numpy as np
import torch
import lpips

from utils import util


ROOT = Path(".")
FINAL_ROOT = ROOT / "SC_ReDDiT_Final"

FINAL_DIR = FINAL_ROOT / "consensus4"
ABLATION_CSV = FINAL_ROOT / "ablation_1_2_3_4.csv"
PER_IMAGE_CSV = FINAL_ROOT / "per_image_consensus4.csv"

# Ground truths from the successful official ReDDiT run
GT_DIR = (
    ROOT
    / "experiments"
    / "lolv2_test_real"
    / "lolv2_test_real_numstep_4_w_snr_0.2_w_gt_0.2_w_lpips_0.1_261026_035359"
    / "results"
    / "gt"
)

SEEDS = [5660, 459, 6429, 2705]

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# CHECK FILES
# ============================================================

for p in [
    FINAL_DIR,
    ABLATION_CSV,
    PER_IMAGE_CSV,
    GT_DIR
]:
    if not p.exists():
        raise RuntimeError(
            f"Required path missing: {p}"
        )


# ============================================================
# LPIPS
# ============================================================

print("\nLoading LPIPS VGG model...")

lpips_fn = lpips.LPIPS(
    net="vgg"
).to(DEVICE)

lpips_fn.eval()


@torch.no_grad()
def calc_lpips(pred, gt):

    p = torch.from_numpy(
        pred.astype(np.float32)
    ).permute(
        2, 0, 1
    ).unsqueeze(0).to(DEVICE)

    g = torch.from_numpy(
        gt.astype(np.float32)
    ).permute(
        2, 0, 1
    ).unsqueeze(0).to(DEVICE)

    return float(
        lpips_fn(p, g).item()
    )


# ============================================================
# RECOMPUTE FINAL SC-ReDDiT METRICS
# ============================================================

psnr_values = []
ssim_values = []
lpips_values = []

for idx in range(1, 101):

    pred_path = (
        FINAL_DIR
        / f"{idx}_SC_ReDDiT.png"
    )

    gt_path = (
        GT_DIR
        / f"{idx}_gt.png"
    )

    pred = cv2.imread(
        str(pred_path),
        cv2.IMREAD_COLOR
    )

    gt = cv2.imread(
        str(gt_path),
        cv2.IMREAD_COLOR
    )

    if pred is None:
        raise RuntimeError(
            f"Missing prediction: {pred_path}"
        )

    if gt is None:
        raise RuntimeError(
            f"Missing GT: {gt_path}"
        )

    psnr_values.append(
        util.calculate_psnr(
            pred,
            gt
        )
    )

    ssim_values.append(
        util.calculate_ssim(
            pred,
            gt
        )
    )

    lpips_values.append(
        calc_lpips(
            pred,
            gt
        )
    )


final_psnr = float(
    np.mean(psnr_values)
)

final_ssim = float(
    np.mean(ssim_values)
)

final_lpips = float(
    np.mean(lpips_values)
)


# ============================================================
# READ ABLATION
# ============================================================

with open(
    ABLATION_CSV,
    newline=""
) as f:

    ablation = list(
        csv.DictReader(f)
    )


# ============================================================
# READ PER-IMAGE RESULTS
# ============================================================

with open(
    PER_IMAGE_CSV,
    newline=""
) as f:

    per_image = list(
        csv.DictReader(f)
    )


wins = sum(
    float(r["gain_psnr"]) > 0
    for r in per_image
)

losses = sum(
    float(r["gain_psnr"]) < 0
    for r in per_image
)

ties = (
    len(per_image)
    - wins
    - losses
)


mean_gain = float(
    np.mean([
        float(r["gain_psnr"])
        for r in per_image
    ])
)


raw_psnr = float(
    np.mean([
        float(
            r["raw_consensus4_psnr"]
        )
        for r in per_image
    ])
)

raw_ssim = float(
    np.mean([
        float(
            r["raw_consensus4_ssim"]
        )
        for r in per_image
    ])
)


# ============================================================
# DISPLAY
# ============================================================

print("\n")
print("=" * 62)
print("               SC-ReDDiT FINAL VERIFICATION")
print("=" * 62)

print("\nMethod:")
print(
    "SC-ReDDiT — Stochastic Consensus ReDDiT "
    "(4 trajectories)"
)

print("\nDataset:")
print(
    "LOLv2-Real Test Set — 100 paired images"
)

print("\nIndependent trajectory seeds:")

for i, seed in enumerate(
    SEEDS,
    1
):
    print(
        f"  Trajectory {i}: {seed}"
    )


print("\n" + "=" * 62)
print("FINAL QUANTITATIVE RESULT")
print("=" * 62)

print(
    f"PSNR   : {final_psnr:.4f} dB"
)

print(
    f"SSIM   : {final_ssim:.4f}"
)

print(
    f"LPIPS  : {final_lpips:.4f}"
)

print("\nOfficial ReDDiT benchmark protocol.")


print("\n" + "=" * 62)
print("RAW / NO-GT-BRIGHTNESS-ADJUSTMENT RESULT")
print("=" * 62)

print(
    f"Raw PSNR : {raw_psnr:.4f} dB"
)

print(
    f"Raw SSIM : {raw_ssim:.4f}"
)


print("\n" + "=" * 62)
print("STOCHASTIC CONSENSUS ABLATION")
print("=" * 62)

print(
    f"{'Trajectories':>12}"
    f"{'PSNR':>13}"
    f"{'Std':>11}"
    f"{'SSIM':>12}"
    f"{'LPIPS':>12}"
)

print("-" * 62)

for row in ablation:

    print(
        f"{int(row['N']):>12}"
        f"{float(row['PSNR_mean']):>13.4f}"
        f"{float(row['PSNR_std']):>11.4f}"
        f"{float(row['SSIM_mean']):>12.4f}"
        f"{float(row['LPIPS_mean']):>12.4f}"
    )


print("\n" + "=" * 62)
print("PER-IMAGE CONSISTENCY")
print("=" * 62)

print(
    f"Images improved : {wins}/100"
)

print(
    f"Images decreased: {losses}/100"
)

print(
    f"Ties            : {ties}/100"
)

print(
    f"Mean PSNR gain from consensus: "
    f"{mean_gain:+.4f} dB"
)


print("\n" + "=" * 62)
print("FINAL PROJECT RESULT")
print("=" * 62)

print(
    f"SC-ReDDiT PSNR  : {final_psnr:.4f} dB"
)

print(
    f"SC-ReDDiT SSIM  : {final_ssim:.4f}"
)

print(
    f"SC-ReDDiT LPIPS : {final_lpips:.4f}"
)

print(
    f"Consensus wins  : {wins}/100 images"
)

print("\nRESULT VERIFIED FROM SAVED OUTPUT IMAGES.")
print("=" * 62)
