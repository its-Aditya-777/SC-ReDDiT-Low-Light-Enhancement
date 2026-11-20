from pathlib import Path
import cv2
import numpy as np
from utils import util

ROOT = Path("experiments") / "lolv2_test_real"
SAVE = Path("ensemble4_results")
SAVE.mkdir(exist_ok=True)

# Find completed official ReDDiT runs
runs = [
    p for p in ROOT.iterdir()
    if p.is_dir()
    and (p / "results" / "output").exists()
    and (p / "results" / "gt").exists()
]

runs = sorted(
    runs,
    key=lambda p: p.stat().st_mtime,
    reverse=True
)[:4]

if len(runs) != 4:
    raise RuntimeError(
        f"Need 4 completed ReDDiT runs, found {len(runs)}"
    )

print("\nUSING RUNS:")
for r in runs:
    print(" ", r.name)

output_dirs = [
    r / "results" / "output"
    for r in runs
]

gt_dir = runs[0] / "results" / "gt"

files = list(output_dirs[0].glob("*_normal.png"))

files = sorted(
    files,
    key=lambda p: int(p.name.split("_")[0])
)

if len(files) != 100:
    raise RuntimeError(
        f"Expected 100 images, found {len(files)}"
    )


# --------------------------------------------------
# First score each saved individual run ourselves
# --------------------------------------------------

print("\nINDIVIDUAL RUNS:")

for run_index, out_dir in enumerate(output_dirs, 1):

    psnrs = []
    ssims = []

    for f in files:

        pred = cv2.imread(
            str(out_dir / f.name),
            cv2.IMREAD_COLOR
        )

        idx = f.name.split("_")[0]

        gt = cv2.imread(
            str(gt_dir / f"{idx}_gt.png"),
            cv2.IMREAD_COLOR
        )

        psnrs.append(
            util.calculate_psnr(pred, gt)
        )

        ssims.append(
            util.calculate_ssim(pred, gt)
        )

    print(
        f"Run {run_index}: "
        f"PSNR={np.mean(psnrs):.4f} "
        f"SSIM={np.mean(ssims):.4f}"
    )


# --------------------------------------------------
# Four-seed consensus
# --------------------------------------------------

ensemble_psnr = []
ensemble_ssim = []

per_image = []

for f in files:

    idx = f.name.split("_")[0]

    preds = []

    for out_dir in output_dirs:

        img = cv2.imread(
            str(out_dir / f.name),
            cv2.IMREAD_COLOR
        )

        if img is None:
            raise RuntimeError(
                f"Missing {f.name} in {out_dir}"
            )

        preds.append(
            img.astype(np.float32)
        )

    # Mean consensus of four stochastic samples
    ensemble = np.mean(
        np.stack(preds, axis=0),
        axis=0
    )

    ensemble = np.clip(
        np.rint(ensemble),
        0,
        255
    ).astype(np.uint8)

    gt = cv2.imread(
        str(gt_dir / f"{idx}_gt.png"),
        cv2.IMREAD_COLOR
    )

    p = util.calculate_psnr(
        ensemble,
        gt
    )

    s = util.calculate_ssim(
        ensemble,
        gt
    )

    ensemble_psnr.append(p)
    ensemble_ssim.append(s)

    per_image.append((idx, p, s))

    cv2.imwrite(
        str(SAVE / f"{idx}_ensemble.png"),
        ensemble
    )


print("\n======================================")
print("ReDDiT 4-SEED CONSENSUS")
print("======================================")

print(
    f"Baseline target : 31.2450 dB"
)

print(
    f"Ensemble PSNR   : "
    f"{np.mean(ensemble_psnr):.4f} dB"
)

print(
    f"Ensemble SSIM   : "
    f"{np.mean(ensemble_ssim):.4f}"
)

print(
    f"Gain vs 31.245  : "
    f"{np.mean(ensemble_psnr)-31.2450:+.4f} dB"
)

print("======================================")
