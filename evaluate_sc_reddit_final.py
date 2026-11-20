from pathlib import Path
from itertools import combinations
import re
import csv
import cv2
import numpy as np
import torch
import lpips

from utils import util


ROOT = Path("experiments") / "lolv2_test_real"
OUT = Path("SC_ReDDiT_Final")
OUT.mkdir(exist_ok=True)

FINAL_IMG_DIR = OUT / "consensus4"
FINAL_IMG_DIR.mkdir(exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ------------------------------------------------------------
# Find latest 4 completed RAW runs
# ------------------------------------------------------------

all_runs = []

for p in ROOT.iterdir():

    if not p.is_dir():
        continue

    raw_dir = p / "results" / "raw"
    gt_dir  = p / "results" / "gt"

    if not raw_dir.exists() or not gt_dir.exists():
        continue

    raw_files = list(raw_dir.glob("*_normal_raw.png"))

    if len(raw_files) == 100:
        all_runs.append(p)


runs = sorted(
    all_runs,
    key=lambda p: p.stat().st_mtime,
    reverse=True
)[:4]

if len(runs) != 4:
    raise RuntimeError(
        f"Need 4 completed raw runs; found {len(runs)}"
    )


print("\n==========================================")
print("SC-ReDDiT FINAL EVALUATION")
print("==========================================")

print("\nUSING THESE 4 RUNS:")

for i, r in enumerate(runs, 1):
    print(f"Run {i}: {r.name}")


# ------------------------------------------------------------
# Recover random seeds from logs if possible
# ------------------------------------------------------------

def find_seed(run):

    log_dir = run / "logs"

    if not log_dir.exists():
        return None

    for f in log_dir.glob("*.log"):

        try:
            txt = f.read_text(
                encoding="utf-8",
                errors="ignore"
            )
        except Exception:
            continue

        m = re.search(
            r"Seed:\s*(\d+)",
            txt
        )

        if m:
            return int(m.group(1))

    return None


seeds = [
    find_seed(r)
    for r in runs
]

print("\nSEEDS:")

for i, s in enumerate(seeds, 1):
    print(
        f"Run {i}: "
        f"{s if s is not None else 'not found'}"
    )


# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------

raw_dirs = [
    r / "results" / "raw"
    for r in runs
]

gt_dir = (
    runs[0]
    / "results"
    / "gt"
)


# ------------------------------------------------------------
# File indices
# ------------------------------------------------------------

files = sorted(
    raw_dirs[0].glob("*_normal_raw.png"),
    key=lambda p: int(
        p.name.split("_")[0]
    )
)

if len(files) != 100:
    raise RuntimeError(
        f"Expected 100 raw images, got {len(files)}"
    )


# ------------------------------------------------------------
# Official ReDDiT GT-mean adjustment
# EXACTLY matching test.py logic
# ------------------------------------------------------------

def official_adjust(pred, gt):

    pred01 = (
        pred.astype(np.float32)
        / 255.0
    )

    gt01 = (
        gt.astype(np.float32)
        / 255.0
    )

    mean_gray_out = cv2.cvtColor(
        pred01,
        cv2.COLOR_BGR2GRAY
    ).mean()

    mean_gray_gt = cv2.cvtColor(
        gt01,
        cv2.COLOR_BGR2GRAY
    ).mean()

    scale = (
        mean_gray_gt
        /
        (mean_gray_out + 1e-12)
    )

    adjusted = np.clip(
        pred01 * scale,
        0.0,
        1.0
    )

    # Match official ReDDiT:
    # truncation via astype(uint8)
    adjusted = (
        adjusted * 255
    ).astype(np.uint8)

    return adjusted


# ------------------------------------------------------------
# LPIPS
#
# Match ReDDiT test.py behavior:
# uint8-range float tensors, not extra normalization.
# ------------------------------------------------------------

print("\nLoading LPIPS VGG...")

lpips_fn = lpips.LPIPS(
    net="vgg"
).to(DEVICE)

lpips_fn.eval()


@torch.no_grad()
def calc_lpips(pred, gt):

    p = torch.from_numpy(
        pred.astype(np.float32)
    ).permute(2, 0, 1).to(DEVICE)

    g = torch.from_numpy(
        gt.astype(np.float32)
    ).permute(2, 0, 1).to(DEVICE)

    return float(
        lpips_fn(p, g).item()
    )


# ------------------------------------------------------------
# Evaluate ONE ensemble combination
# ------------------------------------------------------------

def evaluate_combo(combo, save_final=False):

    psnrs = []
    ssims = []
    lpips_scores = []

    raw_psnrs = []
    raw_ssims = []

    details = []

    for f in files:

        idx = int(
            f.name.split("_")[0]
        )

        preds = []

        for run_idx in combo:

            img = cv2.imread(
                str(
                    raw_dirs[run_idx]
                    /
                    f"{idx}_normal_raw.png"
                ),
                cv2.IMREAD_COLOR
            )

            if img is None:
                raise RuntimeError(
                    f"Missing image {idx} "
                    f"in run {run_idx}"
                )

            preds.append(
                img.astype(np.float32)
            )

        # ============================================
        # RAW STOCHASTIC CONSENSUS
        # ============================================

        consensus_float = np.mean(
            np.stack(preds, axis=0),
            axis=0
        )

        consensus_raw = np.clip(
            consensus_float,
            0,
            255
        ).astype(np.uint8)


        gt = cv2.imread(
            str(
                gt_dir
                /
                f"{idx}_gt.png"
            ),
            cv2.IMREAD_COLOR
        )

        if gt is None:
            raise RuntimeError(
                f"Missing GT {idx}"
            )


        # ============================================
        # RAW / deployable metrics
        # ============================================

        raw_p = util.calculate_psnr(
            consensus_raw,
            gt
        )

        raw_s = util.calculate_ssim(
            consensus_raw,
            gt
        )

        raw_psnrs.append(raw_p)
        raw_ssims.append(raw_s)


        # ============================================
        # OFFICIAL ReDDiT benchmark protocol:
        # apply GT-mean adjustment ONCE
        # ============================================

        adjusted = official_adjust(
            consensus_float,
            gt
        )

        p = util.calculate_psnr(
            adjusted,
            gt
        )

        s = util.calculate_ssim(
            adjusted,
            gt
        )

        l = calc_lpips(
            adjusted,
            gt
        )

        psnrs.append(p)
        ssims.append(s)
        lpips_scores.append(l)

        details.append({
            "image": idx,
            "psnr": p,
            "ssim": s,
            "lpips": l,
            "raw_psnr": raw_p,
            "raw_ssim": raw_s,
        })


        if save_final:

            cv2.imwrite(
                str(
                    FINAL_IMG_DIR
                    /
                    f"{idx}_SC_ReDDiT.png"
                ),
                adjusted
            )


    return {
        "psnr": float(np.mean(psnrs)),
        "ssim": float(np.mean(ssims)),
        "lpips": float(np.mean(lpips_scores)),
        "raw_psnr": float(np.mean(raw_psnrs)),
        "raw_ssim": float(np.mean(raw_ssims)),
        "details": details
    }


# ------------------------------------------------------------
# FULL 1 / 2 / 3 / 4 SAMPLE ABLATION
# ------------------------------------------------------------

summary_rows = []

results_by_n = {}

for n in [1, 2, 3, 4]:

    combos = list(
        combinations(
            range(4),
            n
        )
    )

    combo_results = []

    print(
        f"\n------------------------------------------"
    )

    print(
        f"Evaluating N={n} "
        f"({len(combos)} combinations)"
    )

    print(
        f"------------------------------------------"
    )

    for j, combo in enumerate(
        combos,
        1
    ):

        save_final = (
            n == 4
        )

        result = evaluate_combo(
            combo,
            save_final=save_final
        )

        combo_results.append(
            result
        )

        combo_string = "-".join(
            str(x + 1)
            for x in combo
        )

        print(
            f"N={n} combo={combo_string:7s} | "
            f"PSNR={result['psnr']:.4f} | "
            f"SSIM={result['ssim']:.4f} | "
            f"LPIPS={result['lpips']:.4f}"
        )


    psnr_values = np.array([
        r["psnr"]
        for r in combo_results
    ])

    ssim_values = np.array([
        r["ssim"]
        for r in combo_results
    ])

    lpips_values = np.array([
        r["lpips"]
        for r in combo_results
    ])

    raw_psnr_values = np.array([
        r["raw_psnr"]
        for r in combo_results
    ])

    raw_ssim_values = np.array([
        r["raw_ssim"]
        for r in combo_results
    ])


    aggregate = {
        "N": n,

        "PSNR_mean":
            float(psnr_values.mean()),

        "PSNR_std":
            float(psnr_values.std()),

        "SSIM_mean":
            float(ssim_values.mean()),

        "SSIM_std":
            float(ssim_values.std()),

        "LPIPS_mean":
            float(lpips_values.mean()),

        "LPIPS_std":
            float(lpips_values.std()),

        "RAW_PSNR_mean":
            float(raw_psnr_values.mean()),

        "RAW_SSIM_mean":
            float(raw_ssim_values.mean()),
    }

    summary_rows.append(
        aggregate
    )

    results_by_n[n] = {
        "aggregate": aggregate,
        "combos": combo_results
    }


# ------------------------------------------------------------
# FINAL N=4 RESULT
# ------------------------------------------------------------

final = results_by_n[4]["combos"][0]


# ------------------------------------------------------------
# Compare final consensus against mean single-run performance
# ------------------------------------------------------------

single = results_by_n[1]["aggregate"]

gain_vs_single_psnr = (
    final["psnr"]
    -
    single["PSNR_mean"]
)

gain_vs_single_ssim = (
    final["ssim"]
    -
    single["SSIM_mean"]
)

gain_vs_single_lpips = (
    final["lpips"]
    -
    single["LPIPS_mean"]
)


# ------------------------------------------------------------
# Per-image consensus vs mean of individual runs
# ------------------------------------------------------------

single_details = [
    r["details"]
    for r in results_by_n[1]["combos"]
]

final_details = final["details"]

per_image_rows = []

wins = 0
losses = 0
ties = 0

for i in range(100):

    idx = final_details[i]["image"]

    single_psnr_mean = float(
        np.mean([
            d[i]["psnr"]
            for d in single_details
        ])
    )

    final_psnr = (
        final_details[i]["psnr"]
    )

    gain = (
        final_psnr
        -
        single_psnr_mean
    )

    if gain > 1e-9:
        wins += 1
    elif gain < -1e-9:
        losses += 1
    else:
        ties += 1

    per_image_rows.append({
        "image": idx,
        "single_psnr_mean": single_psnr_mean,
        "consensus4_psnr": final_psnr,
        "gain_psnr": gain,
        "consensus4_ssim":
            final_details[i]["ssim"],
        "consensus4_lpips":
            final_details[i]["lpips"],
        "raw_consensus4_psnr":
            final_details[i]["raw_psnr"],
        "raw_consensus4_ssim":
            final_details[i]["raw_ssim"],
    })


# ------------------------------------------------------------
# Save ablation CSV
# ------------------------------------------------------------

ablation_csv = (
    OUT
    /
    "ablation_1_2_3_4.csv"
)

with open(
    ablation_csv,
    "w",
    newline=""
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=list(
            summary_rows[0].keys()
        )
    )

    writer.writeheader()
    writer.writerows(
        summary_rows
    )


# ------------------------------------------------------------
# Save per-image CSV
# ------------------------------------------------------------

per_image_csv = (
    OUT
    /
    "per_image_consensus4.csv"
)

with open(
    per_image_csv,
    "w",
    newline=""
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=list(
            per_image_rows[0].keys()
        )
    )

    writer.writeheader()
    writer.writerows(
        per_image_rows
    )


# ------------------------------------------------------------
# Best/worst examples
# ------------------------------------------------------------

sorted_rows = sorted(
    per_image_rows,
    key=lambda x:
        x["gain_psnr"],
    reverse=True
)


# ------------------------------------------------------------
# Save summary
# ------------------------------------------------------------

summary_txt = OUT / "FINAL_RESULTS.txt"

with open(
    summary_txt,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "SC-ReDDiT FINAL RESULTS\n"
    )

    f.write(
        "========================\n\n"
    )

    for i, r in enumerate(
        runs,
        1
    ):

        f.write(
            f"Run {i}: {r.name}\n"
        )

        f.write(
            f"Seed {i}: {seeds[i-1]}\n"
        )

    f.write("\n")

    for row in summary_rows:

        f.write(
            f"N={row['N']} | "
            f"PSNR={row['PSNR_mean']:.4f}"
            f"±{row['PSNR_std']:.4f} | "
            f"SSIM={row['SSIM_mean']:.4f}"
            f"±{row['SSIM_std']:.4f} | "
            f"LPIPS={row['LPIPS_mean']:.4f}"
            f"±{row['LPIPS_std']:.4f}\n"
        )

    f.write("\n")

    f.write(
        f"FINAL N=4 PSNR: "
        f"{final['psnr']:.4f}\n"
    )

    f.write(
        f"FINAL N=4 SSIM: "
        f"{final['ssim']:.4f}\n"
    )

    f.write(
        f"FINAL N=4 LPIPS: "
        f"{final['lpips']:.4f}\n"
    )

    f.write(
        f"RAW N=4 PSNR: "
        f"{final['raw_psnr']:.4f}\n"
    )

    f.write(
        f"RAW N=4 SSIM: "
        f"{final['raw_ssim']:.4f}\n"
    )

    f.write(
        f"Gain vs average single PSNR: "
        f"{gain_vs_single_psnr:+.4f}\n"
    )

    f.write(
        f"Gain vs average single SSIM: "
        f"{gain_vs_single_ssim:+.4f}\n"
    )

    f.write(
        f"LPIPS change vs average single: "
        f"{gain_vs_single_lpips:+.4f}\n"
    )

    f.write(
        f"Per-image PSNR wins/losses/ties: "
        f"{wins}/{losses}/{ties}\n"
    )


# ------------------------------------------------------------
# CONSOLE SUMMARY
# ------------------------------------------------------------

print("\n\n==========================================")
print("FINAL ABLATION")
print("==========================================")

print(
    " N |      PSNR       |      SSIM       |      LPIPS"
)

print(
    "---+-----------------+-----------------+----------------"
)

for row in summary_rows:

    print(
        f" {row['N']} | "
        f"{row['PSNR_mean']:.4f} "
        f"± {row['PSNR_std']:.4f} | "
        f"{row['SSIM_mean']:.4f} "
        f"± {row['SSIM_std']:.4f} | "
        f"{row['LPIPS_mean']:.4f} "
        f"± {row['LPIPS_std']:.4f}"
    )


print("\n==========================================")
print("SC-ReDDiT FINAL 4-SAMPLE RESULT")
print("==========================================")

print(
    f"PSNR  : {final['psnr']:.4f} dB"
)

print(
    f"SSIM  : {final['ssim']:.4f}"
)

print(
    f"LPIPS : {final['lpips']:.4f}"
)

print(
    f"\nRaw/no-GT PSNR : "
    f"{final['raw_psnr']:.4f} dB"
)

print(
    f"Raw/no-GT SSIM : "
    f"{final['raw_ssim']:.4f}"
)

print(
    f"\nGain vs average single run:"
)

print(
    f"PSNR  : "
    f"{gain_vs_single_psnr:+.4f} dB"
)

print(
    f"SSIM  : "
    f"{gain_vs_single_ssim:+.4f}"
)

print(
    f"LPIPS : "
    f"{gain_vs_single_lpips:+.4f}"
)

print(
    f"\nPer-image PSNR:"
)

print(
    f"Wins   : {wins}/100"
)

print(
    f"Losses : {losses}/100"
)

print(
    f"Ties   : {ties}/100"
)


print("\nTOP 10 GAINS:")

for r in sorted_rows[:10]:

    print(
        f"{r['image']:3d} | "
        f"{r['single_psnr_mean']:.2f} -> "
        f"{r['consensus4_psnr']:.2f} | "
        f"{r['gain_psnr']:+.2f} dB"
    )


print("\nWORST 10 CHANGES:")

for r in sorted_rows[-10:]:

    print(
        f"{r['image']:3d} | "
        f"{r['single_psnr_mean']:.2f} -> "
        f"{r['consensus4_psnr']:.2f} | "
        f"{r['gain_psnr']:+.2f} dB"
    )


print("\nSaved:")
print(ablation_csv)
print(per_image_csv)
print(summary_txt)
print(FINAL_IMG_DIR)

print("==========================================")
