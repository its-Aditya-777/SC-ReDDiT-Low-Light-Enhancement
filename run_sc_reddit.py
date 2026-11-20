from pathlib import Path
import json
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parent
CONFIG = ROOT / "config" / "lolv2_real_test.json"
DATA_CONFIG = ROOT / "config" / "lolv2_real.yml"
CHECKPOINT = ROOT / "checkpoint" / "lolv2_real_4step_gen.pth"
DATA_ROOT = ROOT / "dataset" / "LOL-v2" / "Real_captured" / "Test"
EXPERIMENT_ROOT = ROOT / "experiments" / "lolv2_test_real"

SEEDS = [5660, 459, 6429, 2705]


def require_file(path):
    if not path.is_file():
        raise SystemExit(f"Required file missing: {path}")


def count_pngs(path):
    return len(list(path.glob("*.png")))


def main():
    for path in [CONFIG, DATA_CONFIG, CHECKPOINT]:
        require_file(path)

    low_dir = DATA_ROOT / "Low"
    normal_dir = DATA_ROOT / "Normal"

    if not low_dir.is_dir() or not normal_dir.is_dir():
        raise SystemExit(
            "LOLv2-Real test folders are missing. Expected:\n"
            f"{low_dir}\n{normal_dir}"
        )

    if count_pngs(low_dir) != 100 or count_pngs(normal_dir) != 100:
        raise SystemExit(
            "Expected exactly 100 PNG images in each LOLv2-Real "
            "test folder."
        )

    try:
        import torch
    except ImportError:
        raise SystemExit("PyTorch is missing. Install PyTorch first.")

    if not torch.cuda.is_available():
        raise SystemExit(
            "An NVIDIA CUDA GPU is required by the current ReDDiT "
            "inference implementation."
        )

    print("SC-ReDDiT: four-seed reproduction")
    print("Seeds:", SEEDS)
    print("Checkpoint:", CHECKPOINT)

    for seed in SEEDS:
        print("\n" + "=" * 65)
        print(f"Running ReDDiT-4 inference with seed {seed}")
        print("=" * 65)

        config = json.loads(CONFIG.read_text(encoding="utf-8"))
        config["name"] = "lolv2_test_real"
        config["phase"] = "test"
        config["seed"] = seed
        config["path"]["resume_state"] = str(CHECKPOINT.resolve())

        temp_path = None

        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                suffix=".json",
                prefix=f"sc_reddit_seed_{seed}_",
                dir=ROOT,
                delete=False,
                encoding="utf-8"
            ) as temp:
                json.dump(config, temp, indent=4)
                temp_path = Path(temp.name)

            command = [
                sys.executable,
                str(ROOT / "test.py"),
                "--dataset", str(DATA_CONFIG),
                "--config", str(temp_path),
                "--phase", "val",
                "--gpu_ids", "0",
                "--w_str", "0.9",
                "--w_snr", "0.2",
                "--w_gt", "0.2",
                "--w_lpips", "0.1"
            ]

            subprocess.run(command, cwd=ROOT, check=True)

        finally:
            if temp_path is not None and temp_path.exists():
                temp_path.unlink()

        candidates = sorted(
            EXPERIMENT_ROOT.glob(f"*seed_{seed}_*"),
            key=lambda p: p.stat().st_mtime,
            reverse=True
        )

        if not candidates:
            raise SystemExit(
                f"Could not locate the experiment directory for seed {seed}."
            )

        raw_dir = candidates[0] / "results" / "raw"

        if count_pngs(raw_dir) != 100:
            raise SystemExit(
                f"Seed {seed} did not produce all 100 raw predictions: "
                f"{raw_dir}"
            )

        print(f"Seed {seed}: all 100 raw predictions verified.")

    print("\nAll four inference runs completed.")
    print("Starting SC-ReDDiT consensus and ablation evaluation.")

    subprocess.run(
        [sys.executable, str(ROOT / "evaluate_sc_reddit_final.py")],
        cwd=ROOT,
        check=True
    )

    print("\nSC-ReDDiT reproduction completed successfully.")
    print("Results:", ROOT / "SC_ReDDiT_Final")


if __name__ == "__main__":
    main()