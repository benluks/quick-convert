"""Run short training trials in isolated processes."""

import argparse
import json
import math
import shutil
import subprocess
from pathlib import Path


def run_trial(command, directory):
    directory.mkdir(parents=True, exist_ok=False)
    log = directory / "trial.log"
    with log.open("w") as stream:
        process = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=False)
    report = directory / "throughput.json"
    if process.returncode:
        message = log.read_text(errors="replace").lower()
        status = "oom" if "cuda out of memory" in message or "cuda error: out of memory" in message else "failed"
        return {"status": status, "returncode": process.returncode}
    if not report.exists():
        return {"status": "failed", "reason": "missing throughput report"}
    return {"status": "completed", "metrics": json.loads(report.read_text())}


def calibrate(output, durations, budgets, overrides, *, warmup=10, measured=20, run_trial_fn=run_trial):
    if any(not math.isfinite(x) or x <= 0 for x in [*durations, *budgets]):
        raise ValueError("Durations and budgets must be finite and positive.")
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    results = []
    for duration in durations:
        for budget in budgets:
            size = int(budget / duration)
            if size < 1:
                continue
            name = f"{duration:g}s_b{size}"
            directory = output / name
            if directory.exists():
                continue
            command = [
                shutil.which("quick-convert") or "quick-convert",
                "benchmark_training_librispeech",
                *overrides,
                "sampling@trainer.train_dataloader_kwargs=random",
                f"++train_dataset.segmentation={{_target_:quick_convert.data.segmentation.FixedDurationSegmentation,"
                f"duration:{duration},drop_remainder:true,duration_cache:outputs/cache/librispeech_durations.json}}",
                f"trainer.train_dataloader_kwargs.batch_size={size}",
                f"++trainer.trainer_kwargs.max_steps={warmup + measured}",
                "trainer.trainer_kwargs.max_epochs=1",
                "trainer.trainer_kwargs.enable_progress_bar=true",
                f"trainer.trainer_kwargs.callbacks.0.warmup_batches={warmup}",
                f"trainer.trainer_kwargs.callbacks.0.output={directory / 'throughput.json'}",
                f"exp_name=calibration_{output.name}_{name}",
            ]
            print(f"Trial {name}: {size * duration:g} audio seconds per batch", flush=True)
            result = run_trial_fn(command, directory)
            results.append(
                {
                    "duration": duration,
                    "batch_size": size,
                    "linear_cost": size * duration,
                    "quadratic_cost": size * duration**2,
                    "command": command,
                    **result,
                }
            )
            temporary = output / "results.tmp"
            temporary.write_text(json.dumps(results, indent=2) + "\n")
            temporary.replace(output / "results.json")
            print(f"{name}: {result['status']}", flush=True)
            if result["status"] == "failed":
                raise RuntimeError(f"Trial failed; inspect {directory / 'trial.log'}")
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--durations", nargs="+", type=float, default=[1.5, 3, 6, 12])
    parser.add_argument("--budgets", nargs="+", type=float, default=[192, 384, 768])
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--measured", type=int, default=20)
    args, overrides = parser.parse_known_args()
    if any(x <= 0 for x in args.durations + args.budgets) or args.warmup < 0 or args.measured <= 0:
        parser.error("Durations, budgets and measured steps must be positive; warmup must be nonnegative.")
    calibrate(args.output, args.durations, args.budgets, overrides, warmup=args.warmup, measured=args.measured)


if __name__ == "__main__":
    main()
