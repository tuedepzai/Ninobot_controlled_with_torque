"""Run deterministic Gazebo episodes and save comparable path/terrain metrics."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
from pathlib import Path
import sys

from ament_index_python.packages import get_package_share_directory
import numpy as np

from nino_rl.core import load_config


def arguments() -> argparse.Namespace:
    share = Path(get_package_share_directory("nino_rl"))
    parser = argparse.ArgumentParser(description="Evaluate a trained Nino PPO policy")
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--config", type=Path, default=share / "config" / "ppo.yaml")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--output", type=Path, default=Path("rl_runs/evaluation"))
    parser.add_argument(
        "--randomized", action="store_true", help="Enable domain randomization during testing"
    )
    return parser.parse_args(sys.argv[1:])


def main() -> None:
    args = arguments()
    if args.episodes <= 0:
        raise SystemExit("--episodes phải lớn hơn 0")
    try:
        from stable_baselines3 import PPO
    except ImportError as error:
        raise SystemExit("Thiếu stable-baselines3; xem README_VI.md") from error
    from nino_rl.ros_env import NinoGazeboEnv

    config = load_config(args.config)
    config["curriculum"]["enabled"] = False
    config["domain_randomization"]["enabled"] = bool(args.randomized)
    env = NinoGazeboEnv(config, total_training_steps=1)
    rows = []
    try:
        model = PPO.load(args.model, device=str(config.get("device", "cuda")))
        for episode in range(args.episodes):
            observation, _ = env.reset(seed=int(config["seed"]) + episode)
            done = False
            info = {}
            while not done:
                action, _ = model.predict(observation, deterministic=True)
                observation, _, terminated, truncated, info = env.step(action)
                done = terminated or truncated
            metrics = dict(info["episode_metrics"])
            metrics["episode"] = episode + 1
            rows.append(metrics)
            print(
                f"Episode {episode + 1}: {metrics['termination']}, "
                f"t={metrics['time_seconds']:.1f}s, "
                f"|e_y|={metrics['mean_abs_lateral_error_m']:.3f}m"
            )
    finally:
        env.close()

    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    csv_path = output / f"evaluation-{stamp}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "episodes": len(rows),
        "success_rate": float(np.mean([row["success"] for row in rows])),
        "mean_time_seconds": float(np.mean([row["time_seconds"] for row in rows])),
        "mean_abs_lateral_error_m": float(
            np.mean([row["mean_abs_lateral_error_m"] for row in rows])
        ),
        "mean_abs_roll_deg": float(np.mean([row["mean_abs_roll_deg"] for row in rows])),
        "mean_abs_pitch_deg": float(np.mean([row["mean_abs_pitch_deg"] for row in rows])),
    }
    json_path = output / f"summary-{stamp}.json"
    with json_path.open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Đã lưu: {csv_path}\nĐã lưu: {json_path}")


if __name__ == "__main__":
    main()
