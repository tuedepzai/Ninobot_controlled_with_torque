"""Train a PPO wheel-torque policy against the running Gazebo world."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import shutil
import sys

from ament_index_python.packages import get_package_share_directory

from nino_rl.core import load_config


def arguments() -> argparse.Namespace:
    default_config = Path(get_package_share_directory("nino_rl")) / "config" / "ppo.yaml"
    parser = argparse.ArgumentParser(description="Train PPO for Nino wheel torques")
    parser.add_argument("--config", type=Path, default=default_config)
    parser.add_argument("--timesteps", type=int, default=500_000)
    parser.add_argument("--output", type=Path, default=Path("rl_runs"))
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--checkpoint-every", type=int, default=25_000)
    parser.add_argument("--check-env", action="store_true")
    return parser.parse_args(sys.argv[1:])


def main() -> None:
    args = arguments()
    if args.timesteps <= 0:
        raise SystemExit("--timesteps phải lớn hơn 0")
    try:
        import torch as th
        from stable_baselines3 import PPO
        from stable_baselines3.common.callbacks import CheckpointCallback
        from stable_baselines3.common.env_checker import check_env
        from stable_baselines3.common.monitor import Monitor
    except ImportError as error:
        raise SystemExit(
            "Thiếu thư viện RL. Kích hoạt .venv và chạy: "
            "pip install -r src/nino_rl/requirements.txt"
        ) from error

    config = load_config(args.config)
    device = str(config.get("device", "cuda"))
    if device.startswith("cuda") and not th.cuda.is_available():
        raise SystemExit(
            "Cấu hình yêu cầu CUDA nhưng torch.cuda.is_available() = False. "
            "Chạy `ros2 run nino_rl check_cuda` để chẩn đoán."
        )

    from nino_rl.ros_env import NinoGazeboEnv

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = args.output.expanduser().resolve() / stamp
    checkpoint_dir = run_dir / "checkpoints"
    tensorboard_dir = run_dir / "tensorboard"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    tensorboard_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.config, run_dir / "ppo.yaml")

    env = NinoGazeboEnv(config, total_training_steps=args.timesteps)
    try:
        if args.check_env:
            check_env(env, warn=True)
        monitored = Monitor(env, filename=str(run_dir / "monitor.csv"))
        ppo = config["ppo"]
        if args.resume:
            model = PPO.load(args.resume, env=monitored, device=device)
            env.global_steps = int(model.num_timesteps)
            env.total_training_steps = int(model.num_timesteps) + args.timesteps
            reset_num_timesteps = False
        else:
            activation = {"elu": th.nn.ELU, "relu": th.nn.ReLU, "tanh": th.nn.Tanh}[
                str(ppo["activation"]).lower()
            ]
            layers = [int(value) for value in ppo["policy_layers"]]
            model = PPO(
                "MlpPolicy",
                monitored,
                learning_rate=float(ppo["learning_rate"]),
                gamma=float(ppo["gamma"]),
                gae_lambda=float(ppo["gae_lambda"]),
                n_steps=int(ppo["n_steps"]),
                batch_size=int(ppo["batch_size"]),
                n_epochs=int(ppo["n_epochs"]),
                clip_range=float(ppo["clip_range"]),
                ent_coef=float(ppo["ent_coef"]),
                vf_coef=float(ppo["vf_coef"]),
                max_grad_norm=float(ppo["max_grad_norm"]),
                policy_kwargs={
                    "activation_fn": activation,
                    "net_arch": {"pi": layers, "vf": layers},
                },
                tensorboard_log=str(tensorboard_dir),
                device=device,
                seed=int(config["seed"]),
                verbose=1,
            )
            reset_num_timesteps = True

        callback = CheckpointCallback(
            save_freq=max(1, int(args.checkpoint_every)),
            save_path=str(checkpoint_dir),
            name_prefix="nino_ppo",
            save_replay_buffer=False,
            save_vecnormalize=True,
        )
        print(f"Bắt đầu train trên {model.device}; kết quả: {run_dir}")
        model.learn(
            total_timesteps=args.timesteps,
            callback=callback,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=True,
        )
        final_path = run_dir / "nino_ppo_final"
        model.save(final_path)
        print(f"Đã lưu policy: {final_path}.zip")
    finally:
        env.close()


if __name__ == "__main__":
    main()
