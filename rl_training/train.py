#!/usr/bin/env python3
"""SAC training, zero-residual baseline, and deterministic policy evaluation."""

import argparse
from copy import deepcopy
from datetime import datetime
import json
from pathlib import Path

import numpy as np
from plotting_compat import prepare_matplotlib_namespace
prepare_matplotlib_namespace()
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
import torch
import yaml

from ros_env import NinoResidualEnv, OBSERVATIONS
from runtime import resolve_device, verify_model_device
from episode_outcome import InfrastructureFault


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['train', 'smoke', 'baseline', 'evaluate'], default='baseline')
    parser.add_argument('--map', required=True)
    parser.add_argument('--config', default=str(root/'config/sac.yaml'))
    parser.add_argument('--steps', type=int, default=10000)
    parser.add_argument('--episodes', type=int, default=1)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--model', help='SAC .zip for evaluation')
    parser.add_argument('--resume', help='SAC .zip to resume training; --steps is the total target')
    parser.add_argument('--replay-buffer', help='Replay .pkl paired with --resume')
    parser.add_argument('--checkpoint-every', type=int, default=100000)
    parser.add_argument('--goal-x', type=float)
    parser.add_argument('--max-episode-seconds', type=float)
    parser.add_argument('--gui', action='store_true')
    parser.add_argument('--device', choices=['cuda', 'cpu', 'auto'], default='cuda',
                        help='Training/inference device; cuda fails clearly if unusable')
    args = parser.parse_args()
    if args.steps <= 0 or args.episodes <= 0:
        parser.error('steps and episodes must be positive')
    if args.mode == 'evaluate' and not args.model:
        parser.error('--model is required for evaluation')
    if args.checkpoint_every <= 0:
        parser.error('--checkpoint-every must be positive')
    if bool(args.resume) != bool(args.replay_buffer):
        parser.error('--resume and --replay-buffer must be used together')
    if args.resume and args.mode not in ('train', 'smoke'):
        parser.error('--resume is only for train/smoke')
    for value in (args.resume, args.replay_buffer):
        if value and not Path(value).is_file():
            parser.error(f'File not found: {value}')
    device, device_info = resolve_device('cpu' if args.mode == 'baseline' else args.device)
    config = yaml.safe_load(Path(args.config).read_text())
    if args.goal_x is not None:
        config['environment']['goal_x'] = args.goal_x
    if args.max_episode_seconds is not None:
        config['environment']['max_episode_seconds'] = args.max_episode_seconds
    if args.gui:
        config['environment']['headless'] = False
    if args.mode == 'smoke':
        # Exercise actual replay/gradient updates quickly, without claiming learning.
        config['sac'].update(learning_starts=8, batch_size=8, buffer_size=1000, net_arch=[32, 32])
        config['environment']['max_episode_seconds'] = 2.0
        if args.steps == 10000:
            args.steps = 32
    run = root/'runs'/(args.mode+'_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    run.mkdir(parents=True)
    (run/'config.yaml').write_text(yaml.safe_dump(config, sort_keys=False))
    metadata = dict(mode=args.mode, seed=args.seed, map=str(Path(args.map).resolve()),
                    observations=OBSERVATIONS, action_order=['left', 'right'],
                    action_scale_nm=config['environment']['residual_limit'],
                    model=str(Path(args.model).resolve()) if args.model else None,
                    deterministic=args.mode == 'evaluate',
                    resume_model=str(Path(args.resume).resolve()) if args.resume else None,
                    resume_replay=str(Path(args.replay_buffer).resolve()) if args.replay_buffer else None,
                    target_timesteps=args.steps, checkpoint_every=args.checkpoint_every,
                    device=device, device_info=device_info, torch=torch.__version__, policy_validated=False)
    (run/'metadata.json').write_text(json.dumps(metadata, indent=2))
    print(f'Run: {run}', flush=True)
    print(f'Device: {device} — {device_info}', flush=True)
    torch.set_num_threads(1)
    env = Monitor(NinoResidualEnv(config, args.map, run,
                                 residual_enabled=args.mode != 'baseline'), str(run/'monitor.csv'))
    model = None
    try:
        if args.mode in ('train', 'smoke'):
            parameters = deepcopy(config['sac'])
            net_arch = parameters.pop('net_arch')
            if args.resume:
                source = Path(args.resume).resolve().parent
                if source.name == 'checkpoints':
                    source = source.parent
                saved_config = yaml.safe_load((source/'config.yaml').read_text())
                if saved_config != config:
                    raise ValueError('Resume config differs from saved config; use the original --config')
                model = SAC.load(args.resume, env=env, device=device, force_reset=True)
                model.load_replay_buffer(args.replay_buffer)
            else:
                model = SAC('MlpPolicy', env, policy_kwargs={'net_arch': net_arch},
                            device=device, seed=args.seed, verbose=1, **parameters)
            remaining = remaining_steps(args.steps, model.num_timesteps)
            evidence = verify_model_device(model, device)
            (run/'device_verification.json').write_text(json.dumps(evidence, indent=2))
            print(f'SB3 verified: actor={evidence["actor_device"]}, critic={evidence["critic_device"]}, '
                  f'forward={evidence["actor_forward_device"]}', flush=True)
            print(f'Target: {args.steps}; restored: {model.num_timesteps}; remaining: {remaining}', flush=True)
            callback = CheckpointCallback(save_freq=args.checkpoint_every, save_path=str(run/'checkpoints'),
                                          name_prefix='sac', save_replay_buffer=True)
            if remaining:
                model.learn(total_timesteps=remaining, callback=callback, log_interval=1,
                            reset_num_timesteps=not bool(args.resume))
            if args.mode == 'smoke' and model._n_updates == 0:
                raise RuntimeError('Smoke test did not perform any SAC gradient updates; increase --steps')
            model.save(str(run/'model'))
            model.save_replay_buffer(str(run/'replay_buffer.pkl'))
            (run/'device_verification.json').write_text(json.dumps(verify_model_device(model, device), indent=2))
            result = dict(timesteps=model.num_timesteps, gradient_updates=model._n_updates,
                          smoke_only=args.mode == 'smoke', policy_validated=False)
            (run/'result.json').write_text(json.dumps(result, indent=2))
            print(json.dumps(result), flush=True)
        else:
            if args.mode == 'evaluate':
                model = SAC.load(args.model, env=env, device=device)
            results = []
            for episode in range(args.episodes):
                observation, info = env.reset(seed=args.seed+episode)
                total_reward, steps = 0.0, 0
                while True:
                    action = np.zeros(2, dtype=np.float32) if model is None else model.predict(
                        observation, deterministic=True)[0]
                    observation, reward, terminated, truncated, info = env.step(action)
                    total_reward += reward
                    steps += 1
                    if terminated or truncated:
                        break
                results.append(dict(episode=episode, reward=total_reward, steps=steps,
                                    outcome=info['outcome'], path_sha256=info['path_sha256'],
                                    planning_requests=info['planning_requests']))
                (run/'result.json').write_text(json.dumps(results, indent=2))
                print(json.dumps(results[-1]), flush=True)
            (run/'result.json').write_text(json.dumps(results, indent=2))
    except Exception as exc:
        snapshot = 'infrastructure_fault' if isinstance(exc, InfrastructureFault) else 'exception'
        if model is not None:
            model.save(str(run/(snapshot+'_model')))
            if model.replay_buffer is not None:
                model.save_replay_buffer(str(run/(snapshot+'_replay.pkl')))
        (run/(snapshot+'.json')).write_text(json.dumps({'reason': str(exc), 'type': type(exc).__name__}, indent=2))
        print('Run stopped for an exception; no fake failure reward was added. '
              'Available model/replay saved for inspection.', flush=True)
        raise
    except KeyboardInterrupt:
        if model is not None:
            model.save(str(run/'interrupted_model'))
            if model.replay_buffer is not None:
                model.save_replay_buffer(str(run/'interrupted_replay.pkl'))
        print('Interrupted; residual cleared and owned sessions stopped.', flush=True)
    finally:
        env.close()


def remaining_steps(target, completed):
    if target < completed:
        raise ValueError(f'Target {target} is below restored timesteps {completed}')
    return target-completed


if __name__ == '__main__':
    main()
