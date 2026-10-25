#!/usr/bin/env python3
"""Export an episode's path error, IMU and torque traces as PNG."""

import argparse
import json
from pathlib import Path

from plotting_compat import prepare_matplotlib_namespace
prepare_matplotlib_namespace()
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('episode_dir', help='Directory containing steps.jsonl')
    args = parser.parse_args()
    directory = Path(args.episode_dir)
    rows = [json.loads(line) for line in (directory/'steps.jsonl').read_text().splitlines()]
    if not rows:
        raise ValueError('No steps recorded')
    t = np.array([r['simulation_time'] for r in rows])
    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True, constrained_layout=True)
    axes[0].plot(t, [r['cross_track'] for r in rows], label='Signed path error')
    config = __import__('yaml').safe_load((directory.parent/'config.yaml').read_text())
    threshold = config['environment']['max_path_deviation']
    axes[0].axhline(threshold, color='r', linestyle='--', label='Failure threshold')
    axes[0].axhline(-threshold, color='r', linestyle='--')
    axes[0].set_ylabel('Path error (m)')
    acceleration = np.asarray([r['acceleration'] for r in rows])
    for i, label in enumerate(('X', 'Y', 'Z')):
        axes[1].plot(t, acceleration[:, i], label=f'Acceleration {label}')
    axes[1].set_ylabel('IMU, gravity removed (m/s²)')
    residual = np.asarray([r['residual_applied'] for r in rows])
    total = np.asarray([r['total_torque'] for r in rows])
    for i, label in enumerate(('left', 'right')):
        axes[2].plot(t, total[:, i], label=f'Total {label}')
        axes[2].plot(t, residual[:, i], '--', label=f'Residual {label}')
    axes[2].set_ylabel('Torque (N·m)')
    axes[2].set_xlabel('Simulation time (s)')
    for ax in axes:
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right')
    fig.suptitle('Nino residual torque episode — sampled at RL step boundaries')
    target = directory/'episode.png'
    fig.savefig(target, dpi=160)
    print(target)


if __name__ == '__main__':
    main()
