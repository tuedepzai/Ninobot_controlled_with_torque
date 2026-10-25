#!/usr/bin/env python3
"""Compare recorded PI baseline and deterministic SAC evaluation, without ROS."""

import argparse
import csv
from datetime import datetime
import json
from pathlib import Path

import numpy as np
import yaml


def episode_metrics(rows):
    if not rows:
        raise ValueError('Empty episode')
    t = np.asarray([r['simulation_time'] for r in rows], dtype=float)
    dt = np.asarray([r.get('step_duration', delta)
                     for r, delta in zip(rows, np.diff(np.r_[0.0, t]))])
    error = np.asarray([r['cross_track'] for r in rows], dtype=float)
    acc = np.asarray([r['acceleration'] for r in rows], dtype=float)
    torque = np.asarray([r['total_torque'] for r in rows], dtype=float)
    intervals = np.diff(np.r_[0.0, t])
    if (not all(np.all(np.isfinite(v)) for v in (t, dt, error, acc, torque))
            or np.any(dt < 0) or np.any(intervals < 0)):
        raise ValueError('Invalid samples or non-increasing simulation time')
    # Action completion can arrive without a new /clock message. Keep the
    # final outcome, but do not give this duplicate terminal sample duration.
    for i in np.flatnonzero(intervals == 0):
        if i != len(rows)-1 or not rows[i].get('outcome'):
            raise ValueError('Duplicate timestamp outside terminal sample')
        dt[i] = 0.0
    if np.sum(dt) <= 0:
        raise ValueError('Episode has no elapsed sampling time')
    if acc.shape != (len(rows), 3) or torque.shape != (len(rows), 2):
        raise ValueError('Unexpected acceleration/torque shape')
    if any(r['planning_requests'] != 1 for r in rows):
        raise ValueError('Episode does not have exactly one planning request')
    hashes = {r['reference_geometry_sha256'] for r in rows}
    if len(hashes) != 1 or None in hashes:
        raise ValueError('Missing or changed reference geometry')
    if any(r.get('outcome') for r in rows[:-1]):
        raise ValueError('Terminal outcome before final sample')
    outcome = rows[-1].get('outcome') or 'INCOMPLETE'
    if outcome not in ('SUCCEEDED', 'FAILED', 'TIME_LIMIT', 'INCOMPLETE'):
        raise ValueError(f'Unexpected outcome: {outcome}')
    return dict(outcome=outcome, samples=len(rows), duration_s=float(t[-1]),
                cross_track_rmse_m=float(np.sqrt(np.mean(error**2))),
                cross_track_rms_m=float(np.sqrt(np.average(error**2, weights=dt))),
                cross_track_peak_m=float(np.max(np.abs(error))),
                acceleration_rms_m_s2=float(np.sqrt(np.average(np.sum(acc**2, axis=1), weights=dt))),
                torque_peak_nm=float(np.max(np.abs(torque))),
                remaining_m=rows[-1]['remaining'], reason=rows[-1].get('reason', ''),
                reference_geometry_sha256=next(iter(hashes)))


def summarize_run(directory):
    directory = Path(directory)
    metadata = json.loads((directory/'metadata.json').read_text())
    config = yaml.safe_load((directory/'config.yaml').read_text())
    episodes, startup_only = [], []
    for folder in sorted(directory.glob('episode_*')):
        path = folder/'steps.jsonl'
        if not path.exists() or not path.stat().st_size:
            startup_only.append(folder.name)
            continue
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        metrics = episode_metrics(rows)
        if metadata['mode'] == 'baseline' and any(
                np.any(np.asarray(r['residual_applied']) != 0) for r in rows):
            raise ValueError('Baseline contains nonzero residual torque')
        episodes.append(dict(episode=folder.name, **metrics))
    complete = [e for e in episodes if e['outcome'] != 'INCOMPLETE']
    if not complete:
        raise ValueError(f'No complete episodes in {directory}')
    aggregate = dict(completed_episodes=len(complete),
                     incomplete_episodes=len(episodes)-len(complete),
                     startup_only_sessions=len(startup_only),
                     outcome_counts={outcome: sum(e['outcome'] == outcome for e in complete)
                                     for outcome in ('SUCCEEDED', 'FAILED', 'TIME_LIMIT')},
                     success_rate=sum(e['outcome'] == 'SUCCEEDED' for e in complete)/len(complete))
    for metric in ('duration_s', 'cross_track_rmse_m', 'cross_track_rms_m', 'cross_track_peak_m',
                   'acceleration_rms_m_s2', 'torque_peak_nm'):
        aggregate[metric+'_mean'] = float(np.mean([e[metric] for e in complete]))
    return dict(run=str(directory.resolve()), metadata=metadata, config=config,
                aggregate=aggregate, episodes=episodes, startup_only=startup_only)


def compare(baseline, policy):
    if baseline['metadata']['mode'] != 'baseline' or policy['metadata']['mode'] != 'evaluate':
        raise ValueError('Use baseline and evaluate runs; training/smoke are not evaluation')
    if baseline['config'] != policy['config']:
        raise ValueError('Run configurations differ; rerun with the same --config')
    for key in ('map', 'seed', 'observations', 'action_order', 'action_scale_nm'):
        if baseline['metadata'][key] != policy['metadata'][key]:
            raise ValueError(f'Run metadata differs: {key}')
    if baseline['aggregate']['completed_episodes'] != policy['aggregate']['completed_episodes']:
        raise ValueError('Use the same number of completed episodes')
    return dict(baseline=baseline, policy=policy, policy_validated=False,
                limitations=[
                    'Metrics use samples at RL step boundaries, not full-rate IMU.',
                    'RMS is weighted by sampled step duration; timing differs across runs.',
                    'AMCL estimates can miss real slip; no ground-truth metric is recorded.',
                    'Same seeds do not make Gazebo/AMCL noise or reference paths identical.',
                    'Short/failed episodes can have lower RMS; compare success first.',
                    'Means describe completed episodes only; incomplete sessions are listed.',
                    'Matching map paths does not prove historical map contents were identical.',
                    'This report does not establish statistical significance or policy quality.'])


def plot_comparison(report, target):
    from plotting_compat import prepare_matplotlib_namespace
    prepare_matplotlib_namespace()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    groups = [report['baseline'], report['policy']]
    axes[0].bar(['PI', 'PI + SAC'], [g['aggregate']['success_rate'] for g in groups])
    axes[0].set(ylabel='Success fraction', ylim=(0, 1.05))
    for ax, key, unit in zip(axes[1:], ('cross_track_rms_m', 'acceleration_rms_m_s2'),
                             ('Path error RMS (m)', 'Acceleration magnitude RMS (m/s²)')):
        for i, group in enumerate(groups):
            for episode in group['episodes']:
                if episode['outcome'] != 'INCOMPLETE':
                    ax.scatter(i, episode[key], marker='o' if episode['outcome'] == 'SUCCEEDED' else 'x',
                               color=('tab:blue', 'tab:orange')[i])
        ax.set(xticks=[0, 1], xticklabels=['PI', 'PI + SAC'], ylabel=unit, xlim=(-0.5, 1.5))
    fig.suptitle('Recorded evaluation: o success, x failure/time limit; sampled IMU')
    fig.savefig(target, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', required=True)
    parser.add_argument('--policy', required=True)
    parser.add_argument('--output', help='New report directory; existing directories are refused')
    args = parser.parse_args()
    report = compare(summarize_run(args.baseline), summarize_run(args.policy))
    output = Path(args.output) if args.output else Path(__file__).resolve().parent/'runs'/(
        'comparison_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    output.mkdir(parents=True, exist_ok=False)
    (output/'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    rows = [dict(group=group, **e) for group in ('baseline', 'policy') for e in report[group]['episodes']]
    with (output/'episodes.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    plot_comparison(report, output/'comparison.png')
    from evaluation_report import write_detailed_report
    write_detailed_report(report, output)
    print(json.dumps({k: report[k]['aggregate'] for k in ('baseline', 'policy')}, indent=2))
    print(f'Report: {output}')


if __name__ == '__main__':
    main()
