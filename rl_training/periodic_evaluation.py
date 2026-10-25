"""Milestones and persistent evaluation records; no ROS or Torch import."""
import csv
import json
from pathlib import Path
import zipfile


def model_timesteps(path):
    with zipfile.ZipFile(path) as archive:
        step = json.loads(archive.read('data'))['num_timesteps']
    if not isinstance(step, int) or step < 0:
        raise ValueError('Invalid saved timestep counter')
    return step


def milestones(completed, target, interval):
    if interval <= 0 or target < completed:
        raise ValueError('Invalid evaluation interval/target')
    next_step = (completed//interval+1)*interval
    if completed and completed % interval == 0 and completed < target:
        yield completed  # Resume can follow a saved milestone whose evaluation failed.
    while next_step < target:
        yield next_step
        next_step += interval
    yield target


def evaluation_row(step, report_path, origin='periodic'):
    report_path = Path(report_path)
    report = json.loads((report_path/'report.json').read_text())
    baseline, policy = (report[key]['aggregate'] for key in ('baseline', 'policy'))
    results = json.loads((Path(report['policy']['run'])/'result.json').read_text())
    return dict(step=step, origin=origin, episodes=policy['completed_episodes'],
                successes=policy['outcome_counts']['SUCCEEDED'], success_rate=policy['success_rate'],
                rmse_cm=policy['cross_track_rmse_m_mean']*100,
                imu_rms_m_s2=policy['acceleration_rms_m_s2_mean'],
                mean_reward=sum(r['reward'] for r in results)/len(results),
                pi_episodes=baseline['completed_episodes'],
                pi_successes=baseline['outcome_counts']['SUCCEEDED'],
                pi_rmse_cm=baseline['cross_track_rmse_m_mean']*100,
                report=str(report_path), model=report['policy']['metadata'].get('model'))


def save_evaluations(folder, rows):
    folder = Path(folder)
    temp = folder/'evaluations.tmp'
    temp.write_text(json.dumps(rows, indent=2))
    temp.replace(folder/'evaluations.json')
    with (folder/'evaluations.csv.tmp').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (folder/'evaluations.csv.tmp').replace(folder/'evaluations.csv')
