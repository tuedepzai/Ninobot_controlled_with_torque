from copy import deepcopy
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from compare_runs import compare, episode_metrics, summarize_run


def samples():
    return [dict(simulation_time=t, step_duration=dt, cross_track=e,
                 acceleration=[a, 0, 0], total_torque=[1, -2], residual_applied=[0, 0],
                 planning_requests=1, reference_geometry_sha256='fixed', remaining=1,
                 outcome=outcome)
            for t, dt, e, a, outcome in ((1, 1, 0, 3, None), (4, 3, 2, 1, 'FAILED'))]


def test_duration_weighted_rms_and_terminal_outcome():
    metric = episode_metrics(samples())
    assert metric['cross_track_rms_m'] == pytest.approx(np.sqrt(3))
    assert metric['cross_track_rmse_m'] == pytest.approx(np.sqrt(2))
    assert metric['acceleration_rms_m_s2'] == pytest.approx(np.sqrt(3))
    assert metric['duration_s'] == 4 and metric['outcome'] == 'FAILED'
    assert metric['torque_peak_nm'] == 2


def test_old_records_use_simulation_time_differences():
    rows = samples()
    for row in rows:
        del row['step_duration']
    assert episode_metrics(rows) == episode_metrics(samples())


@pytest.mark.parametrize('key,value', [('planning_requests', 2),
                                     ('reference_geometry_sha256', 'changed'),
                                     ('cross_track', float('nan')),
                                     ('simulation_time', 0)])
def test_invalid_episode_is_not_silently_compared(key, value):
    rows = samples()
    rows[-1][key] = value
    with pytest.raises(ValueError):
        episode_metrics(rows)


def test_incomplete_and_startup_sessions_excluded_from_success_denominator(tmp_path):
    (tmp_path/'metadata.json').write_text(json.dumps({'mode': 'baseline'}))
    (tmp_path/'config.yaml').write_text(yaml.safe_dump({}))
    for name, outcome in [('episode_1', 'SUCCEEDED'), ('episode_2', None)]:
        folder = tmp_path/name
        folder.mkdir()
        rows = samples()
        rows[-1]['outcome'] = outcome
        (folder/'steps.jsonl').write_text('\n'.join(json.dumps(r) for r in rows))
    (tmp_path/'episode_3').mkdir()
    summary = summarize_run(tmp_path)
    assert summary['aggregate']['success_rate'] == 1
    assert summary['aggregate']['completed_episodes'] == 1
    assert summary['aggregate']['incomplete_episodes'] == 1
    assert summary['aggregate']['startup_only_sessions'] == 1


def test_comparison_rejects_unmatched_conditions_and_smoke():
    base = dict(metadata=dict(mode='baseline', map='/map', seed=1, observations=['x'],
                              action_order=['left', 'right'], action_scale_nm=0.3),
                config={'environment': {'goal_x': 11}}, aggregate={'completed_episodes': 3})
    policy = deepcopy(base)
    policy['metadata']['mode'] = 'evaluate'
    assert compare(base, policy)['policy_validated'] is False
    for mutate in (lambda p: p['config']['environment'].update(goal_x=10),
                   lambda p: p['metadata'].update(mode='smoke'),
                   lambda p: p['metadata'].update(seed=2),
                   lambda p: p['aggregate'].update(completed_episodes=2)):
        invalid = deepcopy(policy)
        mutate(invalid)
        with pytest.raises(ValueError):
            compare(base, invalid)


def test_baseline_with_active_residual_is_rejected(tmp_path):
    (tmp_path/'metadata.json').write_text(json.dumps({'mode': 'baseline'}))
    (tmp_path/'config.yaml').write_text('{}')
    episode = tmp_path/'episode_1'
    episode.mkdir()
    rows = samples()
    rows[0]['residual_applied'] = [0.1, 0]
    (episode/'steps.jsonl').write_text('\n'.join(json.dumps(r) for r in rows))
    with pytest.raises(ValueError, match='nonzero residual'):
        summarize_run(tmp_path)


def test_samples_after_terminal_are_rejected():
    rows = samples()
    rows[0]['outcome'] = 'SUCCEEDED'
    with pytest.raises(ValueError, match='Terminal outcome before final'):
        episode_metrics(rows)


def test_terminal_result_without_clock_advance_keeps_outcome_without_duration():
    rows = samples()
    terminal = deepcopy(rows[-1])
    rows[-1]['outcome'] = None
    terminal.update(step_duration=1e-6, cross_track=1000, acceleration=[1000, 0, 0])
    metric = episode_metrics([*rows, terminal])
    assert metric['outcome'] == 'FAILED'
    assert metric['duration_s'] == 4
    assert metric['cross_track_rms_m'] == pytest.approx(np.sqrt(3))
    assert metric['acceleration_rms_m_s2'] == pytest.approx(np.sqrt(3))
