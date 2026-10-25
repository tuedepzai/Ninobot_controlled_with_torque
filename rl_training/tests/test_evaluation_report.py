import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluation_report import comparison_data, sample_rmse


def test_rmse_uses_sample_count_not_duration_or_signed_mean():
    assert sample_rmse([-3, 3]) == 3
    assert sample_rmse([3, 4]) == pytest.approx(np.sqrt(12.5))


@pytest.mark.parametrize('errors', [[], [float('nan')], [float('inf')]])
def test_invalid_rmse_rejected(errors):
    with pytest.raises(ValueError):
        sample_rmse(errors)


def test_common_progress_excludes_untraversed_tail_but_keeps_failure(tmp_path):
    report = {}
    for key, progress, error, outcome in (
            ('baseline', [0, 1, 2, 3], [1, 1, 1, 99], 'SUCCEEDED'),
            ('policy', [0, 1, 2], [2, 2, 2], 'FAILED')):
        root = tmp_path/key
        folder = root/'episode_1'
        folder.mkdir(parents=True)
        rows = [dict(simulation_time=i+1, progress=p, cross_track=e/100,
                     acceleration=[0, 0, 0], total_torque=[0, 0])
                for i, (p, e) in enumerate(zip(progress, error))]
        (folder/'steps.jsonl').write_text('\n'.join(json.dumps(r) for r in rows))
        report[key] = dict(run=str(root), episodes=[dict(episode='episode_1', outcome=outcome, reason='')])
    groups, pairs = comparison_data(report)
    assert groups[1][0]['outcome'] == 'FAILED'
    assert pairs[0]['end_m'] == 2
    assert pairs[0]['baseline_rmse_cm'] == 1
    assert pairs[0]['policy_rmse_cm'] == 2
