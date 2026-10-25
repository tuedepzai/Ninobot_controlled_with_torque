import json
from pathlib import Path
import sys
import zipfile

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from periodic_evaluation import milestones, model_timesteps, save_evaluations


def test_global_milestones_are_aligned_and_final_target_not_duplicated():
    assert list(milestones(18602, 60000, 20000)) == [20000, 40000, 60000]
    assert list(milestones(18602, 45000, 20000)) == [20000, 40000, 45000]
    assert list(milestones(20000, 60000, 20000)) == [20000, 40000, 60000]
    assert list(milestones(60000, 60000, 20000)) == [60000]


def test_invalid_interval_and_target_are_rejected():
    with pytest.raises(ValueError):
        list(milestones(1, 2, 0))
    with pytest.raises(ValueError):
        list(milestones(3, 2, 1))


def test_saved_step_count_is_read_without_loading_torch(tmp_path):
    path = tmp_path/'model.zip'
    with zipfile.ZipFile(path, 'w') as archive:
        archive.writestr('data', json.dumps({'num_timesteps': 18602}))
    assert model_timesteps(path) == 18602


def test_evaluation_table_keeps_previous_rows_and_writes_csv(tmp_path):
    row = dict(step=20000, rmse_cm=4.2, report='path,with,commas')
    save_evaluations(tmp_path, [row])
    save_evaluations(tmp_path, [row, dict(row, step=40000)])
    assert len(json.loads((tmp_path/'evaluations.json').read_text())) == 2
    assert '"path,with,commas"' in (tmp_path/'evaluations.csv').read_text()
    assert not (tmp_path/'evaluations.tmp').exists()
