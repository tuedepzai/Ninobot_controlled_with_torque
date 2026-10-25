from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from long_run import model_pair
from train import remaining_steps


def test_resume_uses_total_target_instead_of_adding_another_50m():
    assert remaining_steps(50000000, 10000) == 49990000
    assert remaining_steps(32, 32) == 0
    with pytest.raises(ValueError):
        remaining_steps(31, 32)


def test_only_matching_model_and_replay_checkpoint_are_selected(tmp_path):
    folder = tmp_path/'checkpoints'
    folder.mkdir()
    for name in ('sac_100_steps.zip', 'sac_replay_buffer_100_steps.pkl', 'sac_200_steps.zip'):
        (folder/name).touch()
    model, replay = model_pair(tmp_path)
    assert model.name == 'sac_100_steps.zip'
    assert replay.name == 'sac_replay_buffer_100_steps.pkl'


@pytest.mark.parametrize('prefix', ['infrastructure_fault', 'exception'])
def test_infrastructure_snapshot_is_usable_and_old_results_are_kept(tmp_path, prefix):
    for name in (prefix+'_model.zip', prefix+'_replay.pkl'):
        (tmp_path/name).touch()
    model, replay = model_pair(tmp_path)
    assert model.exists() and replay.exists()


def test_unexpected_reset_error_saves_model_and_replay(monkeypatch, tmp_path):
    import gymnasium as gym
    import numpy as np
    import yaml
    import train
    from stable_baselines3 import SAC

    class BrokenReset(gym.Env):
        def __init__(self, *args, **kwargs):
            self.observation_space = gym.spaces.Box(-10, 10, (23,), dtype=np.float32)
            self.action_space = gym.spaces.Box(-1, 1, (2,), dtype=np.float32)

        def reset(self, **kwargs):
            raise RuntimeError('unexpected reset failure')

    config = tmp_path/'config.yaml'
    config.write_text(yaml.safe_dump(dict(environment={'residual_limit': 0.3},
        sac=dict(net_arch=[8, 8], buffer_size=8, batch_size=2, learning_starts=1))))
    monkeypatch.setattr(train, '__file__', str(tmp_path/'train.py'))
    monkeypatch.setattr(train, 'NinoResidualEnv', BrokenReset)
    monkeypatch.setattr(sys, 'argv', ['train.py', '--mode', 'train', '--map', str(tmp_path/'map.yaml'),
        '--config', str(config), '--device', 'cpu', '--steps', '1'])
    with pytest.raises(RuntimeError, match='unexpected reset failure'):
        train.main()
    runs = list((tmp_path/'runs').iterdir())
    assert len(runs) == 1
    model, replay = model_pair(runs[0])
    loaded = SAC.load(model, device='cpu')
    loaded.load_replay_buffer(replay)
    assert loaded.num_timesteps == 0
    assert (runs[0]/'exception.json').exists()
