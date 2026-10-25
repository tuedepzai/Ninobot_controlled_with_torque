from pathlib import Path
import sys
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from runtime import resolve_device


def test_cpu_is_explicit_and_auto_can_fall_back():
    assert resolve_device('cpu')[0] == 'cpu'
    with patch('torch.cuda.is_available', return_value=False):
        assert resolve_device('auto')[0] == 'cpu'


def test_requested_gpu_must_not_silently_fall_back():
    with patch('torch.cuda.is_available', return_value=False):
        with pytest.raises(RuntimeError, match='CUDA unavailable'):
            resolve_device('cuda')


def test_matching_matplotlib_toolkit_is_available():
    from plotting_compat import prepare_matplotlib_namespace
    prepare_matplotlib_namespace()
    from mpl_toolkits.mplot3d import Axes3D
    assert Axes3D.name == '3d'


def test_sb3_device_verification_checks_parameters_and_actor_forward():
    import gymnasium as gym
    from stable_baselines3 import SAC
    from runtime import verify_model_device
    env = gym.make('Pendulum-v1')
    try:
        model = SAC('MlpPolicy', env, device='cpu', policy_kwargs={'net_arch': [8, 8]})
        evidence = verify_model_device(model, 'cpu')
        assert evidence['actor_device'] == 'cpu'
        assert evidence['critic_device'] == 'cpu'
        assert evidence['actor_forward_device'] == 'cpu'
        with pytest.raises(RuntimeError, match='CUDA mismatch'):
            verify_model_device(model, 'cuda')
    finally:
        env.close()
