#!/usr/bin/env python3
"""Check saved SAC smoke evidence and load/predict with the saved model."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from plotting_compat import prepare_matplotlib_namespace
prepare_matplotlib_namespace()
import numpy as np
from stable_baselines3 import SAC

from ros_env import OBSERVATIONS
from runtime import resolve_device


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_dir')
    parser.add_argument('--device', choices=['cpu', 'cuda', 'auto'], default='cpu')
    args = parser.parse_args()
    run = Path(args.run_dir)
    result = json.loads((run/'result.json').read_text())
    assert result['gradient_updates'] > 0
    peak, count, episodes = 0.0, 0, 0
    for path in sorted(run.glob('episode_*/steps.jsonl')):
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        assert rows
        assert all(row['planning_requests'] == 1 for row in rows)
        assert len({row['reference_geometry_sha256'] for row in rows}) == 1
        for row in rows:
            residual = np.asarray(row['residual_applied'])
            assert np.all(np.isfinite(residual))
            assert np.max(np.abs(residual)) <= 0.300001
            assert np.max(np.abs(row['total_torque'])) <= 2.000001
            peak = max(peak, float(np.max(np.abs(residual))))
        count += len(rows)
        episodes += 1
    assert peak > 0.01, 'No nonzero residual was observed in the real torque adapter'
    assert episodes >= 2, 'Smoke should exercise a full simulation reset'
    device, device_info = resolve_device(args.device)
    model = SAC.load(run/'model.zip', device=device)
    action, _ = model.predict(np.zeros(len(OBSERVATIONS), dtype=np.float32), deterministic=True)
    assert action.shape == (2,) and np.all(np.isfinite(action))
    assert np.max(np.abs(action)) <= 1.000001
    evidence = dict(episodes=episodes, logged_steps=count, peak_residual_nm=peak,
                    gradient_updates=result['gradient_updates'], model_load_predict=True,
                    immutable_reference_per_episode=True, total_torque_within_2nm=True,
                    training_device=json.loads((run/'metadata.json').read_text())['device'],
                    loaded_device=str(model.device), device_info=device_info,
                    requested_goal=rows[-1].get('requested_goal'),
                    reference_goal=rows[-1].get('reference_goal'), policy_validated=False)
    (run/'verification.json').write_text(json.dumps(evidence, indent=2))
    print(json.dumps(evidence, indent=2))


if __name__ == '__main__':
    main()
