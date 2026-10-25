#!/usr/bin/env python3
"""Persist a long SAC job, then evaluate PI/SAC sequentially and export RMSE."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
from periodic_evaluation import evaluation_row, milestones, model_timesteps, save_evaluations


def model_pair(run):
    run = Path(run).resolve()
    for model, replay in (('model.zip', 'replay_buffer.pkl'),
                          ('infrastructure_fault_model.zip', 'infrastructure_fault_replay.pkl'),
                          ('exception_model.zip', 'exception_replay.pkl'),
                          ('interrupted_model.zip', 'interrupted_replay.pkl')):
        if (run/model).is_file() and (run/replay).is_file():
            return run/model, run/replay
    for model in sorted((run/'checkpoints').glob('sac_*_steps.zip'),
                        key=lambda p: int(p.stem.split('_')[1]), reverse=True):
        replay = model.with_name(model.name.replace('sac_', 'sac_replay_buffer_').replace('.zip', '.pkl'))
        if replay.is_file():
            return model, replay
    raise ValueError(f'No paired model/replay in {run}')


def main():
    def interrupt(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, interrupt)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resume-train-run', required=True)
    parser.add_argument('--map', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--steps', type=int, default=50000000)
    parser.add_argument('--episodes', type=int, default=10)
    parser.add_argument('--seed', type=int, default=101)
    parser.add_argument('--checkpoint-every', type=int, default=100000)
    parser.add_argument('--eval-every', type=int, default=20000)
    parser.add_argument('--eval-episodes', type=int, default=3)
    parser.add_argument('--import-evaluation', help='Existing comparison directory with assessment.json timestep evidence')
    args = parser.parse_args()
    if min(args.steps, args.episodes, args.checkpoint_every, args.eval_every, args.eval_episodes) <= 0:
        parser.error('steps, episodes and checkpoint-every must be positive')
    source = Path(args.resume_train_run).resolve()
    model, replay = model_pair(source)
    completed = model_timesteps(model)
    if completed > args.steps:
        parser.error('Target is below the restored model timestep count')
    root = Path(__file__).resolve().parent
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    state = dict(pid=os.getpid(), state='STARTING', target_steps=args.steps,
                 source_run=str(source), evaluation_episodes=args.episodes,
                 checkpoint_every=args.checkpoint_every, map=str(Path(args.map).resolve()),
                 evaluation_seed=args.seed, phases={}, eval_every=args.eval_every,
                 periodic_evaluation_episodes=args.eval_episodes, completed_training_steps=completed,
                 training_runs=[], evaluations=0)
    rows, baselines = [], {}
    if args.import_evaluation:
        imported = Path(args.import_evaluation).resolve()
        evidence = json.loads((imported/'assessment.json').read_text())
        rows.append(evaluation_row(evidence['snapshot']['timesteps'], imported, origin='manual'))
        save_evaluations(output, rows)
        state['evaluations'] = len(rows)

    def save():
        state['updated_utc'] = datetime.now(timezone.utc).isoformat()
        temp = output/'status.tmp'
        temp.write_text(json.dumps(state, indent=2))
        temp.replace(output/'status.json')

    def run_phase(name, arguments, milestone):
        state['state'] = name.upper()
        log_name = f'{name}_{milestone:012d}.log'
        if name == 'train':
            state['train_log'] = log_name
            state['phases'].pop('train', None)
        save()
        with (output/log_name).open('w') as log:
            child = subprocess.Popen([sys.executable, '-u', *arguments], stdout=subprocess.PIPE,
                                     stderr=subprocess.STDOUT, text=True, start_new_session=True)
            state['child_pid'] = child.pid
            save()
            try:
                for line in child.stdout:
                    log.write(line)
                    log.flush()
                    if line.startswith(('Run: ', 'Report: ')):
                        state['phases'][name] = line.split(': ', 1)[1].strip()
                        if name == 'train':
                            state['training_runs'].append(state['phases'][name])
                        save()
                code = child.wait()
            except KeyboardInterrupt:
                os.killpg(child.pid, signal.SIGINT)
                child.wait(timeout=120)
                raise
            finally:
                child.stdout.close()
        state.pop('child_pid', None)
        save()
        if code:
            raise RuntimeError(f'{name} exited with code {code}; inspect {output/log_name}')
        return Path(state['phases'][name])

    save()
    try:
        for milestone in milestones(completed, args.steps, args.eval_every):
            state['next_evaluation_step'] = milestone
            trained = run_phase('train', [str(root/'train.py'), '--mode', 'train', '--map', args.map,
                '--config', str(source/'config.yaml'), '--resume', str(model), '--replay-buffer', str(replay),
                '--steps', str(milestone), '--checkpoint-every', str(args.checkpoint_every), '--device', 'cuda'], milestone)
            result = json.loads((trained/'result.json').read_text())
            if result['timesteps'] != milestone:
                raise RuntimeError('Training did not reach the exact milestone; evaluation was not started')
            state['completed_training_steps'] = milestone
            episodes = args.episodes if milestone == args.steps else args.eval_episodes
            common = ['--map', args.map, '--config', str(trained/'config.yaml'),
                      '--episodes', str(episodes), '--seed', str(args.seed)]
            if episodes not in baselines:
                baselines[episodes] = run_phase('baseline', [str(root/'train.py'), '--mode', 'baseline', *common], milestone)
            policy = run_phase('evaluate', [str(root/'train.py'), '--mode', 'evaluate', *common,
                '--model', str(trained/'model.zip'), '--device', 'cuda'], milestone)
            report = run_phase('report', [str(root/'compare_runs.py'), '--baseline', str(baselines[episodes]),
                                         '--policy', str(policy)], milestone)
            rows.append(evaluation_row(milestone, report))
            save_evaluations(output, rows)
            state['evaluations'] = len(rows)
            source, model, replay = trained, trained/'model.zip', trained/'replay_buffer.pkl'
            save()
        state['state'] = 'COMPLETE'
    except KeyboardInterrupt:
        state['state'] = 'INTERRUPTED'
    except Exception as exc:
        state.update(state='FAILED', reason=str(exc))
        raise
    finally:
        save()


if __name__ == '__main__':
    main()
