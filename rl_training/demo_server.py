#!/usr/bin/env python3
"""Local GUI SAC demonstration, controlled through a separate web page."""
import argparse
from collections import deque
from copy import deepcopy
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
import subprocess
from pathlib import Path
from queue import Empty, Queue
from threading import Lock, Thread
import time

import yaml
from plotting_compat import prepare_matplotlib_namespace
prepare_matplotlib_namespace()
from stable_baselines3 import SAC
from ros_env import NinoResidualEnv
from runtime import resolve_device, verify_model_device
import numpy as np


class RestartDemo(Exception):
    def __init__(self, mode=None):
        self.mode = mode


class CloseDemo(Exception):
    pass


class DemoEnv(NinoResidualEnv):
    def __init__(self, control, *args, **kwargs):
        self.control = control
        super().__init__(*args, **kwargs)

    def _before_goal(self):
        self.control.focus_camera()
        self.control.update(state='READY', message='Gazebo ready; press Start on the web page')
        if self.control.auto_start:
            self.control.auto_start = False
            return
        while True:
            if any(child.poll() is not None for child in self.children):
                raise RuntimeError('Gazebo/navigation closed before demo start')
            command = self.control.next_command()
            if command == 'close':
                raise CloseDemo()
            self.control.check_restart(command)
            if command == 'start':
                return
            self.bridge.command([0, 0])
            self.control.idle_telemetry(self)
            time.sleep(0.1)


class Control:
    def __init__(self, model_path, config_path, map_path):
        self.lock, self.commands = Lock(), Queue()
        self.trace = deque(maxlen=600)
        self.data = dict(state='STARTING', message='Loading model and Gazebo',
                         model=str(model_path), model_steps=None, latest={}, device=None)
        self.model_path, self.config_path, self.map_path = model_path, config_path, map_path
        self.root = Path(__file__).resolve().parent/'runs'/('live_demo_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
        self.root.mkdir()
        self.server = None
        self.mode, self.auto_start = 'sac', False
        self.update(active_mode=self.mode)

    def update(self, **values):
        with self.lock:
            self.data.update(values)

    def snapshot(self):
        with self.lock:
            return dict(self.data, trace=list(self.trace), now=time.time())

    def next_command(self):
        try:
            return self.commands.get_nowait()
        except Empty:
            return None

    def command(self, command):
        if command not in ('start', 'stop', 'restart', 'close', 'test_pi', 'test_sac', 'focus'):
            raise ValueError('Unknown command')
        state = self.snapshot()['state']
        if command == 'start' and state != 'READY':
            raise ValueError('Start is allowed only when READY; use restart for a new episode')
        if state in ('CLOSED', 'ERROR'):
            raise ValueError('Demo is closed; restart demo_server.py')
        if command == 'focus':
            Thread(target=self.focus_camera, daemon=True).start()
            return
        self.commands.put(command)

    def check_restart(self, command):
        if command in ('restart', 'test_pi', 'test_sac'):
            raise RestartDemo('pi' if command == 'test_pi' else 'sac' if command == 'test_sac' else None)

    def action_for_mode(self, model, observation):
        if self.mode == 'pi':
            return np.zeros(2, dtype=np.float32)
        return model.predict(observation, deterministic=True)[0]

    def focus_camera(self):
        try:
            success = True
            for service in ('/gui/move_to', '/gui/follow'):
                response = subprocess.run(['gz', 'service', '-s', service, '--reqtype', 'gz.msgs.StringMsg',
                    '--reptype', 'gz.msgs.Boolean', '--timeout', '1500', '--req', 'data: "nino"'],
                    capture_output=True, text=True, timeout=3)
                success = success and response.returncode == 0 and 'true' in response.stdout
            self.update(camera_follow=success)
        except (OSError, subprocess.SubprocessError):
            self.update(camera_follow=False)

    def telemetry(self, info):
        row = {key: info.get(key) for key in ('simulation_time', 'cross_track', 'remaining',
                'base_torque', 'residual_applied', 'total_torque', 'commanded_residual', 'outcome')}
        row['wall_time'] = time.time()
        with self.lock:
            self.data['latest'] = row
            self.trace.append(row)

    def idle_telemetry(self, env):
        data, *_ = env.bridge.snapshot()
        row = {key: list(data[source].data) if source in data else None
               for key, source in (('base_torque', 'base'), ('residual_applied', 'residual'), ('total_torque', 'total'))}
        previous = self.snapshot()['latest']
        row.update({key: previous.get(key) for key in ('cross_track', 'remaining', 'outcome')})
        row.update(simulation_time=env.bridge.get_clock().now().nanoseconds*1e-9)
        self.telemetry(row)

    def run(self):
        env = None
        try:
            import torch
            torch.set_num_threads(1)
            device, _ = resolve_device('cuda')
            model = SAC.load(self.model_path, device=device)
            evidence = verify_model_device(model, device)
            (self.root/'device_verification.json').write_text(json.dumps(evidence, indent=2))
            self.update(device=evidence, model_steps=model.num_timesteps, run=str(self.root))
            config = deepcopy(yaml.safe_load(self.config_path.read_text()))
            config['environment']['headless'] = False
            env = DemoEnv(self, config, self.map_path, self.root, residual_enabled=True)
            while True:
                try:
                    self.update(state='STARTING', message='Opening Gazebo and fixed-path Nav2')
                    env.residual_enabled = self.mode == 'sac'
                    self.update(active_mode=self.mode)
                    with self.lock:
                        self.trace.clear()
                    observation, info = env.reset(seed=101)
                    self.update(state='RUNNING', message='PI + SAC running; no learning' if self.mode == 'sac'
                                else 'PI only running; residual disabled in effort_drive')
                    while True:
                        command = self.next_command()
                        if command == 'close':
                            raise CloseDemo()
                        self.check_restart(command)
                        if command == 'stop':
                            self.update(state='STOPPING', message='Canceling navigation and clearing residual')
                            env._cancel()
                            if not env.result_future.done():
                                raise RuntimeError('Navigation cancellation was not confirmed; closing simulation')
                            self.update(state='STOPPED', message='Navigation canceled; residual cleared')
                            break
                        action = self.action_for_mode(model, observation)
                        observation, reward, terminated, truncated, info = env.step(action)
                        self.telemetry(info)
                        if terminated or truncated:
                            self.update(state=info['outcome'], message=info.get('reason') or info['outcome'])
                            (self.root/'last_result.json').write_text(json.dumps(info, indent=2))
                            break
                    # Keep the GUI and ROS reader alive after stop/success/failure.
                    while True:
                        command = self.next_command()
                        if command == 'close':
                            raise CloseDemo()
                        self.check_restart(command)
                        if any(child.poll() is not None for child in env.children):
                            raise CloseDemo()
                        env.bridge.command([0, 0])
                        self.idle_telemetry(env)
                        time.sleep(0.1)
                except RestartDemo as command:
                    self.mode = command.mode or self.mode
                    self.auto_start = True
                    self.update(state='STARTING', message='Resetting owned simulation to the starting pose')
                    env._cleanup()
                    continue
        except CloseDemo:
            self.update(state='CLOSED', message='Demo closed')
        except Exception as exc:
            self.update(state='ERROR', message=str(exc))
            (self.root/'error.json').write_text(json.dumps({'reason': str(exc)}, indent=2))
        finally:
            if env is not None:
                env.close()
            self.update(cleanup_complete=True)


def make_handler(control):
    class Handler(BaseHTTPRequestHandler):
        def send_json(self, status, data):
            self.send_response(status)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            self.wfile.write(json.dumps(data).encode())

        def do_GET(self):
            if self.path == '/api/demo':
                self.send_json(200, control.snapshot())
            else:
                self.send_error(404)

        def do_POST(self):
            if self.path != '/api/demo/command':
                self.send_error(404)
                return
            origin = self.headers.get('Origin')
            if origin and origin not in ('http://localhost:8766', 'http://127.0.0.1:8766'):
                self.send_error(403)
                return
            try:
                length = int(self.headers.get('Content-Length', 0))
                if not 0 < length <= 256 or not self.headers.get('Content-Type', '').startswith('application/json'):
                    raise ValueError('Small JSON command required')
                command = json.loads(self.rfile.read(length))['command']
                control.command(command)
                self.send_json(202, {'accepted': command})
            except (ValueError, KeyError, TypeError) as exc:
                self.send_json(400, {'error': str(exc)})

        def log_message(self, *args):
            pass
    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True, type=Path)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--map', required=True, type=Path)
    args = parser.parse_args()
    for path in (args.model, args.config, args.map):
        if not path.is_file():
            parser.error(f'Missing file: {path}')
    control = Control(args.model.resolve(), args.config.resolve(), args.map.resolve())
    server = ThreadingHTTPServer(('127.0.0.1', 8770), make_handler(control))
    worker = Thread(target=control.run, name='nino-demo', daemon=True)
    worker.start()
    print(f'Demo: {control.root}; web: http://localhost:8766/demo', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        if worker.is_alive():
            control.commands.put('close')
    finally:
        worker.join(timeout=60)
        server.server_close()


if __name__ == '__main__':
    main()
