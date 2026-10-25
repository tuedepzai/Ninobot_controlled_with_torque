#!/usr/bin/env python3
"""Read-only local dashboard. No ROS node, Torch import, or training controls."""
import argparse
from collections import deque
import csv
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
import mimetypes
from pathlib import Path
import re
import shutil
import subprocess
from threading import Lock
import time
from urllib.parse import parse_qs, urlsplit
from urllib.request import Request, urlopen


class Lines:
    """Incremental UTF-8 reader: incomplete writes wait for the next poll."""
    def __init__(self):
        self.offset, self.pending, self.identity = 0, b'', None

    def read(self, path):
        try:
            stat = path.stat()
        except FileNotFoundError:
            return []
        identity = (stat.st_dev, stat.st_ino)
        if identity != self.identity or stat.st_size < self.offset:
            self.offset, self.pending, self.identity = 0, b'', identity
        with path.open('rb') as stream:
            stream.seek(self.offset)
            content = stream.read(4*1024*1024)
        self.offset += len(content)
        parts = (self.pending+content).split(b'\n')
        self.pending = parts.pop()
        return [p.decode('utf-8', errors='replace').rstrip('\r') for p in parts]


def last_record(path):
    try:
        with path.open('rb') as stream:
            stream.seek(0, 2)
            stream.seek(max(0, stream.tell()-65536))
            lines = stream.read().splitlines()
        for line in reversed(lines):
            try:
                record = json.loads(line)
                if isinstance(record, dict):
                    return record
            except (ValueError, UnicodeDecodeError):
                pass
    except FileNotFoundError:
        pass
    return None


def number(value):
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (ValueError, TypeError):
        return None


def process_matches(pid, marker, required=None):
    try:
        command = Path(f'/proc/{int(pid)}/cmdline').read_bytes().decode(errors='replace')
        return marker in command and (required is None or str(required) in command.split('\0'))
    except (OSError, ValueError, TypeError):
        return False


class JobReader:
    def __init__(self, root, job):
        self.root, self.job = root, job
        self.log = Lines()
        self.monitor = Lines()
        self.steps = Lines()
        self.run = self.episode = None
        self.metrics, self.block = {}, {}
        self.logs = deque(maxlen=90)
        self.history = deque(maxlen=600)
        self.rewards = deque(maxlen=600)
        self.trace = deque(maxlen=300)
        self.completed, self.completed_steps, self.current_steps, self.restored = 0, 0, 0, 0
        self.scan_at, self.folders = 0, []
        self.outcomes = {}
        self.environment = {}
        self.started = None
        self.active_log = None
        self.device_info = None

    def safe_run(self, raw):
        path = Path(raw).resolve()
        if not path.is_relative_to(self.root) or not path.is_dir():
            raise ValueError('Run is outside the configured runs directory')
        return path

    def snapshot(self):
        status = json.loads((self.job/'status.json').read_text())
        raw = status.get('phases', {}).get('train')
        run = self.safe_run(raw) if raw else None
        log_name = status.get('train_log', 'train.log')
        if Path(log_name).name != log_name:
            raise ValueError('Invalid training log filename')
        if run != self.run or log_name != self.active_log:
            self.run, self.active_log = run, log_name
            self.log, self.monitor, self.steps = Lines(), Lines(), Lines()
            self.metrics, self.block, self.outcomes = {}, {}, {}
            self.history.clear()
            self.rewards.clear()
            self.trace.clear()
            self.logs.clear()
            self.completed = self.completed_steps = self.current_steps = 0
            self.restored = status.get('completed_training_steps', 0)
            self.episode, self.started, self.device_info = None, None, None
            self.scan_at, self.folders, self.environment = 0, [], {}
        log_path = self.job/log_name
        for line in self.log.read(log_path):
            self.logs.append(line)
            restored = re.search(r'restored: (\d+)', line)
            if restored:
                self.restored = int(restored[1])
            if line.startswith('---') and 'total_timesteps' in self.block:
                self.metrics = self.block
                self.history.append(self.block.copy())
                self.block = {}
            elif line.startswith('|'):
                parts = [s.strip() for s in line.split('|')]
                if len(parts) >= 4 and number(parts[2]) is not None:
                    self.block[parts[1]] = number(parts[2])
        if raw:
            config = run/'config.yaml'
            if config.is_file() and not self.environment:
                import yaml
                self.environment = yaml.safe_load(config.read_text()).get('environment', {})
            for line in self.monitor.read(run/'monitor.csv'):
                if line.startswith('#'):
                    try:
                        self.started = number(json.loads(line[1:]).get('t_start'))
                    except ValueError:
                        pass
                    continue
                if not line or line.startswith('r,'):
                    continue
                cells = list(csv.reader([line]))[0]
                if len(cells) == 3 and all(number(c) is not None for c in cells):
                    reward, length, elapsed = map(float, cells)
                    self.completed += 1
                    self.completed_steps += int(length)
                    self.rewards.append(dict(episode=self.completed, reward=reward, steps=int(length), elapsed=elapsed))
            if time.monotonic()-self.scan_at > 5:
                # Only inspect tails of the most recent 100 attempts, not all step files.
                self.folders = sorted(run.glob('episode_*'))[-100:]
                self.scan_at = time.monotonic()
            for folder in self.folders:
                if folder.name not in self.outcomes:
                    row = last_record(folder/'steps.jsonl')
                    if row and row.get('outcome'):
                        self.outcomes[folder.name] = row['outcome']
            names = {p.name for p in self.folders}
            self.outcomes = {k: v for k, v in self.outcomes.items() if k in names}
            if self.folders:
                latest = self.folders[-1]
                if latest != self.episode:
                    self.episode, self.steps, self.current_steps = latest, Lines(), 0
                    self.trace.clear()
                for line in self.steps.read(latest/'steps.jsonl'):
                    try:
                        row = json.loads(line)
                        self.current_steps += 1
                        self.trace.append({k: row.get(k) for k in (
                            'simulation_time', 'cross_track', 'progress', 'remaining',
                            'acceleration', 'residual_applied', 'total_torque', 'outcome')})
                    except ValueError:
                        continue
        latest = self.trace[-1] if self.trace else {}
        partial = self.current_steps if not latest.get('outcome') else 0
        progress = max(status.get('completed_training_steps', 0), self.metrics.get('total_timesteps', 0),
                       self.restored+self.completed_steps+partial)
        expected = status.get('state', 'UNKNOWN')
        alive = process_matches(status.get('pid'), 'long_run.py', self.job)
        child_alive = alive and process_matches(status.get('child_pid'), 'train.py')
        terminal = expected in ('COMPLETE', 'FAILED', 'INTERRUPTED')
        observed = expected if terminal or alive else 'STOPPED'
        candidates = [log_path]
        if self.run:
            candidates.append(self.run/'monitor.csv')
        if self.episode:
            candidates.append(self.episode/'steps.jsonl')
        modified = max((p.stat().st_mtime for p in candidates if p.exists()), default=0)
        disk = shutil.disk_usage(self.root)
        samples = list(self.rewards)
        elapsed = samples[-1]['elapsed'] if samples else self.metrics.get('time_elapsed', 0)
        if self.started is not None and alive and expected == 'TRAIN':
            elapsed = max(elapsed, time.time()-self.started)
        new_steps = max(0, progress-self.restored)
        speed = new_steps/elapsed if elapsed and new_steps else None
        target = status.get('target_steps', 0)
        counts = {value: list(self.outcomes.values()).count(value) for value in ('SUCCEEDED', 'FAILED', 'TIME_LIMIT')}
        watched = [(log_path, self.log)]
        if self.run is not None:
            watched.append((self.run/'monitor.csv', self.monitor))
            try:
                self.device_info = json.loads((self.run/'device_verification.json').read_text())
            except (OSError, ValueError):
                pass
        try:
            evaluations = json.loads((self.job/'evaluations.json').read_text())
        except (OSError, ValueError):
            evaluations = []
        return dict(job=self.job.name, run=self.run.name if self.run else None,
                    state=observed, reported_state=expected, worker_alive=alive, trainer_alive=child_alive,
                    now=time.time(), log_age_s=max(0, time.time()-modified), reason=status.get('reason'),
                    target=target, steps=int(progress), restored=self.restored, steps_approximate=True,
                    completed_episodes=self.completed, new_steps_per_s=speed,
                    eta_s=(target-progress)/speed if speed and alive and expected == 'TRAIN' else None,
                    metrics=self.metrics, history=list(self.history), rewards=samples,
                    recent_outcomes=counts, recent_outcome_count=sum(counts.values()),
                    episode=self.episode.name if self.episode else None, trace=list(self.trace),
                    logs=list(self.logs), disk_free_gb=disk.free/1024**3,
                    disk_used_fraction=disk.used/disk.total,
                    deviation_threshold_m=self.environment.get('max_path_deviation'),
                    residual_limit_nm=self.environment.get('residual_limit'),
                    device_verification=self.device_info, evaluations=evaluations,
                    next_evaluation_step=status.get('next_evaluation_step'),
                    eval_every=status.get('eval_every'),
                    catching_up=any(p.exists() and reader.offset < p.stat().st_size for p, reader in watched),
                    last_report=status.get('phases', {}).get('report'))


class Dashboard:
    def __init__(self, root):
        self.root = root.resolve()
        self.readers, self.cached = {}, {}
        self.lock = Lock()

    def jobs(self):
        result = []
        for path in sorted(self.root.glob('long_*/status.json'), key=lambda p: p.stat().st_mtime, reverse=True):
            try:
                state = json.loads(path.read_text())
                result.append(dict(id=path.parent.name, state=state.get('state'),
                                   alive=process_matches(state.get('pid'), 'long_run.py', path.parent)))
            except (ValueError, OSError):
                continue
        return result

    def snapshot(self, job):
        if not re.fullmatch(r'long_[A-Za-z0-9_-]+', job):
            raise ValueError('Invalid job ID')
        folder = (self.root/job).resolve()
        if not folder.is_relative_to(self.root) or not (folder/'status.json').is_file():
            raise ValueError('Unknown job')
        with self.lock:
            cached = self.cached.get(job)
            if cached and time.monotonic()-cached[0] < 1:
                return cached[1]
            reader = self.readers.setdefault(job, JobReader(self.root, folder))
            snapshot = reader.snapshot()
            self.cached[job] = time.monotonic(), snapshot
            return snapshot

    def report_file(self, report, name):
        if not re.fullmatch(r'comparison_[A-Za-z0-9_-]+', report) or Path(name).name != name:
            raise ValueError('Invalid report path')
        path = (self.root/report/name).resolve()
        if not path.is_relative_to(self.root) or path.suffix not in ('.html', '.svg', '.png', '.json', '.csv', '.md'):
            raise ValueError('Invalid report file')
        return path.read_bytes()


def make_handler(dashboard, backend_container=None):
    class Handler(BaseHTTPRequestHandler):
        def demo_request(self, body=None):
            target = 'http://127.0.0.1:8770'+urlsplit(self.path).path
            if backend_container:
                script = ('import urllib.request,sys; '
                          'body=sys.stdin.buffer.read() if sys.argv[2]=="POST" else None; '
                          'req=urllib.request.Request(sys.argv[1],data=body,headers={"Content-Type":"application/json"},method=sys.argv[2]); '
                          'sys.stdout.buffer.write(urllib.request.urlopen(req,timeout=5).read())')
                response = subprocess.run(['docker', 'exec', '-i', backend_container, 'python3', '-c',
                    script, target, 'POST' if body is not None else 'GET'], input=body or b'',
                    capture_output=True, timeout=8, check=True)
                return response.stdout
            request = Request(target, data=body, headers={'Content-Type': 'application/json'},
                              method='POST' if body is not None else 'GET')
            return urlopen(request, timeout=5).read()

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
                body = self.demo_request(self.rfile.read(length))
                self.send_response(202)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Cache-Control', 'no-store')
                self.end_headers()
                self.wfile.write(body)
            except (ValueError, OSError, subprocess.SubprocessError) as exc:
                self.send_error(400, str(exc))

        def do_GET(self):
            url = urlsplit(self.path)
            try:
                if url.path == '/api/demo':
                    body, content_type = self.demo_request(), 'application/json'
                elif url.path == '/demo':
                    body = (Path(__file__).parent/'dashboard'/'demo.html').read_bytes()
                    content_type = 'text/html; charset=utf-8'
                elif backend_container and (url.path in ('/api/jobs', '/api/state') or url.path.startswith('/report/')):
                    # Local relay when Docker Desktop's host network is not
                    # forwarded to Windows localhost. No Docker/WSLg changes.
                    script = ('import urllib.request,sys; '
                              'sys.stdout.buffer.write(urllib.request.urlopen(sys.argv[1],timeout=5).read())')
                    result = subprocess.run(['docker', 'exec', backend_container, 'python3', '-c', script,
                        'http://127.0.0.1:8765'+self.path], capture_output=True, timeout=8, check=True)
                    body = result.stdout
                    content_type = 'application/json' if url.path.startswith('/api/') else mimetypes.guess_type(url.path)[0] or 'application/octet-stream'
                elif url.path == '/api/jobs':
                    body, content_type = json.dumps(dashboard.jobs()).encode(), 'application/json'
                elif url.path == '/api/state':
                    job = parse_qs(url.query).get('job', [''])[0]
                    body, content_type = json.dumps(dashboard.snapshot(job), allow_nan=False).encode(), 'application/json'
                elif url.path.startswith('/report/'):
                    parts = url.path.split('/')
                    if len(parts) != 4:
                        raise ValueError('Invalid report path')
                    body = dashboard.report_file(parts[2], parts[3])
                    content_type = mimetypes.guess_type(url.path)[0] or 'application/octet-stream'
                elif url.path == '/':
                    body = (Path(__file__).parent/'dashboard'/'index.html').read_bytes()
                    content_type = 'text/html; charset=utf-8'
                else:
                    self.send_error(404)
                    return
                self.send_response(200)
                self.send_header('Content-Type', content_type)
                self.send_header('Cache-Control', 'no-store')
                self.send_header('X-Content-Type-Options', 'nosniff')
                self.end_headers()
                self.wfile.write(body)
            except (ValueError, OSError, subprocess.SubprocessError) as exc:
                self.send_error(400, str(exc))

        def log_message(self, *args):
            pass
    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', type=Path, default=Path(__file__).resolve().parent/'runs')
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--backend-container', help='Relay API from the dashboard running inside this container')
    args = parser.parse_args()
    server = ThreadingHTTPServer(('127.0.0.1', args.port), make_handler(Dashboard(args.runs), args.backend_container))
    print(f'Nino training dashboard: http://localhost:{args.port}', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
