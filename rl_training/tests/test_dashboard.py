import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dashboard import Dashboard, JobReader, Lines, last_record


def test_incremental_reader_holds_partial_utf8_and_does_not_duplicate(tmp_path):
    path = tmp_path/'log'
    reader = Lines()
    path.write_bytes(b'one\n'+bytes([0xc4]))
    assert reader.read(path) == ['one']
    with path.open('ab') as stream:
        stream.write(bytes([0x91])+b'\n')
    assert reader.read(path) == ['đ']
    assert reader.read(path) == []
    path.write_bytes(b'x\n')
    assert reader.read(path) == ['x']


def test_tail_skips_a_partial_json_record(tmp_path):
    path = tmp_path/'steps.jsonl'
    path.write_text('{"outcome":"SUCCEEDED"}\n{"outc')
    assert last_record(path)['outcome'] == 'SUCCEEDED'


def setup_job(tmp_path):
    job = tmp_path/'long_test'
    run = tmp_path/'train_test'
    job.mkdir()
    run.mkdir()
    (job/'status.json').write_text(json.dumps(dict(state='TRAIN', pid=None, child_pid=None,
        target_steps=50000000, phases={'train': str(run)})))
    (job/'train.log').write_text('Target: 50000000; restored: 10000; remaining: 49990000\n'
        '---\n| total_timesteps | 10005 |\n| critic_loss | 1.2 |\n---\n')
    (run/'monitor.csv').write_text('#{}\nr,l,t\n25,5,10\n')
    previous = run/'episode_00001'
    previous.mkdir()
    (previous/'steps.jsonl').write_text('{"outcome":"SUCCEEDED"}\n')
    latest = run/'episode_00002'
    latest.mkdir()
    (latest/'steps.jsonl').write_text('{"simulation_time":0.1,"outcome":null}\n'*2)
    return job, run, latest


def test_live_steps_are_restored_plus_new_and_dead_worker_is_not_running(tmp_path):
    job, run, latest = setup_job(tmp_path)
    reader = JobReader(tmp_path, job)
    first = reader.snapshot()
    assert first['steps'] == 10007
    assert first['state'] == 'STOPPED' and first['reported_state'] == 'TRAIN'
    assert first['recent_outcomes']['SUCCEEDED'] == 1
    assert len(first['history']) == 1
    assert first['new_steps_per_s'] == pytest.approx(0.7)
    assert reader.snapshot()['steps'] == 10007
    with (latest/'steps.jsonl').open('a') as stream:
        stream.write('{"simulation_time":0.3,"outcome":null}\n')
    assert reader.snapshot()['steps'] == 10008
    assert reader.snapshot()['completed_episodes'] == 1


def test_terminal_episode_is_not_counted_twice(tmp_path):
    job, run, latest = setup_job(tmp_path)
    (latest/'steps.jsonl').write_text('{"outcome":"FAILED"}\n'*2)
    with (run/'monitor.csv').open('a') as stream:
        stream.write('-30,2,20\n')
    snapshot = JobReader(tmp_path, job).snapshot()
    assert snapshot['steps'] == 10007
    assert snapshot['completed_episodes'] == 2


def test_starting_job_without_run_is_supported(tmp_path):
    folder = tmp_path/'long_starting'
    folder.mkdir()
    (folder/'status.json').write_text('{"state":"STARTING","phases":{}}')
    snapshot = JobReader(tmp_path, folder).snapshot()
    assert snapshot['run'] is None and snapshot['steps'] == 0


@pytest.mark.parametrize('job', ['../etc', 'long_../../etc', '/etc', 'unknown'])
def test_api_refuses_path_traversal_and_arbitrary_directories(tmp_path, job):
    with pytest.raises(ValueError):
        Dashboard(tmp_path).snapshot(job)


def test_status_cannot_point_outside_runs(tmp_path):
    job, run, _ = setup_job(tmp_path)
    (job/'status.json').write_text(json.dumps({'phases': {'train': str(tmp_path.parent)}}))
    with pytest.raises(ValueError, match='outside'):
        JobReader(tmp_path, job).snapshot()


def test_live_speed_uses_current_wall_time_not_last_finished_episode(monkeypatch, tmp_path):
    import dashboard
    job, run, _ = setup_job(tmp_path)
    (run/'monitor.csv').write_text('#{"t_start":900}\nr,l,t\n25,5,10\n')
    monkeypatch.setattr(dashboard.time, 'time', lambda: 1000)
    monkeypatch.setattr(dashboard, 'process_matches', lambda *args: True)
    snapshot = JobReader(tmp_path, job).snapshot()
    assert snapshot['new_steps_per_s'] == pytest.approx(7/100)


def test_http_serves_dashboard_and_rejects_unknown_paths(tmp_path):
    from threading import Thread
    from http.server import ThreadingHTTPServer
    from urllib.request import urlopen
    from urllib.error import HTTPError
    from dashboard import make_handler
    setup_job(tmp_path)
    server = ThreadingHTTPServer(('127.0.0.1', 0), make_handler(Dashboard(tmp_path)))
    worker = Thread(target=server.serve_forever, daemon=True)
    worker.start()
    url = f'http://127.0.0.1:{server.server_port}'
    try:
        with urlopen(url+'/') as response:
            assert response.status == 200
            assert b'TRAINING OBSERVATORY' in response.read()
        with urlopen(url+'/api/state?job=long_test') as response:
            assert json.load(response)['steps'] == 10007
        with pytest.raises(HTTPError) as fault:
            urlopen(url+'/api/state?job=../etc')
        assert fault.value.code == 400
        with pytest.raises(HTTPError) as fault:
            urlopen(url+'/train.py')
        assert fault.value.code == 404
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=2)


def test_dashboard_accepts_new_training_segments_without_double_counting(tmp_path):
    job, run, _ = setup_job(tmp_path)
    reader = JobReader(tmp_path, job)
    assert reader.snapshot()['steps'] == 10007
    next_run = tmp_path/'train_next'
    next_run.mkdir()
    (next_run/'monitor.csv').write_text('#{}\nr,l,t\n10,2,10\n')
    (job/'train_20000.log').write_text('Target: 20000; restored: 10007; remaining: 9993\n')
    state = dict(state='TRAIN', completed_training_steps=10007, train_log='train_20000.log',
                 phases={'train': str(next_run)})
    (job/'status.json').write_text(json.dumps(state))
    snapshot = reader.snapshot()
    assert snapshot['steps'] == 10009
    assert snapshot['completed_episodes'] == 1
    assert reader.snapshot()['steps'] == 10009


def test_static_report_access_is_limited_to_report_directories(tmp_path):
    dashboard = Dashboard(tmp_path)
    report = tmp_path/'comparison_test'
    report.mkdir()
    (report/'report.html').write_text('ok')
    assert dashboard.report_file('comparison_test', 'report.html') == b'ok'
    with pytest.raises(ValueError):
        dashboard.report_file('comparison_test', '../train.py')
    with pytest.raises(ValueError):
        dashboard.report_file('train_test', 'metadata.json')
