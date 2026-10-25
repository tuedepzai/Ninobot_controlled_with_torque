from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from demo_server import Control, RestartDemo


def test_pi_mode_does_not_call_the_sac_network():
    control = object.__new__(Control)
    control.mode = 'pi'
    def predict(*args, **kwargs):
        pytest.fail('SAC must not be used in PI mode')
    assert np.array_equal(control.action_for_mode(SimpleNamespace(predict=predict), None), [0, 0])


def test_sac_mode_uses_deterministic_action():
    control = object.__new__(Control)
    control.mode = 'sac'
    def predict(observation, deterministic):
        assert deterministic is True
        return np.array([0.2, -0.3]), None
    assert np.allclose(control.action_for_mode(SimpleNamespace(predict=predict), None), [0.2, -0.3])


@pytest.mark.parametrize('command,mode', [('test_pi', 'pi'), ('test_sac', 'sac'), ('restart', None)])
def test_mode_change_requires_a_fresh_episode(command, mode):
    control = object.__new__(Control)
    with pytest.raises(RestartDemo) as request:
        control.check_restart(command)
    assert request.value.mode == mode


def test_web_command_allowlist_and_cross_origin_guard():
    from collections import deque
    from queue import Queue
    from threading import Lock, Thread
    from http.server import ThreadingHTTPServer
    from urllib.request import Request, urlopen
    from urllib.error import HTTPError
    import json
    from demo_server import make_handler
    control = object.__new__(Control)
    control.lock, control.commands, control.trace = Lock(), Queue(), deque()
    control.data = {'state': 'READY'}
    server = ThreadingHTTPServer(('127.0.0.1', 0), make_handler(control))
    worker = Thread(target=server.serve_forever, daemon=True)
    worker.start()
    url = f'http://127.0.0.1:{server.server_port}/api/demo/command'
    try:
        request = Request(url, data=b'{"command":"test_pi"}',
                          headers={'Content-Type': 'application/json'})
        with urlopen(request) as response:
            assert response.status == 202
            assert json.load(response)['accepted'] == 'test_pi'
        assert control.next_command() == 'test_pi'
        request = Request(url, data=b'{"command":"test_sac"}',
                          headers={'Content-Type': 'application/json', 'Origin': 'https://example.com'})
        with pytest.raises(HTTPError) as fault:
            urlopen(request)
        assert fault.value.code == 403
        assert control.commands.empty()
        request = Request(url, data=b'{"command":"set_torque_1000"}',
                          headers={'Content-Type': 'application/json'})
        with pytest.raises(HTTPError) as fault:
            urlopen(request)
        assert fault.value.code == 400
        assert control.commands.empty()
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=2)
