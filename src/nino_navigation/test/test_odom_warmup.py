from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from odom_warmup import OdomWarmup


def test_one_recent_sample_does_not_mean_ready():
    stream = OdomWarmup()
    stream.observe(1.0, 100)
    assert not stream.ready(1.01)


def test_requires_continuous_advancing_stream():
    stream = OdomWarmup()
    for i in range(51):
        stream.observe(i*0.02, i)
    assert stream.ready(1.0)
    assert not stream.ready(1.3)


def test_gap_restarts_warmup_instead_of_starting_episode():
    stream = OdomWarmup()
    for i in range(51):
        stream.observe(i*0.02, i)
    stream.observe(1.5, 100)
    assert not stream.ready(1.5)


def test_repeated_stamp_does_not_refresh_stream():
    stream = OdomWarmup()
    stream.observe(0, 10)
    assert not stream.observe(1, 10)
    assert stream.received == 0
    assert not stream.ready(1)


def test_clock_reset_requires_new_warmup():
    stream = OdomWarmup()
    for i in range(51):
        stream.observe(i*0.02, i+100)
    stream.observe(1.02, 0)
    assert not stream.ready(1.02)
