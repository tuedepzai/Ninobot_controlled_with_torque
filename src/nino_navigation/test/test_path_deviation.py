import math
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from path_deviation import cross_track_error


def test_distance_to_segment_interior_not_nearest_vertex():
    assert cross_track_error(5, 0.2, ((0, 0), (10, 0))) == pytest.approx(0.2)


def test_endpoints_and_duplicate_points():
    assert cross_track_error(-1, 0, ((0, 0), (0, 0), (1, 0))) == 1
    assert cross_track_error(2, 0, ((0, 0), (1, 0))) == 1
    assert cross_track_error(1, 1, ((0, 0),)) == pytest.approx(math.sqrt(2))


def test_corner_and_threshold_boundary():
    assert cross_track_error(0.9, 0.5, ((0, 0), (1, 0), (1, 1))) == pytest.approx(0.1)
    assert not cross_track_error(0.5, 0.2, ((0, 0), (1, 0))) > 0.2
    assert cross_track_error(0.5, 0.201, ((0, 0), (1, 0))) > 0.2


@pytest.mark.parametrize('x,y,points', [
    (0, 0, ()), (float('nan'), 0, ((0, 0),)), (0, 0, ((float('inf'), 0),))])
def test_invalid_input(x, y, points):
    with pytest.raises(ValueError):
        cross_track_error(x, y, points)
