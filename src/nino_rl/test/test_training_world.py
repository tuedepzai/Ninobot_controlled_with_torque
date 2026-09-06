from math import tan
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np


WORLD = (
    Path(__file__).parents[2]
    / "nino_description"
    / "worlds"
    / "long_hall.sdf"
)


def test_training_world_has_dense_cables_and_clear_spawn_zone():
    root = ET.parse(WORLD).getroot()
    cable_model = root.find(".//model[@name='cable_bumps']")
    assert cable_model is not None
    collisions = cable_model.findall(".//collision")
    assert len(collisions) == 29
    assert len({collision.attrib["name"] for collision in collisions}) == 29

    for collision in collisions:
        pose = [float(value) for value in collision.findtext("pose").split()]
        radius = float(collision.findtext("geometry/cylinder/radius"))
        length = float(collision.findtext("geometry/cylinder/length"))
        x_center, y_center, z_center, _, _, yaw = pose
        assert np.isclose(y_center, 0.0)
        assert np.isclose(z_center, radius)
        assert 0.008 <= radius <= 0.022
        assert length >= 4.0

        # At the spawn pad's Y edges, the angled cable must still stay outside
        # its X interval [-0.65, 0.65].
        half_pad_y = 0.50
        x_variation = abs(tan(yaw) * half_pad_y)
        if x_center > 0.0:
            assert x_center - x_variation > 0.65
        else:
            assert x_center + x_variation < -0.65

    marker = root.find(".//model[@name='spawn_clear_zone']")
    assert marker is not None
    assert marker.find(".//collision") is None

