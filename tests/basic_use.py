import torch
import pytest
import numpy as np

from motion.motion import DynamicData
from motion.static_motion import StaticData


EXAMPLE_EDGE_ROT = 'tests/tests_data/edge_rot.npy'
EXAMPLE_MOTION_PATH = 'tests/tests_data/motion_1304.npy'
EXAMPLE_CHARACTER = 'jasper'

@pytest.fixture
def edge_rot():
    return np.load(EXAMPLE_EDGE_ROT, allow_pickle=True).item()

@pytest.fixture
def raw_motion():
    return np.load(EXAMPLE_MOTION_PATH, allow_pickle=True)[np.newaxis, ...]

@pytest.fixture
def statics(edge_rot):
    return StaticData.init_from_motion(edge_rot, character_name=EXAMPLE_CHARACTER,
                                            n_channels=4,
                                            enable_global_position=True,
                                            enable_foot_contact=True)
    

def test_static_init_from_bvh(statics):
    assert len(statics.parents) == 20


def test_dynamic_init(statics, raw_motion):
    # motion_data, normalisation_data = motion_from_raw(args, motion_data_raw, motion_statics)
    dynamics = DynamicData(torch.from_numpy(raw_motion.transpose(0, 2, 1, 3)), statics, use_velocity=False)

    assert dynamics.n_joints == len(statics.parents)