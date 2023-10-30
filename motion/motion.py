''''Module for motion representation.'''
import torch
import numpy as np

from Motion import BVH
from Motion.Animation import Animation
from Motion.transforms import repr6d2quat
from Motion.Quaternions import Quaternions
from Motion.AnimationStructure import children_list, get_sorted_order
from motion.static_motion import StaticData
from motion.utils.transformations import expand_topology_edges


class DynamicData:
    '''Motion dynamic data representation including all joints rotations / locations.'''
    def __init__(self, motion: torch.Tensor,
     motion_statics: StaticData, use_velocity=False):
        # Shape is B  x K x J x T = batch x channels x joints x frames
        self.motion = motion.clone()
        self.motion_statics = motion_statics

        # self.assert_shape_is_right()

        self.use_velocity = use_velocity

    def _assert_shape_is_right(self):
        foot_contact_joints = self.motion_statics.foot_number \
                                if self.motion_statics.enable_foot_contact else 0
        global_position_joint = 1 if self.motion_statics.enable_global_position else 0

        assert self.motion.shape[-3] == self.motion_statics.n_channels
        assert self.motion.shape[-2] == len(self.motion_statics.parents) + \
                                             global_position_joint + foot_contact_joints

    @classmethod
    def init_from_bvh(cls, bvf_filepath: str, character_name: str,
                      enable_global_position=False,
                      enable_foot_contact=False,
                      use_velocity=False,
                      rotation_representation='quaternion'):

        animation, names, _ = BVH.load(bvf_filepath)

        motion_statics = StaticData(animation.parents,
                            animation.offsets, names, character_name,
                            enable_global_position=enable_global_position,
                            enable_foot_contact=enable_foot_contact,
                            rotation_representation=rotation_representation)

        return cls(animation.rotations.qs, motion_statics , use_velocity=use_velocity)

    def sub_motion(self, motion):
        return self.__class__(motion, self.motion_statics, use_velocity=self.use_velocity)

    def __iter__(self):
        if self.motion.ndim == 4:
            iterator = (self.sub_motion(motion) for motion in self.motion).__iter__()
        elif self.motion.ndim == 3:
            iterator = [self.sub_motion(self.motion)].__iter__()

        return iterator

    def __getitem__(self, slice_val):
        return self.sub_motion(self.motion[slice_val])

    @property
    def shape(self):
        return self.motion.shape

    @property
    def n_frames(self):
        return self.motion.shape[-1]

    @property
    def n_channels(self):
        return self.motion.shape[-3]

    @property
    def n_joints(self):
        return len(self.motion_statics .names)

    @property
    def edge_rotations(self) -> torch.Tensor:
        return self.motion[..., :self.n_joints, :].detach().cpu()
        # Return only joints representing motion, maybe having a batch dim

    def foot_contact(self) -> torch.Tensor:
        return [[{foot: motion[0, self.n_joints + 1 + idx, frame].numpy().item()
                for idx, foot in enumerate(self.motion_statics .foot_names)}
                for frame in range(self.n_frames)]
                for motion in self.motion]

    @property
    def root_location(self) -> torch.Tensor:
        '''drop the 4th item in the position tensor'''
        location = self.motion[..., :3, self.n_joints, :].detach().cpu()
        location = np.cumsum(location, axis=1) if self.use_velocity else location

        return location  # K x T

    def un_normalise(self, mean: torch.Tensor, std: torch.Tensor):
        return self.sub_motion(self.motion * std + mean)

    def sample_frames(self, frames_indexes: [int]):
        return self.sub_motion(self.motion[..., frames_indexes])

    def _basic_anim(self):
        offsets = self.motion_statics.offsets

        positions = np.repeat(offsets[np.newaxis], self.n_frames, axis=0)
        positions[:, 0] = self.root_location.transpose(0, 1)

        orients = Quaternions.id(self.n_joints)
        rotations = Quaternions(self.edge_rotations.permute(2, 1, 0).numpy())

        if rotations.shape[-1] == 6:  # repr6d
            rotations = repr6d2quat(rotations)

        anim_edges = Animation(rotations, positions, orients, offsets, self.motion_statics .parents)

        return anim_edges

    def move_rotation_values_to_parents(self, anim_exp):
        children_all_joints = children_list(anim_exp.parents)
        for idx, children_one_joint in enumerate(children_all_joints[1:]):
            parent_idx = idx + 1
            if len(children_one_joint) > 0:  # not leaf
                assert len(children_one_joint) == 1 or \
                         (anim_exp.offsets[children_one_joint] == np.zeros(3)).all() and (
                                anim_exp.rotations[:, children_one_joint] ==
                                Quaternions.id((self.n_frames, 1))).all()
                anim_exp.rotations[:, parent_idx] = anim_exp.rotations[:, children_one_joint[0]]
            else:
                anim_exp.rotations[:, parent_idx] = Quaternions.id((self.n_frames))

        return anim_exp

    def to_anim(self):
        anim_edges = self._basic_anim()

        sorted_order = get_sorted_order(anim_edges.parents)
        anim_edges_sorted = anim_edges[:, sorted_order]
        names_sorted = self.motion_statics.names[sorted_order]

        anim_exp, _, names_exp, _ = expand_topology_edges(anim_edges_sorted,
                                                          names=names_sorted,
                                                          nearest_joint_ratio=1)
        anim_exp = self.move_rotation_values_to_parents(anim_exp)

        return anim_exp, names_exp
