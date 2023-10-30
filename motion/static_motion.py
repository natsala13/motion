'''Class representing a motion skeleton and any static data of a motion.'''
import functools

import numpy as np

from Motion import BVH
from motion.utils.utils import StaticConfig, EdgePoint


class StaticMotionOneHierarchyLevel:
    """Representing a static layer in the down sampling process including parents and feet indexes.

    Attributes:
        parents: List of parents  at specific hierarchy level - representing the skeleton.
        pooling_list: Dictionary from every joint in that hierarchy level to the next one.
        use_global_position: flag whether the model is using global position.
        feet_indices list on feet indices in case foot_contact is on o.w empty list.
    """
    def __init__(self, parents: [int], pooling_list: {int: [int]},
                  use_global_position: bool, feet_indices: [str]):
        self.parents = parents
        self.pooling_list = pooling_list
        self.use_global_position = use_global_position
        self.feet_indices = feet_indices

    @classmethod
    def keep_dim_layer(cls, parents: [int], *args, **kwargs):
        layer = cls(parents, None, *args, **kwargs)
        layer.pooling_list = {joint_idx: [joint_idx] for joint_idx in range(layer.edges_number)}

        return layer

    @property
    def foot_contact(self):
        return self.feet_indices is not None

    @property
    def edges_number(self):
        return len(self.parents) + self.use_global_position + len(self.feet_indices)

    @property
    def edges_number_after_pooling(self):
        return len(self.pooling_list)

    def fake_parents(self, parents=None):
        # In case debug is needed to be compared to older method.
        if not parents:
            parents = self.parents
        return parents + ([-2] if self.use_global_position else []) + \
                         ([(-3, index) for index in self.feet_indices])


class StaticData:
    '''Static data of a motion - the skeleton in addition to any data not depending on frame time'''
    def __init__(self, parents: [int], offsets: np.array, names: [str], character_name: str,
                 n_channels=4, enable_global_position=False, enable_foot_contact=False,
                 rotation_representation='quaternion'):
        self._offsets = offsets.copy() if offsets is not None else None
        self.names = names.copy() if names is not None else None
        self.config = StaticConfig(character_name)

        self.parents_list, self.skeletal_pooling_dist_1_edges = \
                                                        self.calculate_all_pooling_levels(parents)

        self.skeletal_pooling_dist_1 = [{edge[1]: [e[1] for e in pooling[edge]] for edge in pooling}
                                        for pooling in self.skeletal_pooling_dist_1_edges]

        self.skeletal_pooling_dist_0 = [{edge[1]: [pooling[edge][-1][1]] for edge in pooling}
                                        for pooling in self.skeletal_pooling_dist_1_edges]

        # Configurations
        self.__n_channels = n_channels

        self.enable_global_position = enable_global_position
        self.enable_foot_contact = enable_foot_contact

        if enable_global_position:
            self._enable_global_position()
        if enable_foot_contact:
            self._enable_foot_contact()
        if rotation_representation == 'repr6d':
            self._enable_repr6d()

    @classmethod
    def init_from_bvh(cls, bvf_filepath: str, *args, **kwargs):
        animation, names, _ = BVH.load(bvf_filepath)
        return cls(animation.parents, animation.offsets, names, *args, **kwargs)

    @classmethod
    def init_from_motion(cls, motion, **kwargs):
        offsets = np.concatenate([motion['offset_root'][np.newaxis, :], motion['offsets_no_root']])
        return cls(motion['parents_with_root'], offsets, motion['names_with_root'], **kwargs)

    @classmethod
    def init_joint_static(cls, joint, **kwargs):
        motion_statics = cls(joint.parents_list[-1], offsets=None,
                             names=joint.parents_list[-1], n_channels=3,
                            enable_foot_contact=False, rotation_representation=False, **kwargs)

        motion_statics.parents_list = joint.parents_list
        motion_statics.skeletal_pooling_dist_1 = joint.skeletal_pooling_dist_1
        motion_statics.skeletal_pooling_dist_0 = joint.skeletal_pooling_dist_0

        return motion_statics

    @property
    def parents(self):
        return self.parents_list[-1][:len(self.names)]

    @property
    def entire_motion(self) -> np.ndarray:
        raise NotImplementedError

    @property
    def offsets(self) -> np.ndarray:
        return self._offsets.copy() if self._offsets is not None else None

    @staticmethod
    def edge_list(parents: [int]) -> [EdgePoint]:
        return [EdgePoint(dst, src + 1) for src, dst in enumerate(parents[1:])]

    @property
    def n_channels(self) -> int:
        return self.__n_channels

    def _enable_repr6d(self):
        self.__n_channels = 6

    def _enable_marker4(self):
        self.__n_channels = 12
    # @n_channels.setter
    # def n_channels(self, val: int) -> None:
    #     self.__n_channels = val

    @property
    def character_name(self):
        return self.config.character_name

    @property
    def n_edges(self):
        return [len(parents) for parents in self.parents_list]

    def save_to_bvh(self, out_filepath: str) -> None:
        raise NotImplementedError

    def _enable_global_position(self):
        """
        add a special entity that would be the global position.
        The entity is appended to the edges list.
        No need to really add it in edges_list and all other structures that are based on tuples.
        We add it only to the structures that are based on indices.
        Its neighboring edges are the same as the neighbors of root """
        for pooling_list in [self.skeletal_pooling_dist_0, self.skeletal_pooling_dist_1]:
            for pooling_hierarchical_stage in pooling_list:
                n_small_stage = max(pooling_hierarchical_stage.keys()) + 1
                n_large_stage = max(val for edge in pooling_hierarchical_stage.values()
                                     for val in edge) + 1
                pooling_hierarchical_stage[n_small_stage] = [n_large_stage]

    @property
    def foot_names(self):
        return self.config['feet_names']

    @property
    @functools.lru_cache()
    def foot_indexes(self):
        """Run overs pooling list and calculate foot location at each level"""
        indexes = [i for i, name in enumerate(self.names) if name in self.foot_names]
        all_foot_indexes = [indexes]
        for pooling in self.skeletal_pooling_dist_1[::-1]:
            all_foot_indexes += [[k for k in pooling if any(foot in pooling[k]
                                    for foot in all_foot_indexes[-1])]]

        return all_foot_indexes[::-1]

    @property
    def foot_number(self):
        return len(self.foot_indexes[-1])

    def _enable_foot_contact(self):
        """ add special entities that would be the foot contact labels.
        The entities are appended to the edges list.
        No need to really add them in edges_list and all other structures that are based on tuples.
        We add them only to the structures that are based on indices.
        Their neighboring edges are the same as the neighbors of the feet """
        for pooling_list in [self.skeletal_pooling_dist_0, self.skeletal_pooling_dist_1]:
            for pooling_hierarchical_stage, foot_indexes in zip(pooling_list, self.foot_indexes):
                for _ in foot_indexes:
                    n_small_stage = max(pooling_hierarchical_stage.keys()) + 1
                    n_large_stage = max(val for edge in pooling_hierarchical_stage.values()
                                        for val in edge) + 1
                    pooling_hierarchical_stage[n_small_stage] = [n_large_stage]

    def hierarchical_upsample_layer(self,
                                    layer: int,
                                     pooling_dist=1) -> StaticMotionOneHierarchyLevel:

        assert pooling_dist in [0, 1]
        skeletal_pooling_dist = self.skeletal_pooling_dist_1 if pooling_dist == 1 \
                                                            else self.skeletal_pooling_dist_0

        return StaticMotionOneHierarchyLevel(self.parents_list[layer],
                                             skeletal_pooling_dist[layer - 1],
                                             self.enable_global_position,
                                             self.foot_indexes[layer] if self.enable_foot_contact
                                                                      else [])

    def hierarchical_keep_dim_layer(self, layer: int) -> StaticMotionOneHierarchyLevel:
        return StaticMotionOneHierarchyLevel.keep_dim_layer(self.parents_list[layer],
                                                            self.enable_global_position,
                                    self.foot_indexes[layer] if self.enable_foot_contact else [])

    def number_of_joints_in_hierarchical_levels(self) -> [int]:
        feet_lengths = [len(feet) for feet in self.foot_indexes] if self.enable_foot_contact \
                                                                 else [0] * len(self.parents_list)

        return [len(parents) + self.enable_global_position + feet_length
                 for parents, feet_length in zip(self.parents_list, feet_lengths)]

    @staticmethod
    def _topology_degree(parents: [int]):
        joints_degree = [0] * len(parents)

        for joint in parents[1:]:
            joints_degree[joint] += 1

        return joints_degree

    @staticmethod
    def _find_seq(index: int, joints_degree: [int], parents: [int]) -> [[int]]:
        """Recursive search to find a list of all straight sequences of a skeleton."""
        if joints_degree[index] == 0:
            return [[index]]

        all_sequences = []
        if joints_degree[index] > 1 and index != 0:
            all_sequences = [[index]]

        children_list = [dst for dst, src in enumerate(parents) if src == index]

        for dst in children_list:
            sequence = StaticData._find_seq(dst, joints_degree, parents)
            sequence[0] = [index] + sequence[0]
            all_sequences += sequence

        return all_sequences

    @staticmethod
    def _find_leaves(index: int, joints_degree: [int], parents: [int]) -> [[int]]:
        """Recursive search to find a list of all leaves and their connected
         joint in a skeleton rest position"""
        if joints_degree[index] == 0:
            return []

        all_leaves_pool = []
        connected_leaves = []

        children_list = [dst for dst, src in enumerate(parents) if src == index]

        for dst in children_list:
            leaves = StaticData._find_leaves(dst, joints_degree, parents)
            if leaves:
                all_leaves_pool += leaves
            else:
                connected_leaves += [dst]

        if connected_leaves:
            all_leaves_pool += [[index] + connected_leaves]

        return all_leaves_pool

    @staticmethod
    def _edges_from_joints(joints: [int]):
        return list(zip(joints[:-1], joints[1:]))

    @staticmethod
    def _pooling_for_edges_list(edges: [EdgePoint]) -> list:
        """Return a list sublist of edges of length 2."""
        pooling_groups = [edges[i:i + 2] for i in range(0, len(edges), 2)]
        # If we have an odd numbers of edges pull 3 of them in once.
        if len(pooling_groups) > 1 and len(pooling_groups[-1]) == 1:
            pooling_groups[-2] += pooling_groups[-1]
            pooling_groups = pooling_groups[:-1]

        return pooling_groups

    @staticmethod
    def flatten_dict(values):
        return {k: sublist[k] for sublist in values for k in sublist}

    @staticmethod
    def _calculate_degree1_pooling(parents: [int], degree: [int]) -> {EdgePoint: [EdgePoint]}:
        """Pooling for complex skeleton by trimming long sequences into smaller ones."""
        all_sequences = StaticData._find_seq(0, degree, parents)
        edges_sequences = [StaticData._edges_from_joints(seq) for seq in all_sequences]

        pooling = [{(edge[0][0], edge[-1][-1]): edge
                    for edge in StaticData._pooling_for_edges_list(edges)}
                     for edges in edges_sequences]

        pooling = StaticData.flatten_dict(pooling)

        return pooling

    @staticmethod
    def _calculate_leaves_pooling(parents: [int], degree: [int]) -> {EdgePoint: [EdgePoint]}:
        all_sequences = StaticData._find_seq(0, degree, parents)
        edges_sequences = [StaticData._edges_from_joints(seq) for seq in all_sequences]

        all_joints = [joint for joint, d in enumerate(degree) if d > 0]
        pooling = {}

        for joint in all_joints:
            pooling[joint] = [edge[0] for edge in edges_sequences if edge[0][0] == joint]

        # return {pooling[k][0]: pooling[k] for k in pooling}
        return {value[0]: value for _, value in pooling.items()}

    @staticmethod
    def _calculate_pooling_for_level(parents: [int], degree: [int]) -> {EdgePoint: [EdgePoint]}:
        if any(d == 1 for d in degree):
            return StaticData._calculate_degree1_pooling(parents, degree)

        return StaticData._calculate_leaves_pooling(parents, degree)

    @staticmethod
    def _normalise_joints(pooling: {EdgePoint: [EdgePoint]}) -> {EdgePoint: [EdgePoint]}:
        max_joint = 0
        joint_to_new_joint: {int: int} = {-1: -1, 0: 0}
        new_edges = {}

        for edge in sorted(pooling, key=lambda x: x[1]):
            if edge[1] > max_joint:
                max_joint += 1
                joint_to_new_joint[edge[1]] = max_joint

            new_joint = tuple(joint_to_new_joint[e] for e in edge)
            new_edges[new_joint] = pooling[edge]

        return new_edges

    @staticmethod
    def _edges_to_parents(edges: [EdgePoint]):
        return [edge[0] for edge in edges]

    def calculate_all_pooling_levels(self, parents0):
        all_parents = [list(parents0)]
        all_poolings = []
        degree = StaticData._topology_degree(all_parents[-1])

        while len(all_parents[-1]) > 2:
            pooling = self._calculate_pooling_for_level(all_parents[-1], degree)
            pooling[(-1, 0)] = [(-1, 0)]

            normalised_pooling = self._normalise_joints(pooling)
            normalised_parents = self._edges_to_parents(normalised_pooling.keys())

            all_parents += [normalised_parents]
            all_poolings += [normalised_pooling]

            degree = StaticData._topology_degree(all_parents[-1])

        return all_parents[::-1], all_poolings[::-1]
