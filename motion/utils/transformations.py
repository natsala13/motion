'''Tranformation to the static skeleton utils'''

import copy

import numpy as np

from Motion.Quaternions import Quaternions
from Motion.AnimationStructure import children_list, get_sorted_order


def expand_topology_edges(anim, req_joint_idx=None, names=None,
                             offset_len_mean=None, nearest_joint_ratio=0.9):
    assert nearest_joint_ratio == 1, 'currently not supporting nearest_joint_ratio != 1'

    # we do not want to change inputs that are given as views
    anim = copy.deepcopy(anim)
    req_joint_idx = copy.deepcopy(req_joint_idx)
    names = copy.deepcopy(names)
    offset_len_mean = copy.deepcopy(offset_len_mean)

    n_frames, n_joints_all = anim.shape
    if req_joint_idx is None:
        req_joint_idx = np.arange(n_joints_all)
    if names is None:
        names = np.array([str(i) for i in range(len(req_joint_idx))])

    # fix the topology according to required joints
    parent_req = np.zeros(n_joints_all) # fixed parents according to req
    n_children_req = np.zeros(n_joints_all) # number of children per joint in the fixed topology
    children_all = children_list(anim.parents)
    for idx in req_joint_idx:
        child = idx
        parent = anim.parents[child]
        while parent not in np.append(req_joint_idx, -1):
            child = parent
            parent = anim.parents[child]
        parent_req[idx] = parent
        if parent != -1:
            n_children_req[parent] += 1

    # find out how many joints have multiple children
    super_parents = np.where(n_children_req > 1)[0]
    n_super_children = n_children_req[super_parents].sum().astype(int) # total num of children

    if n_super_children == 0:
        return anim, req_joint_idx, names, offset_len_mean  # can happen in lower hierarchy levels

    # prepare space for expanded joints, at the end of each array
    anim.offsets = np.append(anim.offsets, np.zeros(shape=(n_super_children,3)), axis=0)
    anim.positions = np.append(anim.positions,
                                 np.zeros(shape=(n_frames, n_super_children,3)), axis=1)
    anim.rotations = Quaternions(np.append(anim.rotations,
                                 Quaternions.id((n_frames, n_super_children)), axis=1))
    anim.orients = Quaternions(np.append(anim.orients, Quaternions.id(n_super_children), axis=0))
    anim.parents = np.append(anim.parents, np.zeros(n_super_children, dtype=int))
    names = np.append(names, np.zeros(n_super_children, dtype='<U40'))
    if offset_len_mean is not None:
        offset_len_mean = np.append(offset_len_mean, np.zeros(n_super_children))

    # fix topology and names
    new_joint_idx = n_joints_all
    req_joint_idx = np.append(req_joint_idx, new_joint_idx + np.arange(n_super_children))
    for parent in super_parents:
        for child in children_all[parent]:
            if child in req_joint_idx:
                anim.parents[new_joint_idx] = parent
                anim.parents[child] = new_joint_idx
                names[new_joint_idx] = names[parent]+'_'+names[child]

                new_joint_idx += 1

    # sort data items in a topological order
    sorted_order = get_sorted_order(anim.parents)
    anim = anim[:, sorted_order]
    names = names[sorted_order]
    if offset_len_mean is not None:
        offset_len_mean = offset_len_mean[sorted_order]

    # assign updated locations to req_joint_idx
    sorted_order_inversed = {num: i for i, num in enumerate(sorted_order)}
    sorted_order_inversed[-1] = -1
    req_joint_idx = np.array([sorted_order_inversed[i] for i in req_joint_idx])
    req_joint_idx.sort()

    return anim, req_joint_idx, names, offset_len_mean
