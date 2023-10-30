# pylint: skip-file
'''Debug class in order to offer debug functionnality not nessecary in normal flow.'''
import torch
import numpy as np
import matplotlib.pyplot as plt

from motion.motion import DynamicData

def get_foot_location(motion_data: torch.Tensor, motion_statics,
                         use_global_position: bool, use_velocity: bool):
    n_motions, _, _, n_frames = motion_data.shape
    data_dtype = motion_data.dtype

    # fk class must have root at index 0
    offsets = np.repeat(motion_statics.offsets[np.newaxis], n_motions, axis=0)
    offsets = torch.from_numpy(offsets).to(motion_data.device).type(data_dtype)
    fk = ForwardKinematicsJoint(motion_statics.parents, offsets)
    motion_for_fk = motion_data.transpose(1, 3)
    # samples x features x joints x frames  ==>  samples x frames x joints x features

    if use_global_position:
        #  last 'joint' is global position. use only first 3 features out of it.
        glob_pos = motion_for_fk[:, :, -1, :3]
        if use_velocity:
            glob_pos = torch.cumsum(glob_pos, dim=1)
        motion_for_fk = motion_for_fk[:, :, :-1]
    else:
        glob_pos = torch.zeros_like(motion_for_fk[:, :, 0, :3])
    joint_location = fk.forward_edge_rot(motion_for_fk, glob_pos)
    # compute foot contact

    foot_indexes = motion_statics.foot_indexes[-1]
    foot_location = joint_location[:, :, foot_indexes]

    return foot_location, foot_indexes, offsets

def plot(name: str):
    """Decorator for any plot method to create a new figure, title and save it."""
    def decorator(plot_method: callable):
        def wrapper(self, *args, **kwargs):
            plt.figure()
            plot_method(self, *args, **kwargs)
            plt.title(name)
            plt.savefig(f'fc/train_{name.replace(" ", "_")}')
        return wrapper
    return decorator


class DebugDynamic(DynamicData):
    '''Debug class for plotting any dynamic motion.'''
    @property
    def foot_location(self):
        label_idx = self.n_joints + self.motion_statics .enable_global_position
        location, _, _ = get_foot_location(self.motion[:, :, :label_idx], self.motion_statics ,
                                    use_global_position=self.motion_statics.enable_global_position,
                                    use_velocity=self.use_velocity)

        return location

    @property
    def foot_velocity(self):
        return (self.foot_location[:, 1:] - self.foot_location[:, :-1]).pow(2).sum(axis=-1).sqrt()

    @property
    def predicted_foot_contact(self):
        label_idx = self.n_joints + self.motion_statics .enable_global_position
        predicted_foot_contact = self.motion[..., 0, label_idx:, :]
        return torch.sigmoid((predicted_foot_contact - 0.5) * 2 * 6)

    @property
    def foot_contact_loss(self):
        return self.predicted_foot_contact[..., 1:] * self.foot_velocity

    def foot_movement_index(self) -> float:
        """ calculate the amount of foot movement."""
         # B x K x feet x T
        foot_location = self.motion[..., self.motion_statics .foot_indexes[-1], :].clone()
         # B x feet x T
        foot_velocity = ((foot_location[..., 1:] - foot_location[..., :-1]) ** 2).sum(axis=1)

        return foot_velocity.sum(axis=1).sum(axis=1)

    @plot('foot location')
    def plot_foot_location(self, index) -> None:
        my_foot = self.foot_location[index, ..., 1].detach().cpu().numpy()
        plt.plot(my_foot)

    @plot('foot velocity')
    def plot_foot_velocity(self, index) -> None:
        my_vel = self.foot_velocity[index].detach().cpu().numpy()
        plt.plot(my_vel)

    @plot('foot contact')
    def plot_foot_contact_labels(self, index) -> None:
        plt.plot(self.predicted_foot_contact[index].transpose(0, 1).detach().cpu().numpy())

    @plot('loss')
    def plot_foot_contact_loss(self, index) -> None:
        plt.plot(self.foot_contact_loss)

    def plot_all_4(self, index):
        self.plot_foot_location(index)
        self.plot_foot_velocity(index)
        self.plot_foot_contact_labels(index)
        self.plot_foot_contact_loss(index)
