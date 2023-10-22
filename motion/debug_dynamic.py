import torch
import matplotlib.ptplot as plt

from motion import DynamicData
from utils.foot import get_foot_location


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
    @property
    def foot_location(self):
        label_idx = self.n_joints + self.motion_statics .enable_global_position
        location, _, _ = get_foot_location(self.motion[:, :, :label_idx], self.motion_statics ,
                                           use_global_position=self.motion_statics .enable_global_position,
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
        foot_location = self.motion[..., self.motion_statics .foot_indexes[-1], :].clone()  # B x K x feet x T
        foot_velocity = ((foot_location[..., 1:] - foot_location[..., :-1]) ** 2).sum(axis=1)  # B x feet x T

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
