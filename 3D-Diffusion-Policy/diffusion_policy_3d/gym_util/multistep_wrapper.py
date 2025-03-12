import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
from collections import defaultdict, deque
import dill
from klampt.model import trajectory
from klampt.math import so3

def rotation_6d_to_matrix(d6: np.ndarray) -> np.ndarray:
    """
    Converts 6D rotation representation by Zhou et al. [1] to rotation matrix
    using Gram--Schmidt orthogonalization per Section B of [1].
    Args:
        d6: 6D rotation representation, of size (6,)

    Returns:
        rotation matrix of size (3, 3)

    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """

    a1, a2 = d6[:3], d6[3:]
    b1 = a1 / np.linalg.norm(a1)
    b2 = a2 - np.dot(b1, a2) * b1
    b2 = b2 / np.linalg.norm(b2)
    b3 = np.cross(b1, b2)
    return np.stack((b1, b2, b3), axis=-1)

def stack_repeated(x, n):
    return np.repeat(np.expand_dims(x,axis=0),n,axis=0)

def repeated_box(box_space, n):
    return spaces.Box(
        low=stack_repeated(box_space.low, n),
        high=stack_repeated(box_space.high, n),
        shape=(n,) + box_space.shape,
        dtype=box_space.dtype
    )

def repeated_space(space, n):
    if isinstance(space, spaces.Box):
        return repeated_box(space, n)
    elif isinstance(space, spaces.Dict):
        result_space = spaces.Dict()
        for key, value in space.items():
            result_space[key] = repeated_space(value, n)
        return result_space
    else:
        raise RuntimeError(f'Unsupported space type {type(space)}')


def take_last_n(x, n):
    x = list(x)
    n = min(len(x), n)
    
    if isinstance(x[0], torch.Tensor):
        return torch.stack(x[-n:])
    else:
        return np.array(x[-n:])



def dict_take_last_n(x, n):
    result = dict()
    for key, value in x.items():
        result[key] = take_last_n(value, n)
    return result


def aggregate(data, method='max'):
    if isinstance(data[0], torch.Tensor):
        if method == 'max':
            # equivalent to any
            return torch.max(torch.stack(data))
        elif method == 'min':
            # equivalent to all
            return torch.min(torch.stack(data))
        elif method == 'mean':
            return torch.mean(torch.stack(data))
        elif method == 'sum':
            return torch.sum(torch.stack(data))
        else:
            raise NotImplementedError()
    else:
        if method == 'max':
            # equivalent to any
            return np.max(data)
        elif method == 'min':
            # equivalent to all
            return np.min(data)
        elif method == 'mean':
            return np.mean(data)
        elif method == 'sum':
            print(data)
            return np.sum(data)
        else:
            raise NotImplementedError()


def stack_last_n_obs(all_obs, n_steps):
    assert(len(all_obs) > 0)
    all_obs = list(all_obs)
    if isinstance(all_obs[0], np.ndarray):
        result = np.zeros((n_steps,) + all_obs[-1].shape, 
            dtype=all_obs[-1].dtype)
        start_idx = -min(n_steps, len(all_obs))
        result[start_idx:] = np.array(all_obs[start_idx:])
        if n_steps > len(all_obs):
            # pad
            result[:start_idx] = result[start_idx]
    elif isinstance(all_obs[0], torch.Tensor):
        result = torch.zeros((n_steps,) + all_obs[-1].shape, 
            dtype=all_obs[-1].dtype)
        start_idx = -min(n_steps, len(all_obs))
        result[start_idx:] = torch.stack(all_obs[start_idx:])
        if n_steps > len(all_obs):
            # pad
            result[:start_idx] = result[start_idx]
    else:
        raise RuntimeError(f'Unsupported obs type {type(all_obs[0])}')
    return result


class MultiStepWrapper(gym.Wrapper):
    def __init__(self, 
            env, 
            n_obs_steps, 
            n_action_steps, 
            max_episode_steps=None,
            reward_agg_method='max',
        ):
        super().__init__(env)
        self._action_space = repeated_space(env.action_space, n_action_steps)
        self._observation_space = repeated_space(env.observation_space, n_obs_steps)
        self.max_episode_steps = max_episode_steps
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.reward_agg_method = reward_agg_method
        self.n_obs_steps = n_obs_steps

        self.obs = deque(maxlen=n_obs_steps+1)
        self.reward = list()
        self.done = list()
        self.info = defaultdict(lambda : deque(maxlen=n_obs_steps+1))
    
    def reset(self):
        """Resets the environment using kwargs."""
        obs = super().reset()

        self.obs = deque([obs], maxlen=self.n_obs_steps+1)
        self.reward = list()
        self.done = list()
        self.info = defaultdict(lambda : deque(maxlen=self.n_obs_steps+1))

        obs = self._get_obs(self.n_obs_steps)
        return obs

    def step(self, action):
        """
        actions: (n_action_steps,) + action_shape
        """
        for act in action:
            if len(self.done) > 0 and self.done[-1]:
                # termination
                break
            observation, reward, done, info = super().step(act)

            self.obs.append(observation)
            self.reward.append(reward)
            if (self.max_episode_steps is not None) \
                and (len(self.reward) >= self.max_episode_steps):
                # truncation
                done = True
            self.done.append(done)
            self._add_info(info)

        observation = self._get_obs(self.n_obs_steps)
        # reward = aggregate(self.reward, self.reward_agg_method)
        reward = 0
        done = aggregate(self.done, 'max')
        info = dict_take_last_n(self.info, self.n_obs_steps)
        return observation, reward, done, info
    
    def step_interpolate(self, action, interpolate_steps=2):
        """
        Interpolates between actions with user specified number of steps using linear interpolation
        actions: (n_action_steps,) + action_shape
        """
        
        interpolated_actions = []

        for i in range(len(action) - 1):
            rot1, trans1 = action[i][:6], action[i][6:]
            rot2, trans2 = action[i + 1][:6], action[i + 1][6:]

            rot1_matrix = rotation_6d_to_matrix(rot1)
            rot2_matrix = rotation_6d_to_matrix(rot2)
            R1 = so3.from_matrix(rot1_matrix)
            R2 = so3.from_matrix(rot2_matrix)
            T1 = [R1, trans1.tolist()]
            T2 = [R2, trans2.tolist()]

            traj = trajectory.SE3Trajectory(times=[0, interpolate_steps+1], milestones=[T1, T2])
            
            for j in range(interpolate_steps + 1):
                T = traj.eval(j)
                rot_6d = np.array(T[0]).reshape(3, 3, order='F')[:2].flatten().tolist()
                trans = T[1]
                interpolated_actions.append(np.concatenate((rot_6d, trans)))
        
        interpolated_actions.append(action[-1])

        action = interpolated_actions

        for act in action:
            if len(self.done) > 0 and self.done[-1]:
                # termination
                break
            observation, reward, done, info = super().step(act)

            self.obs.append(observation)
            self.reward.append(reward)
            if (self.max_episode_steps is not None) \
                and (len(self.reward) >= self.max_episode_steps):
                # truncation
                done = True
            self.done.append(done)
            # self._add_info(info)

        observation = self._get_obs(self.n_obs_steps)
        # reward = aggregate(self.reward, self.reward_agg_method)
        reward = None
        done = aggregate(self.done, 'max')
        info = dict_take_last_n(self.info, self.n_obs_steps)
        return observation, reward, done, info

    def _get_obs(self, n_steps=1):
        """
        Output (n_steps,) + obs_shape
        """
        assert(len(self.obs) > 0)
        if isinstance(self.observation_space, spaces.Box):
            return stack_last_n_obs(self.obs, n_steps)
        elif isinstance(self.observation_space, spaces.Dict):
            result = dict()
            for key in self.observation_space.keys():
                result[key] = stack_last_n_obs(
                    [obs[key] for obs in self.obs],
                    n_steps
                )
            return result
        else:
            raise RuntimeError('Unsupported space type')

    def _add_info(self, info):
        for key, value in info.items():
            self.info[key].append(value)
    
    def get_rewards(self):
        return self.reward
    
    def get_attr(self, name):
        return getattr(self, name)

    def run_dill_function(self, dill_fn):
        fn = dill.loads(dill_fn)
        return fn(self)
    
    def get_infos(self):
        result = dict()
        for k, v in self.info.items():
            result[k] = list(v)
        return result
