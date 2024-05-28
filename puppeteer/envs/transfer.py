from copy import deepcopy

import gym
import numpy as np
import torch
from dm_control.locomotion.tasks.reference_pose import tracking
from envs import dm_control_wrapper

from envs.wrappers.time_limit import TimeLimit
from envs.wrappers.humanoid import HumanoidWrapper, flatten_dict
from envs.tasks.run_through_corridor import RunThroughCorridor
from envs.tasks.walk import Stand, Walk, Run
from envs.tracking import CMU_HUMANOID_OBSERVABLES
from tdmpc2 import TDMPC2


TASKS = {
	'corridor': {
		'constructor': RunThroughCorridor,
		'task_kwargs': {
			'arena_type': 'corridor',
			'target_velocity': 6.0,
		},
	},
	'gaps-corridor': {
		'constructor': RunThroughCorridor,
		'task_kwargs': {
			'arena_type': 'gaps-corridor',
			'target_velocity': 6.0,
		},
	},
	'walls-corridor': {
		'constructor': RunThroughCorridor,
		'task_kwargs': {
			'arena_type': 'walls-corridor',
			'target_velocity': 6.0,
		},
	},
	'stairs-corridor': {
		'constructor': RunThroughCorridor,
		'task_kwargs': {
			'arena_type': 'stairs-corridor',
			'target_velocity': 6.0,
		},
	},
	'hurdles-corridor': {
		'constructor': RunThroughCorridor,
		'task_kwargs': {
			'arena_type': 'hurdles-corridor',
			'target_velocity': 6.0,
		},
	},
	'stand': {
		'constructor': Stand,
		'task_kwargs': {},
	},
	'walk': {
		'constructor': Walk,
		'task_kwargs': {},
	},
	'run': {
		'constructor': Run,
		'task_kwargs': {},
	},
}


class TransferWrapper(gym.Wrapper):
	"""
	Wrapper for hierarchical transfer tasks.
	"""

	def __init__(self, env, cfg):
		super().__init__(env)
		self.cfg = cfg
		self.low_level_cfg = deepcopy(cfg)
		
		if self.cfg.low_level_fp is not None: # use low-level policy
			# modify cfg to match low-level policy
			state_dict = torch.load(cfg.low_level_fp, map_location='cpu')
			self.low_level_cfg.obs = 'state'
			self.low_level_cfg.obs_shape = {'state': (state_dict['model']['_encoder.state.0.weight'].size(1),)}
			self.low_level_cfg.action_dim = 56

			# load low-level policy
			self.low_level_policy = TDMPC2(self.low_level_cfg)
			self.low_level_policy.load(state_dict)
			self.low_level_policy.model.eval()

			# redefine action space
			self.action_space = gym.spaces.Box(
				low=-1, high=1, shape=(15,), dtype=np.float32
			)

	def _preprocess_obs(self, obs):
		low_level_obs = {}
		for k in CMU_HUMANOID_OBSERVABLES:
			low_level_obs[k] = obs[k]
		high_level_obs = dict()
		high_level_obs.update(low_level_obs)
		try:
			aux_keys = self.env._env.task.auxiliary_obs_keys
		except:
			aux_keys = ()
		aux_keys += ('walker/target', 'walker/egocentric_camera')
		for k in aux_keys:
			if k in obs:
				high_level_obs[k] = obs[k]
		if self.cfg.obs == 'rgb' and 'walker/egocentric_camera' in high_level_obs:
			# replace egocentric camera key with third-person camera rendering
			high_level_obs['walker/egocentric_camera'] = self.render(height=64, width=64).copy()
		return low_level_obs, high_level_obs

	def reset(self):
		self._t = 0
		obs = self.env.reset()
		self._low_level_obs, obs = self._preprocess_obs(obs)
		return obs

	def step(self, action):
		if self.cfg.low_level_fp is not None: # use low-level policy
			action = action * self.cfg.action_scale

			# get low-level observation
			low_level_obs = deepcopy(self._low_level_obs)
			low_level_obs['walker/reference_appendages_pos'] = low_level_obs['walker/appendages_pos'] + action
			low_level_obs = torch.from_numpy(flatten_dict(low_level_obs))

			# get low-level action
			action = self.low_level_policy.act(low_level_obs, t0=self._t==0, eval_mode=True)

		obs, reward, done, info = self.env.step(action)
		self._low_level_obs, obs = self._preprocess_obs(obs)
		self._t += 1
		return obs, reward, done, info
	
	def render(self, height=384, width=384):
		camera_id = 0 if 'corridor' in self.cfg.task else 3
		return self.env.physics.render(height, width, camera_id=camera_id)


def make_env(cfg):
	"""
	Make CMU Humanoid environment for transfer tasks.
	"""
	if cfg.task not in TASKS:
		raise ValueError('Unknown task:', cfg.task)

	task_kwargs = dict(
		physics_timestep=tracking.DEFAULT_PHYSICS_TIMESTEP,
		control_timestep=0.03,
	)
	task_kwargs.update(TASKS[cfg.task]['task_kwargs'])
	
	env = dm_control_wrapper.DmControlWrapper.make_env_constructor(
		TASKS[cfg.task]['constructor'])(task_kwargs=task_kwargs)
	env = TransferWrapper(env, cfg)

	max_episode_steps = 500
	env = HumanoidWrapper(env, cfg, max_episode_steps=max_episode_steps)
	env = TimeLimit(env, max_episode_steps=max_episode_steps)
	env.max_episode_steps = env._max_episode_steps

	return env
