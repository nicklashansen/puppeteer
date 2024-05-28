import warnings

import gym

from envs.wrappers import TensorWrapper
from envs.tracking import make_env as make_tracking_env
from envs.transfer import make_env as make_transfer_env


warnings.filterwarnings('ignore', category=DeprecationWarning)


def make_env(cfg):
	"""
	Make an environment for TD-MPC2 experiments.
	"""
	gym.logger.set_level(40)
	env = None
	for fn in [make_transfer_env, make_tracking_env]:
		try:
			env = fn(cfg)
			break
		except ValueError:
			pass
	assert env is not None, f'Failed to create environment for task {cfg.task}'
	env = TensorWrapper(env)
	try: # Dict
		cfg.obs_shape = {k: v.shape for k, v in env.observation_space.spaces.items()}
	except: # Box
		cfg.obs_shape = {cfg.get('obs', 'state'): env.observation_space.shape}
	cfg.action_dim = env.action_space.shape[0]
	cfg.episode_length = env.max_episode_steps
	cfg.seed_steps = max(1000, 5*cfg.episode_length)
	return env
