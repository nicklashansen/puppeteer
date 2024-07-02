import numpy as np
import gym


def flatten_dict(d):
	return np.concatenate([d[k] for k in sorted(d.keys())], axis=-1)


class HumanoidWrapper(gym.Wrapper):
	def __init__(self, env, cfg, max_episode_steps=100_000):
		super().__init__(env)
		self.env = env
		self.cfg = cfg
		self._max_episode_steps = max_episode_steps
		self._t = 0
		obs = self._preprocess_obs(env.reset())
		if self.cfg.obs == 'state':
			obs_shape = obs.shape
			self.observation_space = gym.spaces.Box(
				low=np.full(obs_shape, -np.inf),
				high=np.full(obs_shape, np.inf),
				dtype=np.float32,
			)
		elif self.cfg.obs == 'rgb':
			self.observation_space = gym.spaces.Dict(
				state=gym.spaces.Box(-np.inf, np.inf, shape=(obs['state'].shape), dtype=np.float32),
				rgb=gym.spaces.Box(0, 255, shape=obs['rgb'].shape, dtype=np.uint8)
			)

	def _preprocess_obs(self, obs):
		if 'walker/egocentric_camera' in obs:
			rgb = obs.pop('walker/egocentric_camera').transpose(2, 0, 1)
		state = flatten_dict(obs)
		if self.cfg.obs == 'state':
			return state
		elif self.cfg.obs == 'rgb':
			return {'state': state, 'rgb': rgb}

	def reset(self, **kwargs):
		self._t = 0
		obs = self.env.reset(**kwargs)
		return self._preprocess_obs(obs)

	def step(self, action):
		self._t += 1
		obs, reward, done, info = self.env.step(action)
		if 'time_in_clip' in info:
			info['success'] = info['time_in_clip'] >= info['last_time_in_clip']
		else:
			info['success'] = False
		info['truncated'] = self._t == self._max_episode_steps or info['success']
		info['terminated'] = done and not info['truncated']
		done = info['truncated'] or info['terminated']
		info['done'] = done
		obs = self._preprocess_obs(obs)
		return obs, reward, done, info

	@property
	def unwrapped(self):
		return self.env.unwrapped

	def render(self, mode='rgb_array', width=384, height=384):
		camera_id = 0 if 'corridor' in self.cfg.task else 3
		return self.env.physics.render(height, width, camera_id=camera_id)
