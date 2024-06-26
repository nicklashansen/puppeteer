from copy import deepcopy

import torch
from tensordict.tensordict import TensorDict
from torchrl.data.replay_buffers import ReplayBuffer, LazyTensorStorage
from torchrl.data.replay_buffers.samplers import SliceSampler

from common.mocap_dataset import MocapBuffer


class Buffer():
	"""
	Replay buffer for TD-MPC2 training. Based on torchrl.
	Uses CUDA memory if available, and CPU memory otherwise.
	"""

	def __init__(self, cfg):
		self.cfg = cfg
		self._device = torch.device('cuda')
		self._capacity = min(cfg.buffer_size, cfg.steps) + cfg.episode_length
		self._sampler = SliceSampler(
			num_slices=self.cfg.batch_size,
			end_key=None,
			traj_key='episode',
			truncated_key=None,
			strict_length=True,
		)
		self._batch_size = cfg.batch_size * (cfg.horizon+1)
		self._num_eps = 0

	@property
	def capacity(self):
		"""Return the capacity of the buffer."""
		return self._capacity
	
	@property
	def num_eps(self):
		"""Return the number of episodes in the buffer."""
		return self._num_eps

	def _reserve_buffer(self, storage):
		"""
		Reserve a buffer with the given storage.
		"""
		return ReplayBuffer(
			storage=storage,
			sampler=self._sampler,
			pin_memory=True,
			prefetch=2,
			batch_size=self._batch_size,
		)

	def _init(self, tds):
		"""Initialize the replay buffer. Use the first episode to estimate storage requirements."""
		print(f'Buffer capacity: {self._capacity:,}')
		bytes_per_step = sum([
				(v.numel()*v.element_size() if not isinstance(v, TensorDict) \
				else sum([x.numel()*x.element_size() for x in v.values()])) \
			for v in tds.values()
		]) / len(tds)
		total_bytes = bytes_per_step*self._capacity
		print(f'Storage required: {total_bytes/1e9:.2f} GB')
		print(f'Using CUDA memory for replay buffer.')
		return self._reserve_buffer(
			LazyTensorStorage(self._capacity, device=torch.device('cuda'))
		)

	def _to_device(self, *args, device=None):
		if device is None:
			device = self._device
		return (arg.to(device, non_blocking=True) \
			if arg is not None else None for arg in args)

	def _prepare_batch(self, td):
		"""
		Prepare a sampled batch for training (post-processing).
		Expects `td` to be a TensorDict with batch size TxB.
		"""
		obs = td['obs']
		action = td['action'][1:]
		reward = td['reward'][1:].unsqueeze(-1)
		terminated = td['terminated'][-1].unsqueeze(-1)
		return self._to_device(obs, action, reward, terminated)

	def add(self, td):
		"""Add an episode to the buffer."""
		if len(td) <= self.cfg.horizon+1:
			print(f'Warning: episode of length {len(td)} is too short and will be ignored.')
			return self._num_eps
		td['episode'] = torch.ones_like(td['reward'], dtype=torch.int64) * self._num_eps
		if self._num_eps == 0:
			self._buffer = self._init(td)
		self._buffer.extend(td)
		self._num_eps += 1
		return self._num_eps

	def sample(self):
		"""Sample a batch of subsequences from the buffer."""
		td = self._buffer.sample().view(-1, self.cfg.horizon+1).permute(1, 0)
		return self._prepare_batch(td)


class EnsembleBuffer(Buffer):
	"""
	Ensemble of an offline dataloader and an online replay buffer.
	"""

	def __init__(self, cfg):
		_cfg = deepcopy(cfg)
		_cfg.batch_size //= 2
		self._offline = MocapBuffer(_cfg)
		super().__init__(_cfg)

	def sample(self):
		"""Sample a batch of subsequences from the two buffers."""
		obs0, action0, reward0, terminated0 = self._offline.sample()
		obs1, action1, reward1, terminated1 = super().sample()
		return torch.cat([obs0, obs1], dim=1), \
			torch.cat([action0, action1], dim=1), \
			torch.cat([reward0, reward1], dim=1), \
			torch.cat([terminated0, terminated1], dim=0)
