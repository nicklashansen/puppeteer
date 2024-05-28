from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
from tensordict.tensordict import TensorDict

from common import layers, math, init


class WorldModel(nn.Module):
	"""
	TD-MPC2 implicit world model architecture.
	"""

	def __init__(self, cfg):
		super().__init__()
		self.cfg = cfg
		self._encoder = layers.enc(cfg)
		self._dynamics = layers.mlp(cfg.latent_dim + cfg.action_dim, 2*[cfg.mlp_dim], cfg.latent_dim, act=layers.SimNorm(cfg))
		self._reward = layers.mlp(cfg.latent_dim + cfg.action_dim, 2*[cfg.mlp_dim], max(cfg.num_bins, 1))
		self._terminated = layers.mlp(cfg.latent_dim, 2*[cfg.mlp_dim], 1)
		self._pi = layers.mlp(cfg.latent_dim, 2*[cfg.mlp_dim], 2*cfg.action_dim)
		self._Qs = layers.Ensemble([layers.mlp(cfg.latent_dim + cfg.action_dim, 2*[cfg.mlp_dim], max(cfg.num_bins, 1), dropout=cfg.dropout) for _ in range(cfg.num_q)])
		self.apply(init.weight_init)
		init.zero_([self._reward[-1].weight, self._Qs.params[-2]])
		self._target_Qs = deepcopy(self._Qs).requires_grad_(False)
		self.log_std_min = torch.tensor(cfg.log_std_min)
		self.log_std_dif = torch.tensor(cfg.log_std_max) - self.log_std_min

	@property
	def total_params(self):
		return sum(p.numel() for p in self.parameters() if p.requires_grad)
		
	def to(self, *args, **kwargs):
		"""
		Overriding `to` method to also move additional tensors to device.
		"""
		super().to(*args, **kwargs)
		self.log_std_min = self.log_std_min.to(*args, **kwargs)
		self.log_std_dif = self.log_std_dif.to(*args, **kwargs)
		return self
	
	def train(self, mode=True):
		"""
		Overriding `train` method to keep target Q-networks in eval mode.
		"""
		super().train(mode)
		self._target_Qs.train(False)
		return self

	def track_q_grad(self, mode=True):
		"""
		Enables/disables gradient tracking of Q-networks.
		Avoids unnecessary computation during policy optimization.
		"""
		for p in self._Qs.parameters():
			p.requires_grad_(mode)

	def soft_update_target_Q(self):
		"""
		Soft-update target Q-networks using Polyak averaging.
		"""
		with torch.no_grad():
			for p, p_target in zip(self._Qs.parameters(), self._target_Qs.parameters()):
				p_target.data.lerp_(p.data, self.cfg.tau)

	def encode(self, obs):
		"""
		Encodes an observation into its latent representation.
		This implementation assumes a single state-based observation.
		"""
		if isinstance(obs, (dict, TensorDict)):
			out = {}
			for k, v in obs.items():
				if k == 'rgb' and v.ndim == 5:
					out[k] = torch.stack([self._encoder[k](o) for o in v])
				else:
					out[k] = self._encoder[k](v)
			return torch.stack([out[k] for k in out.keys()]).mean(0)
		return self._encoder[self.cfg.obs](obs)

	def pixel_encode(self, obs):
		"""
		Encodes a pixel observation into its latent representation.
		"""
		original_ndim = obs.ndim
		# Ensure obs has a batch dimension if missing
		if original_ndim == 3:
			obs = obs.unsqueeze(0).unsqueeze(0)
		elif original_ndim == 4:
			obs = obs.unsqueeze(1)

		assert self.cfg.vision, "Vision configuration must be enabled."
		assert obs.ndim == 5, "Input observation must have 5 dimensions"
		z = [self._pixel_encoder(obs[i]) for i in range(obs.shape[0])]
		z = torch.stack(z)
		if original_ndim == 3:
			z = z.squeeze(0)
		elif original_ndim == 4:
			z = z.squeeze(1)
		return z

	def next(self, z, a):
		"""
		Predicts the next latent state given the current latent state and action.
		"""
		z = torch.cat([z, a], dim=-1)
		return self._dynamics(z)
	
	def reward(self, z, a):
		"""
		Predicts instantaneous (single-step) reward.
		"""
		z = torch.cat([z, a], dim=-1)
		return self._reward(z)
	
	def terminated(self, z):
		"""
		Predicts termination signal.
		"""
		return torch.sigmoid(self._terminated(z))

	def pi(self, z):
		"""
		Samples an action from the policy prior.
		The policy prior is a Gaussian distribution with
		mean and (log) std predicted by a neural network.
		"""
		# Gaussian policy prior
		mu, log_std = self._pi(z).chunk(2, dim=-1)
		log_std = math.log_std(log_std, self.log_std_min, self.log_std_dif)
		eps = torch.randn_like(mu)

		log_pi = math.gaussian_logprob(eps, log_std, size=None)
		pi = mu + eps * log_std.exp()
		mu, pi, log_pi = math.squash(mu, pi, log_pi)

		return mu, pi, log_pi, log_std

	def Q(self, z, a, return_type='min', target=False):
		"""
		Predict state-action value.
		`return_type` can be one of [`min`, `avg`, `all`]:
			- `min`: return the minimum of two randomly subsampled Q-values.
			- `avg`: return the average of two randomly subsampled Q-values.
			- `all`: return all Q-values.
		`target` specifies whether to use the target Q-networks or not.
		"""
		assert return_type in {'min', 'avg', 'all'}
			
		z = torch.cat([z, a], dim=-1)
		out = (self._target_Qs if target else self._Qs)(z)

		if return_type == 'all':
			return out

		Q1, Q2 = out[np.random.choice(self.cfg.num_q, 2, replace=False)]
		Q1, Q2 = math.two_hot_inv(Q1, self.cfg), math.two_hot_inv(Q2, self.cfg)
		return torch.min(Q1, Q2) if return_type == 'min' else (Q1 + Q2) / 2
