import absl.logging
absl.logging.set_verbosity(absl.logging.WARNING)
from pathlib import Path
from typing import Any, Dict, Sequence, Text, Optional, Tuple, Union

import numpy as np
from gym import spaces
from dm_control.locomotion.mocap import cmu_mocap_data
from dm_control.locomotion.tasks.reference_pose import types
from dm_control.locomotion.tasks.reference_pose import tracking
from dm_control.locomotion.walkers import cmu_humanoid
from envs import dm_control_wrapper

from envs.wrappers.time_limit import TimeLimit
from envs.wrappers.humanoid import HumanoidWrapper


CMU_HUMANOID_OBSERVABLES = (
	'walker/actuator_activation',
	'walker/appendages_pos',
	'walker/body_height',
	'walker/joints_pos',
	'walker/joints_vel',
	'walker/sensors_accelerometer',
	'walker/sensors_gyro',
	'walker/sensors_torque',
	'walker/sensors_touch',
	'walker/sensors_velocimeter',
	'walker/world_zaxis'
)
REFERENCE_OBSERVABLES = (
	'walker/reference_appendages_pos',
)
TRACKING_OBSERVABLES = CMU_HUMANOID_OBSERVABLES + REFERENCE_OBSERVABLES


class MocapTrackingGymEnv(dm_control_wrapper.DmControlWrapper):
	"""
	Wraps the MultiClipMocapTracking into a Gym env.
	Adapted from https://github.com/microsoft/MoCapAct/blob/main/mocapact/envs/tracking.py
	"""

	def __init__(
		self,
		clip_ids: Optional[Sequence[Text]] = None,
		ref_steps: Tuple[int] = (0,),
		reward_type: str = 'comic',
		mocap_path: Optional[Union[str, Path]] = None,
		task_kwargs: Optional[Dict[str, Any]] = None,
		environment_kwargs: Optional[Dict[str, Any]] = None,
		act_noise: float = 0.01,
		enable_all_proprios: bool = False,
		enable_cameras: bool = False,
		include_clip_id: bool = False,
		display_ghost: bool = True,

		# for rendering
		width: int = 640,
		height: int = 480,
		camera_id: int = 3
	):
		assert clip_ids is not None, 'clip_ids must be specified'
		self._dataset = types.ClipCollection(ids=clip_ids)
		self._enable_all_proprios = enable_all_proprios
		self._enable_cameras = enable_cameras
		self._include_clip_id = include_clip_id
		task_kwargs = task_kwargs or dict()
		task_kwargs['ref_path'] = mocap_path if mocap_path else cmu_mocap_data.get_path_for_cmu(version='2020')
		task_kwargs['dataset'] = self._dataset
		task_kwargs['ref_steps'] = ref_steps
		task_kwargs['reward_type'] = reward_type
		if display_ghost:
			task_kwargs['ghost_offset'] = np.array([1., 0., 0.])
		super().__init__(
			tracking.MultiClipMocapTracking,
			task_kwargs=task_kwargs,
			environment_kwargs=environment_kwargs,
			act_noise=act_noise,
			width=width,
			height=height,
			camera_id=camera_id
		)

	def _get_walker(self):
		return cmu_humanoid.CMUHumanoidPositionControlledV2020

	def _create_env(
		self,
		task_type,
		task_kwargs,
		environment_kwargs,
		act_noise=0.,
		arena_size=(12., 12.)
	):
		env = super()._create_env(task_type, task_kwargs, environment_kwargs, act_noise)
		walker = env._task._walker
					
		if self._enable_all_proprios:
			walker.observables.enable_all()
			walker.observables.prev_action.enabled = False # this observable is not implemented
			if not self._enable_cameras:
				# TODO: procedurally find the cameras
				walker.observables.egocentric_camera.enabled = False
				walker.observables.body_camera.enabled = False
			env.reset()
		return env
	
	def _create_observation_space(self) -> spaces.Dict:
		obs_spaces = dict()
		s_dic = self._env.observation_spec()
		for k in TRACKING_OBSERVABLES:
			v = s_dic[k]
			if v.dtype == np.float64 and np.prod(v.shape) > 0:
				obs_spaces[k] = spaces.Box(
					-np.infty,
					np.infty,
					shape=(np.prod(v.shape),),
					dtype=np.float32
				)
			elif v.dtype == np.uint8:
				tmp = v.generate_value()
				obs_spaces[k] = spaces.Box(
					v.minimum.item(),
					v.maximum.item(),
					shape=tmp.shape,
					dtype=np.uint8
				)
			elif k == 'walker/clip_id' and self._include_clip_id:
				obs_spaces[k] = spaces.Discrete(len(self._dataset.ids))
		return spaces.Dict(obs_spaces)

	def step(self, action: np.ndarray) -> Tuple[Dict[str, np.ndarray], float, bool, Dict[str, Any]]:
		obs, reward, done, info = super().step(action)
		if 'walker/time_in_clip' in obs:
			info['time_in_clip'] = obs['walker/time_in_clip'].item()
			info['start_time_in_clip'] = self._start_time_in_clip
			info['last_time_in_clip'] = self._last_time_in_clip
		return obs, reward, done, info

	def reset(self):
		time_step = self._env.reset()
		obs = self.get_observation(time_step)
		if 'walker/time_in_clip' in obs:
			self._start_time_in_clip = obs['walker/time_in_clip'].item()
			self._last_time_in_clip = self._env.task._last_step / (len(self._env.task._clip_reference_features['joints'])-1)
		return obs


def select_clips(cfg):
	"""
	Selects a subset of available mocap clips to track.
	"""
	clip_ids = cfg.get('clip_ids', None)
	if clip_ids is None: # select random subset of clips
		fp = Path(cfg.data_dir)
		clips = [f.stem for f in fp.glob('*.hdf5')]
		print(f'Found {len(clips)} clips in {cfg.data_dir}')
		num_clips = cfg.get('num_clips', 'all')
		if num_clips != 'all':
			clips = np.random.choice(clips, int(num_clips), replace=False)
		print(f'Selected {len(clips)} clips to track')
	else:
		try:
			clips = [clip.upper() for clip in clip_ids.split(',')]
		except:
			clips = [clip_ids.upper()]
	return clips


def make_env(cfg):
	"""
	Make CMU Humanoid environment for the Mocap Tracking task.
	"""
	if cfg.task != 'tracking':
		raise ValueError('Unknown task:', cfg.task)
	env = MocapTrackingGymEnv(clip_ids=select_clips(cfg), ref_steps=[1])
	env = HumanoidWrapper(env, cfg, max_episode_steps=1_000)
	env = TimeLimit(env, max_episode_steps=1_000)
	env.max_episode_steps = env._max_episode_steps
	return env
