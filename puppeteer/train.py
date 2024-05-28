import os
os.environ['MUJOCO_GL'] = 'egl'
os.environ['LAZY_LEGACY_OP'] = '0'
import warnings
warnings.filterwarnings('ignore')
import torch

import hydra
from termcolor import colored

from common.parser import parse_cfg
from common.seed import set_seed
from common.buffer import Buffer, EnsembleBuffer
from envs import make_env
from tdmpc2 import TDMPC2
from trainer import Trainer
from common.logger import Logger

torch.backends.cudnn.benchmark = True


@hydra.main(config_name='config', config_path='.')
def train(cfg: dict):
	"""
	Script for training Puppeteer agents for humanoid control tasks.

	Most relevant args:
		`task`: task name (default: tracking)
		`steps`: number of training/environment steps (default: 10M)
		`seed`: random seed (default: 1)

	See config.yaml for a full list of args.

	Example usage:
	```
		$ python train.py task=tracking
	```
	"""
	assert torch.cuda.is_available()
	assert cfg.steps > 0, 'Must train for at least 1 step.'
	cfg = parse_cfg(cfg)
	set_seed(cfg.seed)
	print(colored('Work dir:', 'yellow', attrs=['bold']), cfg.work_dir)

	buffer_cls = EnsembleBuffer if cfg.task == 'tracking' else Buffer
	trainer = Trainer(
		cfg=cfg,
		env=make_env(cfg),
		agent=TDMPC2(cfg),
		buffer=buffer_cls(cfg),
		logger=Logger(cfg),
	)
	trainer.train()
	print('\nTraining completed successfully')


if __name__ == '__main__':
	train()
