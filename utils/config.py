import os
from pathlib import Path

import torch

# Settings
## Machine used
LOCAL = False

## PyTorch
USE_CUDA = True
DEVICE = 'cuda' if torch.cuda.is_available() and USE_CUDA else 'cpu'
EPOCHS = 1
BATCH_SIZE = 128 # Should be about 128 for 32GB vram and 256 for 80GB vram
NUM_WORKERS = len(os.sched_getaffinity(0)) # number of avaliable CPU cores
WARMUP_STEPS = 1_000

USE_AMP = True
VERBOSE = True
SAVE_MODEL = True

# Paths to data
DATA_FOLDER = Path('/dtu/blackhole/1a/163226') if not LOCAL else Path(os.getcwd())
TRIPLES_SMALL_PATH = DATA_FOLDER / 'triples.train.small.tsv'