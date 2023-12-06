import os
from pathlib import Path

import torch

from utils.batch_size import get_batch_size_from_vram

# Settings
## Machine used
LOCAL = False

## PyTorch
USE_CUDA = True
DEVICE = 'cuda' if torch.cuda.is_available() and USE_CUDA else 'cpu'
EPOCHS = 2
BATCH_SIZE = get_batch_size_from_vram() 
NUM_WORKERS = len(os.sched_getaffinity(0)) # number of avaliable CPU cores
WARMUP_STEPS = 1_000

USE_AMP = True
VERBOSE = True
SAVE_MODEL = True

# Paths to data
DATA_FOLDER = Path('/dtu/blackhole/1a/163226') if not LOCAL else Path(os.getcwd())
TRIPLES_SMALL_PATH = DATA_FOLDER / 'triples.train.small.tsv'

# Evaluation params
EVALUATION_PATH = Path('/dtu/blackhole/1b/167931/SBERT_models') if not LOCAL else Path(os.getcwd()) # '/dtu/blackhole/1a/163226'
EVALUATION_MODEL_PATH = '2epochs' # Looks into data folder e.g Path(/dtu/blackhole/1a/163226/) / <EVALUATION_MODEL_PATH> <- 1epoch