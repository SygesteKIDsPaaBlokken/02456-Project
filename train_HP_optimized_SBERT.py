#%% General imports
import torch.optim
from torch.utils.data import DataLoader

from sentence_transformers import losses
import optuna

from models.SBERT import SBERT
from utils.MSMarcoDatasetSmall import MSMarcoSmallPandas
from utils.config import TRIPLES_SMALL_PATH, DEVICE, BATCH_SIZE, NUM_WORKERS, USE_AMP, VERBOSE, WARMUP_STEPS, EPOCHS, SAVE_MODEL

# %% Load study
print("########################### Loading hyper parameters ###########################")
study_name = 'sbert_HP_1m'
storage = optuna.storages.JournalStorage(
    optuna.storages.JournalFileStorage(f"./data/SBERT_hyperparam_opt/{study_name}.log"),  # NFS path for distributed optimization
)

study = optuna.load_study(study_name=study_name, storage=storage)
hyper_parameters = study.best_params
# %% Setup dataloader
print("########################### Loading data ###########################")
qidpidtriples = MSMarcoSmallPandas(TRIPLES_SMALL_PATH)
train_dataloader = DataLoader(
    qidpidtriples,
    shuffle=True,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS
)

# %% Save settings
print("########################### Loading settings ###########################")
blackhole = '/dtu/blackhole/1b/167931/'
output_path = blackhole + "SBERT_models/HP_model"
checkpoint_path = blackhole + "SBERT_models/HP_model/checkpoints"

# %% Setup model
print("########################### Creating model ###########################")
SBERT_model = SBERT(device=DEVICE, pooling_mode=hyper_parameters['pooling_mode'])
model = SBERT_model.model

optimizer = getattr(torch.optim, hyper_parameters['optimizer'])
loss_func = getattr(losses, hyper_parameters['train_loss'])(model=model)
lr = hyper_parameters['lr']
#%% train model
print("########################### Training ###########################")
model.fit(
    train_objectives=[(train_dataloader, loss_func)],
    epochs = EPOCHS,
    warmup_steps = WARMUP_STEPS,
    use_amp = USE_AMP,
    show_progress_bar = VERBOSE,
    output_path = output_path,
    save_best_model = SAVE_MODEL,
    checkpoint_path= checkpoint_path,
    optimizer_class=optimizer,
    optimizer_params={'lr': lr}
)