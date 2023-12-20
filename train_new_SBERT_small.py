#%% General imports
from torch.utils.data import DataLoader

from sentence_transformers import losses

from models.SBERT import SBERT
from utils.MSMarcoDatasetSmall import MSMarcoSmallPandas
from utils.config import TRIPLES_SMALL_PATH, DEVICE, BATCH_SIZE, NUM_WORKERS, USE_AMP, VERBOSE, WARMUP_STEPS, EPOCHS, SAVE_MODEL, DATA_FOLDER

# %% Setup dataloader
qidpidtriples = MSMarcoSmallPandas(TRIPLES_SMALL_PATH)
train_dataloader = DataLoader(
    qidpidtriples,
    shuffle=True,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS
)

# %% Save settings
blackhole = '/dtu/blackhole/1b/167931/'
output_path = blackhole + "SBERT_models/3epochs"
checkpoint_path = blackhole + "SBERT_models/3epochs_checkpoints"


# %% Setup model
SBERT_model = SBERT(device=DEVICE)
model = SBERT_model.model
model.to(DEVICE)

#%% Setup loss
train_loss = losses.MultipleNegativesRankingLoss(model=model)

#%% train model
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs = EPOCHS,
    warmup_steps = WARMUP_STEPS,
    use_amp = USE_AMP,
    show_progress_bar = VERBOSE,
    output_path = output_path,
    save_best_model = SAVE_MODEL,
    checkpoint_path= checkpoint_path
)