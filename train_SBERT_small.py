#%% General imports
from torch.utils.data import DataLoader

from sentence_transformers import losses

from models.SBERT import SBERT
from utils.MSMarcoDatasetSmall import MSMarcoSmallPandas
from utils.config import TRIPLES_SMALL_PATH, DEVICE, BATCH_SIZE, NUM_WORKERS, USE_AMP, VERBOSE, WARMUP_STEPS, EPOCHS, SAVE_MODEL

# Setup dataloader
qidpidtriples = MSMarcoSmallPandas(TRIPLES_SMALL_PATH)
train_dataloader = DataLoader(
    qidpidtriples,
    shuffle=True,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS
)

# %% Setup model
SBERT_model = SBERT()
model = SBERT_model.model

model.to(DEVICE)
print(model.device)

#%% Setup loss
train_loss = losses.MultipleNegativesRankingLoss(model=model)

#%% train model
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs = EPOCHS,
    warmup_steps = WARMUP_STEPS,
    use_amp = USE_AMP,
    show_progress_bar = VERBOSE,
    output_path = '/dtu/blackhole/1b/167931/SBERT_models/2epochs',
    save_best_model = SAVE_MODEL,
    checkpoint_path='/dtu/blackhole/1b/167931/SBERT_models/2epochs_checkpoints'
)
