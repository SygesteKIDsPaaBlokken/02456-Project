import torch
from torch.utils.data import DataLoader
from sentence_transformers import losses

from models.SBERT import SBERT
from utils.config import DEVICE, USE_AMP, VERBOSE, SAVE_MODEL, WARMUP_STEPS, NUM_WORKERS, BATCH_SIZE, TRIPLES_SMALL_PATH
from utils.MSMarcoDatasetSmall import MSMarcoSmallPandas
from utils.MSMarcoDatasetDev2 import make_evaluator

triplets_path = TRIPLES_SMALL_PATH

TRAIN_DATA_LIMIT = 1_000_000
qidpidtriples = MSMarcoSmallPandas(triplets_path, limit=TRAIN_DATA_LIMIT)
train_dataloader = DataLoader(
    qidpidtriples,
    shuffle=True,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
)

evaluator = make_evaluator()

def objective(trial):
    pooling_mode = trial.suggest_categorical("pooling_mode", ["mean", "max", "cls"])
    model = SBERT(pooling_mode=pooling_mode).model
    model.to(DEVICE)

    optimizer_class = trial.suggest_categorical("optimizer", [torch.optim.AdamW, torch.optim.SGD])
    lr = trial.suggest_loguniform("lr", 1e-7, 1e-3)

    train_loss = losses.MultipleNegativesRankingLoss(model=model)
    total_steps = TRAIN_DATA_LIMIT // BATCH_SIZE
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs = 5,
        evaluator=evaluator,
        evaluation_steps=2_000,
        callback= lambda score,epoch,step: trial.report(score, epoch*total_steps + step),
        warmup_steps = WARMUP_STEPS,
        optimizer_class=optimizer_class,
        optimizer_params={'lr': lr},
        use_amp = USE_AMP,
        show_progress_bar = VERBOSE,
        output_path = 'trained_models/1epoch',
        save_best_model = SAVE_MODEL,
    )
