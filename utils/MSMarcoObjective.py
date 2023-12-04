from pathlib import Path

from optuna.trial import Trial

import torch
from torch.utils.data import DataLoader
from sentence_transformers import losses

from models.SBERT import SBERT
from utils.config import DEVICE, USE_AMP, VERBOSE, SAVE_MODEL, WARMUP_STEPS, NUM_WORKERS, BATCH_SIZE, DATA_FOLDER, TRIPLES_SMALL_PATH
from utils.MSMarcoDatasetSmall import MSMarcoSmallPandas
from utils.MSMarcoDatasetDev2 import make_evaluator

class MSMarcoObjective:
    def __init__(self, limit: int = None) -> None:

        qidpidtriples = MSMarcoSmallPandas(TRIPLES_SMALL_PATH, limit=limit)
        self.train_dataloader = DataLoader(
            qidpidtriples,
            shuffle=True,
            batch_size=BATCH_SIZE,
            num_workers=NUM_WORKERS,
        )

        self.evaluator = make_evaluator()

    def objective(self, trial: Trial):
        pooling_mode = trial.suggest_categorical("pooling_mode", ["mean", "max", "cls"])
        model = SBERT(pooling_mode=pooling_mode).model
        model.to(DEVICE)

        optimizer_name = trial.suggest_categorical("optimizer", ['AdamW', 'SGD'])
        optimizer_class = getattr(torch.optim, optimizer_name)

        lr = trial.suggest_loguniform("lr", 1e-7, 1e-3)

        train_loss = losses.MultipleNegativesRankingLoss(model=model)

        model.fit(
            train_objectives=[(self.train_dataloader, train_loss)],
            epochs = 10,
            evaluator=self.evaluator,
            evaluation_steps=100_000,
            callback=trial.report,
            warmup_steps = WARMUP_STEPS,
            optimizer_class=optimizer_class,
            optimizer_params={'lr': lr},
            use_amp = USE_AMP,
            show_progress_bar = VERBOSE,
            output_path = 'trained_models/1epoch',
            save_best_model = SAVE_MODEL,
        )
    