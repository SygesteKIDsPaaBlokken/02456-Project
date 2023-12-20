from pathlib import Path

from optuna.trial import Trial

import torch
from torch.utils.data import DataLoader
from sentence_transformers import losses

from models.SBERT import SBERT
from utils.config import DEVICE, USE_AMP, VERBOSE, SAVE_MODEL, WARMUP_STEPS, NUM_WORKERS, BATCH_SIZE, TRIPLES_SMALL_PATH
from utils.MSMarcoDatasetSmall import MSMarcoSmallPandas
from utils.MSMarcoEvaluator import make_evaluator

class MSMarcoObjective:
    def __init__(self, run_id: str, limit: int = None) -> None:

        qidpidtriples = MSMarcoSmallPandas(TRIPLES_SMALL_PATH, limit=limit)
        self.train_dataloader = DataLoader(
            qidpidtriples,
            shuffle=True,
            batch_size=BATCH_SIZE,
            num_workers=NUM_WORKERS,
        )
        self.limit = limit

        self.evaluator = make_evaluator()
        self.last_evaluation_score = None
        self.run_id = run_id

    def objective(self, trial: Trial):
        pooling_mode = trial.suggest_categorical("pooling_mode", ["mean", "max", "cls"])
        model = SBERT(pooling_mode=pooling_mode).model
        model.to(DEVICE)

        optimizer_name = trial.suggest_categorical("optimizer", ['AdamW', 'SGD'])
        optimizer_class = getattr(torch.optim, optimizer_name)

        lr = trial.suggest_float("lr", 1e-7, 1e-3, log=True)

        train_loss_name = trial.suggest_categorical("train_loss", ['MultipleNegativesRankingLoss', 'TripletLoss'])
        train_loss_class = getattr(losses, train_loss_name)
        train_loss = train_loss_class(model=model)

        n_steps_epoch = self.limit // BATCH_SIZE
        self.best_evaluation_score = 0

        model.fit(
            train_objectives=[(self.train_dataloader, train_loss)],
            epochs = 1,
            evaluator=self.evaluator,
            evaluation_steps=n_steps_epoch//10,
            callback=lambda score,epoch,step: trial.report(*self.evaluator_out_to_trial_report_in(n_steps_epoch, score, epoch, step)),
            warmup_steps = WARMUP_STEPS,
            optimizer_class=optimizer_class,
            optimizer_params={'lr': lr},
            use_amp = USE_AMP,
            show_progress_bar = VERBOSE,
            output_path = f'trained_models/hyper_param/{self.run_id}',
            save_best_model = SAVE_MODEL,
        )

        return self.last_evaluation_score

    def evaluator_out_to_trial_report_in(self, n_steps_epoch: int, score: float, epoch: int, step: int) -> tuple[int, int]:
        self.last_evaluation_score = score
        total_steps = 0

        if step == -1:
            total_steps = (epoch + 1) * n_steps_epoch
        
        else:
            total_steps = epoch*n_steps_epoch + step

        if score > self.best_evaluation_score:
            if VERBOSE: print(f"Epoch: {epoch}, step: {step}, total steps: {total_steps}, score: {score}")
            self.best_evaluation_score = score
        
        return score, total_steps
    