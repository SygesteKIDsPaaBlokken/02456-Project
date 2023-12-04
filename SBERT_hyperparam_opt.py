# %% Imports
import optuna
import logging
import sys
import time

optuna.logging.get_logger("optuna")\
    .addHandler(logging.StreamHandler(sys.stdout))
# %% Study settings
study_name = "sbert_HP"  # Unique identifier of the study.
storage = optuna.storages.JournalStorage(
    optuna.storages.JournalFileStorage(f"./data/SBERT_hyperparam_opt/{study_name}.log"),  # NFS path for distributed optimization
)
# %% Creating study, SHOULD ONLY BE RUN ONCE!
study = optuna.create_study(study_name=study_name, storage=storage)

# %% Objective
def objective(trial):
    x = trial.suggest_float("x", -10, 10)
    time.sleep(5)
    return (x - 2) ** 2

# %% Loading study
study = optuna.load_study(
        study_name=study_name, storage=storage
    )

# %% Run study
study.optimize(objective, n_trials=10)