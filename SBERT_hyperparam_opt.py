# %% Imports
import optuna
import logging
import sys

from utils.MSMarcoObjective import MSMarcoObjective

optuna.logging.get_logger("optuna")\
    .addHandler(logging.StreamHandler(sys.stdout))
# %% Study settings
study_name = "sbert_HP"  # Unique identifier of the study.
storage = optuna.storages.JournalStorage(
    optuna.storages.JournalFileStorage(f"./data/SBERT_hyperparam_opt/{study_name}.log"),  # NFS path for distributed optimization
)
# %% Creating study, SHOULD ONLY BE RUN ONCE!
# study = optuna.create_study(study_name=study_name, storage=storage)

# %% Objective
objective = MSMarcoObjective(1_000_000).objective

# %% Loading study
study = optuna.load_study(
        study_name=study_name, storage=storage
    )

# %% Run study
study.optimize(objective, n_trials=10, gc_after_trial=True)