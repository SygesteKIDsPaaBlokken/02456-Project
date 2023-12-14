# %% Imports
import optuna

# %% Load previous study
study_name = 'sbert_HP'
storage = optuna.storages.JournalStorage(
    optuna.storages.JournalFileStorage(f"./data/SBERT_hyperparam_opt/{study_name}.log"),  # NFS path for distributed optimization
)
prev_study = optuna.load_study(study_name, storage)

# %% Load new study
study_name = 'sbert_HP_1m'
storage = optuna.storages.JournalStorage(
    optuna.storages.JournalFileStorage(f"./data/SBERT_hyperparam_opt/{study_name}.log"),  # NFS path for distributed optimization
)
new_study = optuna.load_study(study_name, storage)

new_study.add_trials(prev_study.trials)