# %% Imports
import optuna

# %% Load previous study
study_name = 'sbert_HP'
storage = optuna.storages.JournalStorage(
    optuna.storages.JournalFileStorage(f"./data/SBERT_hyperparam_opt_minimized/{study_name}.log"),  # NFS path for distributed optimization
)
prev_study = optuna.load_study(study_name, storage)

# %% Load new study
study_name = 'sbert_HP_1m'
storage = optuna.storages.JournalStorage(
    optuna.storages.JournalFileStorage(f"./data/SBERT_hyperparam_opt_minimized/{study_name}.log"),  # NFS path for distributed optimization
)
new_study = optuna.load_study(study_name, storage)

# %% Merge
study_name = 'sbert_HP_1m_merge'
storage = optuna.storages.JournalStorage(
    optuna.storages.JournalFileStorage(f"./data/SBERT_hyperparam_opt_minimized/{study_name}.log"),  # NFS path for distributed optimization
)
merge_study = optuna.create_study(study_name=study_name, storage=storage, direction=optuna.study.StudyDirection.MAXIMIZE)
merge_study.add_trials(prev_study.trials)
merge_study.add_trials(new_study.trials)

# %% Add minimizing trials to new maximizing study
study_name = 'sbert_HP_1m'
storage = optuna.storages.JournalStorage(
    optuna.storages.JournalFileStorage(f"./data/SBERT_hyperparam_opt/{study_name}.log"),  # NFS path for distributed optimization
)

study = optuna.load_study(study_name=study_name, storage=storage)
# %%
study.add_trials(merge_study.trials)