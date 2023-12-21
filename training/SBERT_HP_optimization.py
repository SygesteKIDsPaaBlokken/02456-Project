# %% Imports
import optuna
import logging
import sys
from uuid import uuid4

from utils.MSMarcoObjective import MSMarcoObjective
from utils.config import DEVICE, USE_AMP, WARMUP_STEPS, BATCH_SIZE

optuna.logging.get_logger("optuna")\
    .addHandler(logging.StreamHandler(sys.stdout))
# %% Study settings
study_name = "sbert_HP_1m"  # Unique identifier of the study.
storage = optuna.storages.JournalStorage(
    optuna.storages.JournalFileStorage(f"./data/SBERT_hyperparam_opt/{study_name}.log"),  # NFS path for distributed optimization
)
# %% Creating study, SHOULD ONLY BE RUN ONCE!
# study = optuna.create_study(study_name=study_name, storage=storage, direction=optuna.study.StudyDirection.MAXIMIZE)

# %% Generate run/job id
run_id = str(uuid4())

# %% Objective
data_size = 1_000_000
objective = MSMarcoObjective(run_id, data_size).objective

# %% Loading study
study = optuna.load_study(
        study_name=study_name, storage=storage
    )

# %% Run study
print(f"""###################### Hyperparameter tuning: ######################
Run id: {run_id}
Batch size: {BATCH_SIZE}
Warmup steps: {WARMUP_STEPS}
Use AMP: {USE_AMP}
Data size: {data_size}
Device: {DEVICE}

Study: {study_name}
""")
study.optimize(objective, timeout=23*60*60, gc_after_trial=True)