# %% Imports
import pandas as pd
from utils.config import RANKING_PATH
import os

# %% Get rankings to convert
ranking_files_to_clean = [f for f in os.listdir(RANKING_PATH) if 'top1000' in f]
ranking_files_to_clean
# %% Generate top 100 rankings
columns_to_transform = {
    'query':'qid',
    'passage': 'pid'
}

for f in ranking_files_to_clean:
    rankings_top1000 = pd.read_csv(RANKING_PATH.joinpath(f), index_col=0)

    # Get top100
    rankings_top100 = rankings_top1000.loc[
        rankings_top1000.groupby('query')['score']\
            .nlargest(100)\
            .index\
            .get_level_values(1)
    ]

    # Rename the columns
    rankings_top100.rename(columns=columns_to_transform, inplace=True)

    # Convert ids from floats to ints
    rankings_top100 = rankings_top100.astype({c:'int' for c in columns_to_transform.values()})

    # Save
    rankings_top100.to_csv(RANKING_PATH.joinpath(f.replace('top1000','top100')), index=False)

# %% Clean top 1000 rankings
for f in ranking_files_to_clean:
    rankings_top1000 = pd.read_csv(RANKING_PATH.joinpath(f), index_col=0)

    # Rename the columns
    rankings_top1000.rename(columns=columns_to_transform, inplace=True)

    # Convert ids from floats to ints
    rankings_top1000 = rankings_top1000.astype({c:'int' for c in columns_to_transform.values()})

    # Save
    rankings_top1000.to_csv(RANKING_PATH.joinpath(f), index=False)