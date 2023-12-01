from time import time
import numpy as np
import pandas as pd
from tqdm import tqdm


if __name__ == '__main__':
    N = 1_000_000
    t = time()
    df_queries = pd.read_csv('data/queries.train.tsv', sep='\t', header=None)
    # tripletsNew = pd.read_csv('tripletsReduced.tsv', sep='\t', header=None) # For convenience while creating the files
    dfTriplets = pd.read_csv('data/qidpidtriples.train.full.2.tsv', sep='\t', header=None)
    dfCollection = pd.read_csv('data/collection.tsv', sep='\t', header=None)
    dfScoring = pd.read_csv('data/0-3scoringTestSet.txt', sep=' ')
    print(f'Loading CSV files took {time() - t:.4} seconds')
    t = time()
    essentialPassages = dfScoring['passage'].unique()
    essentialPassages = essentialPassages[essentialPassages > N]
    collectNew = np.vstack((dfCollection.iloc[:N], dfCollection.iloc[essentialPassages]))

    N2 = len(collectNew)
    tripletsNew = dfTriplets.iloc[np.logical_or((dfTriplets.values[:, 1] < N2), (dfTriplets.values[:, 2] < N2))]

    queries_new = []
    unique_queries = np.unique(tripletsNew.values[:, 0])
    for row in tqdm(df_queries.values[:, :]):
        if sum(row[0] == unique_queries):
            queries_new.append(row)

    pd.DataFrame(np.array(queries_new).T).to_csv('data/queries.trainReduced.tsv', sep='\t', index=False, header=False)
    pd.DataFrame(collectNew).to_csv('data/collectionReduced.tsv', sep='\t', index=False, header=False)
    pd.DataFrame(tripletsNew).to_csv('data/tripletsReduced.tsv', sep='\t', index=False, header=False)
    print(f'Saving took {time() - t:.4} seconds')
    pass
