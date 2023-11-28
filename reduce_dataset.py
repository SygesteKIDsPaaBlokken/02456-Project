from time import time
import numpy as np
import pandas as pd


if __name__ == '__main__':
    N = 1_000_000
    t = time()
    dfTriplets = pd.read_csv('qidpidtriples.train.full.2.tsv', sep='\t', header=None)
    dfCollection = pd.read_csv('Fasttext/collection.tsv', sep='\t', header=None)
    dfScoring = pd.read_csv('0-3scoringTestSet.txt', sep=' ')
    print(f'Loading CSV files took {time() - t:.4} seconds')
    t = time()
    essentialPassages = dfScoring['passage'].unique()
    essentialPassages = essentialPassages[essentialPassages > N]
    collectNew = np.vstack((dfCollection.iloc[:N], dfCollection.iloc[essentialPassages]))

    N2 = len(collectNew)
    tripletsNew = dfTriplets.iloc[np.logical_or((dfTriplets.values[:, 1] < N2), (dfTriplets.values[:, 2] < N2))]

    pd.DataFrame(collectNew).to_csv('Fasttext/collectionReduced.tsv', sep='\t', index=False, header=False)
    pd.DataFrame(tripletsNew).to_csv('tripletsReduced.tsv', sep='\t', index=False, header=False)
    print(f'Saving took {time() - t:.4} seconds')
