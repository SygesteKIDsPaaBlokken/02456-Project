import numpy as np
import pandas as pd
from pandas import DataFrame
from tqdm import tqdm
import os, datetime, argparse

def MRRRank(dfEval, dfModel, model_name, MaxMRRRank=10):
    '''
    Calculates MRR score for the given model and evaluation data using MRR@10 as default.
    Model must be structured as columns ['qid', 'pid', 'score'].
    '''

    query_ids = np.unique(dfEval.iloc[:,0])
    results = []

    MRR = 0
    ranking = []



    for n, qid in enumerate(tqdm(query_ids)):

        query = dfEval[dfEval.iloc[:,0] == qid]
        retrievals = dfModel[dfModel.iloc[:,0]==qid]

        relevant_passages = query.loc[query['score'] > 0,'passage'].values

        if qid in np.unique(dfModel.iloc[:,0]):
            for i in range(0,MaxMRRRank):

                passage = retrievals.iloc[i,1]

                if passage in relevant_passages:
                    MRR += 1/(i+1)
                    ranking.append(i+1)
                    print(f"Found first relevant retrieval at rank: {i+1} for test query #{n+1}")
                    break

    MRR /= len(query_ids)
    print('='*30)
    print(f'MRR@{MaxMRRRank}: {MRR:.4f}')
    print('Number of test queries: ', len(query_ids))
    print('='*30)

    path = 'data/evaluations/MRR.csv'
    if os.path.exists(path):
        pd.DataFrame({'model':model_name,'MRR':MRR}, index=[0]).to_csv(path, mode='a', header=False, index=False)
    else:
        print('Creating new file data/MRR.csv since it does not exist already')
        pd.DataFrame({'model':model_name,'MRR':MRR}, index=[0]).to_csv(path, index=False)
    return results
if __name__ == '__main__':
    

    parser = argparse.ArgumentParser(description='MRR evaluation script')
    parser.add_argument('--dir', type=str, default='data/rankings/', help='Path to directory containing rankings')
    parser.add_argument('--eval_file', type=str, default='data\\0-3scoringTestSet.txt', help='Path to evaluation file')
    args = parser.parse_args()
    csv_files = [filename for filename in os.listdir(args.dir) if filename.endswith('.csv')]
    dfEval = pd.read_csv(args.eval_file, sep=' ')

    for f in csv_files:
        name = os.path.basename(f)
        dfModel =  pd.read_csv(args.dir + f)
        if dfModel.columns[0] == 'Unnamed: 0':
            dfModel = dfModel.drop(columns=['Unnamed: 0'])
       
        MRRRank(dfEval, dfModel, name)

    MRR_path = 'data/evaluations/MRR.csv'
    pd.read_csv(MRR_path).sort_values(by='MRR', ascending=False).to_csv(MRR_path, index=False)
