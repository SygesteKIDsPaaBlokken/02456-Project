import numpy as np
import pandas as pd
from pandas import DataFrame
from tqdm import tqdm
import os, datetime, argparse

from utils.config import MODEL_NAME


def MRRRank(dfEval, dfModel, model_name, MaxMRRRank=10):
    '''
    Calculates MRR score for the given model and evaluation data using MRR@10 as default.
    '''
    query_ids = np.unique(dfEval['query'])
    results = []

    MRR = 0
    ranking = []

    for n, qid in enumerate(tqdm(query_ids)):

        query = dfEval[dfEval['query'] == qid]
        retrievals = dfModel[dfModel['query']==qid]
        relevant_passages = query.loc[query['score'] > 0,'passage'].values

        if qid in np.unique(dfModel['query']):
            for i in range(0,MaxMRRRank):
                if retrievals['passage'].iloc[i] in relevant_passages:
                    MRR += 1/(i+1)
                    ranking.append(i+1)
                    print(f"Found first relevant retrieval at rank: {i+1} for test query #{n+1}")
                    break

    MRR /= len(query_ids)
    print('='*30)
    print(f'MRR@{MaxMRRRank}: {MRR:.4f}')
    print('Number of test queries: ', len(query_ids))
    print('='*30)

    path = 'data/MRR.csv'
    if os.path.exists(path):
        pd.DataFrame({'model':model_name,'MRR':MRR,'timestamp':datetime.datetime.now()}, index=[0]).to_csv(path, mode='a', header=False, index=False)
    else:
        print('Creating new file data/MRR.csv since it does not exist already')
        pd.DataFrame({'model':model_name,'MRR':MRR,'timestamp':datetime.datetime.now()}, index=[0]).to_csv(path, index=False)
    return results

def evaluate_model(
        eval_ds: DataFrame,
        model_rankings: DataFrame,
        top_k: list[int],
        model_name: str = MODEL_NAME,
        save_evaluation: bool = True,
        save_path: str = None
    ):

    save_evaluation = save_evaluation or save_path is not None
    if save_path is None: save_path = f'data/evaluations/{model_name}.csv'

    k_scores = [0 for _ in top_k]
    k_counts = [0 for _ in top_k]
    k_max_scores = [0 for _ in top_k]
    k_max_counts = [0 for _ in top_k]

    query_ids = np.unique(eval_ds['query'])
    for qid in tqdm(query_ids):
        
        # Find the relevant rows
        query_eval_rows = eval_ds[eval_ds['query'] == qid]
        for i, k in enumerate(top_k):
            # Take the top k scores in the evaluation data, to find the maximum possible score
            query_true_best_scores = query_eval_rows.sort_values('score',ascending=False)[:k]['score']
            k_max_scores[i] += query_true_best_scores.sum()
            k_max_counts[i] += (query_true_best_scores>0).sum()

            # Determine the top k summed score for the query
            top_k_passages = model_rankings[model_rankings['qid'] == qid][:k]['pid']
            evaluation_scores = query_eval_rows.loc[query_eval_rows['passage'].isin(top_k_passages),'score']
            
            k_scores[i] += evaluation_scores.sum()
            k_counts[i] += (evaluation_scores > 0).sum()
    
    model_evaluation = DataFrame({
        'topK':top_k,
        'score':k_scores,
        'max_score':k_max_scores,
        'count':k_counts,
        'max_count':k_max_counts
    })

    if save_evaluation:
        model_evaluation.to_csv(save_path, index=False)
    
    return model_evaluation

def top1000Evaluation(dfTop, dfModel, topK, name):
    # USELESS
    results = [[] for _ in topK]
    for qid in tqdm(np.unique(dfTop.values[:, 0])):
        # Find the relevant columns to speed up computation:
        evalRows = dfTop[dfTop.values[:, 0] == qid].values[:, :2]
        for i, k in enumerate(topK):
            # Could be a list comprehension:
            passagesFT = dfModel[dfModel['query'] == qid][:k]['passage']
            # Check for each passage whether it is there
            results[i].append(sum([sum(passage == evalRows[:, 1]) for passage in passagesFT]))

    print(f'{name} (average overlap)')
    sFT = ''
    for k, resultFT in zip(topK, results):
        sFT += f'{k}: {np.average(resultFT):.2}/{k}\t'
    print(sFT)
    return results



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='MRR evaluation script')
    parser.add_argument('--eval_file', type=str, default='data\\0-3scoringTestSet.txt', help='Path to evaluation file')
    parser.add_argument('--model', type=str, help='Path to model file', required=True)
    parser.add_argument('--name', type=str, help='Name of model to be saved as', required=True)
    args = parser.parse_args()


    dfEval = pd.read_csv(args.eval_file, sep=' ')
    dfModel = pd.read_csv(args.model)

    print('Evaluation model:', args.model)
    MRRRank(dfEval, dfModel, args.name)
