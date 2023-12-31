import numpy as np
import pandas as pd
from pandas import DataFrame
from tqdm import tqdm
import os, datetime, argparse, math
import scipy.stats as stats

from utils.config import MODEL_NAME

def get_CI(df:pd.DataFrame, topK=5, alpha = 0.05, metric='score'):
    row = df[df['topK'] == topK].iloc[0]
    n = row[f'max_{metric}']
    relevant = row[metric]
    p_hat = relevant/n
    z_critical = stats.norm.ppf(1 - alpha / 2)
    margin_of_error = z_critical * math.sqrt((p_hat * (1 - p_hat)) / n)
    lower_bound = p_hat - margin_of_error
    upper_bound = p_hat + margin_of_error
    return p_hat*100, margin_of_error*100, lower_bound*100, upper_bound*100

def MRRRank(dfEval, dfModel, model_name, MaxMRRRank=10, mode='a',show_progress=False):
    '''
    Calculates MRR score for the given model and evaluation data using MRR@10 as default.
    '''
    query_ids = np.unique(dfEval.iloc[:,0])

    MRR = 0
    ranking = []

    for n, qid in enumerate(tqdm(query_ids,disable=not show_progress)):

        query = dfEval[dfEval.iloc[:,0] == qid]
        retrievals = dfModel[dfModel['qid']==qid]
        relevant_passages = query.loc[query.iloc[:, 3] > 0, [2]].values

        if qid in np.unique(dfModel['qid']):
            for i in range(0,MaxMRRRank):
                if retrievals['pid'].iloc[i] in relevant_passages:
                    MRR += 1/(i+1)
                    ranking.append(i+1)
                    #print(f"Found first relevant retrieval at rank: {i+1} for test query #{n+1}")
                    break

    MRR /= len(query_ids)
    if show_progress:
        print('='*30)
        print(f'MRR@{MaxMRRRank}: {MRR:.4f}')
        print('Number of test queries: ', len(query_ids))
        print('='*30)

    path = 'data/MRR.csv'

    if mode == 'a' and os.path.exists(path):
        pd.DataFrame({'model':model_name,'MRR':MRR,'timestamp':datetime.datetime.now()}, index=[0]).to_csv(path, mode='a', header=False, index=False)
    elif mode == 'a' and not os.path.exists(path):
        print('Creating new file data/MRR.csv since it does not exist already')
        pd.DataFrame({'model':model_name,'MRR':MRR,'timestamp':datetime.datetime.now()}, index=[0]).to_csv(path, index=False)
    return MRR

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

    query_ids = np.unique(eval_ds.iloc[:,0])
    for qid in tqdm(query_ids):
        
        # Find the relevant rows
        query_eval_rows = eval_ds[eval_ds.iloc[:,0] == qid]
        for i, k in enumerate(top_k):
            # Take the top k scores in the evaluation data, to find the maximum possible score
            query_true_best_scores = query_eval_rows.sort_values(query_eval_rows.columns[3],ascending=False)[:k].iloc[:,3]
            k_max_scores[i] += query_true_best_scores.sum()
            k_max_counts[i] += (query_true_best_scores>0).sum()

            # Determine the top k summed score for the query
            top_k_passages = model_rankings[model_rankings['qid'] == qid][:k]['pid']
            evaluation_scores = query_eval_rows.loc[query_eval_rows.iloc[:, 2].isin(top_k_passages), query_eval_rows.columns[3]]
            
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

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='MRR evaluation script')
    parser.add_argument('--eval_file', type=str, default='data\\0-3scoringTestSet.txt', help='Path to evaluation file')
    parser.add_argument('--model', type=str, help='Path to model file', required=True)
    parser.add_argument('--name', type=str, help='Name of model to be saved as', required=True)
    args = parser.parse_args()


    dfEval = pd.read_csv(args.eval_file, sep=' ')
    dfModel = pd.read_csv(args.model)
    print(dfEval)
    print('Evaluation model:', args.model)
    MRRRank(dfEval, dfModel, args.name)
