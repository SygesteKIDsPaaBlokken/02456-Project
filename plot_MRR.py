import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils.cb91visuals import *

def plot_MRR(MRR_results:pd.DataFrame, save_path: str = None):
    plt.figure(figsize=(10, 5))
    #plt.title('MRR@10 scores')
    #plt.xlabel('MRR')
    #plt.ylabel('Model')
    plt.tick_params('x',labeltop=True, labelbottom=False, bottom=False, top=False)
    plt.yticks(np.arange(len(MRR_results)), MRR_results['model'])
    plt.barh(np.arange(len(MRR_results)), MRR_results['MRR']) 
    if save_path is not None:
        plt.savefig(save_path)
    plt.show()


if __name__ == '__main__':
    df = pd.read_csv('data/evaluations/MRR.csv')
    top100_models = df[df['model'].str.contains(r'^(?=.*top100)(?!.*top1000).*$')]
    print(top100_models)
    top100_models['model'] = top100_models['model'].str.replace('_top100.csv', '').str.replace('_', ' ')
    top100_models = top100_models.sort_values(by='MRR', ascending=True)
    plot_MRR(MRR_results=top100_models, save_path='assets/MRR.png')