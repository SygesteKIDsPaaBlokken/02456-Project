import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from utils.evaluation import get_CI
from utils.cb91visuals import *

DIR_PATH = 'data/evaluations/'

def filter_files(DIR_PATH):
    """Get top100 csv files from directory"""
    csv_files = [file for file in os.listdir(DIR_PATH) if file.endswith('.csv')]
    filtered_files = [file for file in csv_files if 'top1000' and 'MRR' not in file]
    
    return filtered_files

def render_bar_chart(save_fig = False, topK=5, metric='score'):
    """
    Plots bar chart
    """
    files = filter_files(DIR_PATH)
    files_to_exclude = ['Fuzzy.csv']
    files = [file for file in files if file not in files_to_exclude]
    
    parse_name = lambda x: x.replace('.csv', '').replace('_', ' ')
    data = [(parse_name(file), *get_CI(pd.read_csv(DIR_PATH + file), topK=topK, metric=metric)) for file in files]
    data_sorted = sorted(data,key=lambda x:x[1])

    for i, d in enumerate(data_sorted):
        container = plt.barh(i, d[1], label=d[0], color='b')
        plt.bar_label(container, labels=[f'{d[1]:.2f} \u00B1 {d[2]:.2f}'], label_type='edge', fontsize=8, padding=d[2]*5)
        plt.errorbar(d[1], i, xerr=d[2],fmt='o', markersize=2, color='#353839', capsize=5)

    plt.xlim(0,100)

    x = np.arange(len(files))
    plt.yticks(x, [model[0] for model in data_sorted]) 
    plt.title(f'% of relevant passages for top {topK} retrievals')
   
    if save_fig:
        plt.savefig(f'assets/accuracy_with_ci.png', transparent=True)

    plt.show()

    return data_sorted

def render_MRR(MRR_results:pd.DataFrame, save_path: str = None):
    MRR_results = MRR_results[MRR_results['model'] != 'Fuzzy']
    plt.figure(figsize=(10, 5))
    plt.yticks(np.arange(len(MRR_results)), MRR_results['model'])
    container = plt.barh(np.arange(len(MRR_results)), MRR_results['MRR']) 
    plt.bar_label(container, label_type='edge', fontsize=12, padding=5, fmt='%.2f')
    if save_path is not None:
        plt.savefig(save_path)
    plt.show()


if __name__ == '__main__':
    render_bar_chart(save_fig=True, topK=10, metric='score') # Beware, topK=10
    render_bar_chart(save_fig=True, topK=10, metric='count') # Beware, topK=10

    # MRR
    df = pd.read_csv('data/evaluations/MRR.csv')
    top100_models = df[df['model'].str.contains(r'^(?=.*top100)(?!.*top1000).*$')]
    
    top100_models['model'] = top100_models['model'].str.replace('_top100.csv', '').str.replace('_', ' ')
    top100_models = top100_models.sort_values(by='MRR', ascending=True)
    render_MRR(MRR_results=top100_models, save_path='assets/MRR.png')
    