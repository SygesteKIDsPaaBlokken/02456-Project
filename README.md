# 02456-Project
This project compares different information retrieval methods using [the Passage ranking dataset](https://microsoft.github.io/msmarco/TREC-Deep-Learning-2019#passage-ranking-dataset) from MS MARCO dataset from TREC 2019.


# Setup
Create environment and install dependencies:
```
conda env create -f env.yml
```

# Models
The implementation of the different models can be found in the [models folder](models).

# Local data
/dtu/blackhole/1a/163226
- collection.tsv
- queries.eval.tsv
- queries.train.tsv
- queries.dev.tsv

# Evaluation
**MRR@10**
To get MRR@10 score and append it to data/MRR.csv
```py
python utils/evaluation.py --model data/sbert_RankingResults.csv --name SBERT_1epoch 
```
For help use
```
python utils/evaluation.py --help
```