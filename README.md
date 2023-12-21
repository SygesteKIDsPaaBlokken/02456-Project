# 02456-Project
This project compares different information retrieval methods using [the Passage ranking dataset](https://microsoft.github.io/msmarco/TREC-Deep-Learning-2019#passage-ranking-dataset) from MS MARCO dataset from TREC 2019.


# Setup
## Conda
Create environment and install dependencies:
```
conda env create -f env.yml
```

## venv and pip
```
python -m venv SBERT
source SBERT/bin/activate
pip install -r requirements.txt
```

# Models
The implementation of the SBERT model can be found in the [models folder](models).

# Config
The configuration for training parameters, data paths etc can be found in [config](utils/config.py).

# Training
## MSMarco Triplet Small
To train a SBERT model from scratch use:
```
python train_new_SBERT_small.py
```

To continue training a SBERT model use:
```
python train_existing_SBERT_SMALL.py
```

# Local data
/dtu/blackhole/1a/163226
- triples.train.small.tsv
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