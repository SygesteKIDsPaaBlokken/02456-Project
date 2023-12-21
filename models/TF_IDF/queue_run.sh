#!/bin/sh
#BSUB -q gpua100
#BSUB -J TFIDF
#BSUB -n 4
#BSUB -gpu "num=1"
#BSUB -W 2:00
#BSUB -R "rusage[mem=64GB]"
#BSUB -o out/JLOG_%J.out

echo '=================== Load modules: Started ==================='
module load python3/3.11.3
module load scipy/1.10.1-python-3.11.3
nvidia-smi
module load cuda/12.1
echo '=================== Load modules: Succeded ==================='

echo '=================== Activate environment: Start ==================='
source ~/courses/02456/02456/bin/activate
echo '=================== Activate environment: Succeded ==================='

echo '=================== Executing script: Start ==================='
python3 TF_IDF_search.py
echo '=================== Executing script: Succeded ===================