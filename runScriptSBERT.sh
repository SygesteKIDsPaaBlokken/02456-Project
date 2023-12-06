#!/bin/sh
#BSUB -J SBERT_Training
#BSUB -o out/torch_gpu_%J.out
#BSUB -e out/torch_gpu_%J.err
#BSUB -q gpua100
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -n 4
#BSUB -R "rusage[mem=32G]"
#BSUB -R "select[gpu80gb]"
#BSUB -R "span[hosts=1]"
#BSUB -W 24:00
### -- send notification at start --
#BSUB -B
### -- send notification at completion--
#BSUB -N
# end of BSUB options

echo '=================== Load modules: Started ==================='
module load python3/3.11.3
echo '=================== Load modules: Succeded ==================='

echo '=================== Activate environment: Start ==================='
source ~/courses/02456/02456/bin/activate
echo '=================== Activate environment: Succeded ==================='

echo '=================== Executing script: Start ==================='
python3 train_SBERT_small.py
echo '=================== Executing script: Succeded ==================='