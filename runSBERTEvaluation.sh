#BSUB -J EVAL_SBERT
#BSUB -o out/SBERTeval_%J.out
#BSUB -e out/SBERTeval_%J.err
#BSUB -q gpuv100
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -n 4
#BSUB -R "rusage[mem=8G]"
###BSUB -R "select[gpu80gb]"
#BSUB -R "span[hosts=1]"
#BSUB -W 24:00
### -- send notification at start --
#BSUB -B
### -- send notification at completion--
#BSUB -N
# end of BSUB options
echo '=================== Load modules: Started ==================='
module load python3/3.9.11
echo '=================== Load modules: Succeded ==================='

echo '=================== Activate environment: Start ==================='
source .venv/bin/activate
echo '=================== Activate environment: Succeded ==================='

echo '=================== Executing script: Start ==================='
python3 rank_sbert.py
echo '=================== Executing script: Succeded ==================='
