#BSUB -J EVAL_SBERT
#BSUB -o out/HPSBERTeval_%J.out
#BSUB -e out/HPSBERTeval_%J.err
#BSUB -q gpua100
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -n 4
#BSUB -R "rusage[mem=24G]"
###BSUB -R "select[gpu80gb]"
#BSUB -R "span[hosts=1]"
#BSUB -W 12:00
### -- send notification at start --
#BSUB -B
### -- send notification at completion--
#BSUB -N
# end of BSUB options
echo '=================== Load modules: Started ==================='
module load python3/3.11.3
echo '=================== Load modules: Succeded ==================='

echo '=================== Activate environment: Start ==================='
source venv/bin/activate
echo '=================== Activate environment: Succeded ==================='

echo '=================== Executing script: Start ==================='
python3 evaluate_HP_SBERT.py
echo '=================== Executing script: Succeded ==================='
