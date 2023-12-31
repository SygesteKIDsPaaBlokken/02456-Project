#BSUB -J SBERT_ranking
#BSUB -o out/SBERT_ranking_%J.out
#BSUB -e out/SBERT_ranking_%J.err
#BSUB -q gpuv100
#BSUB -gpu "num=1"
#BSUB -n 4
#BSUB -R "rusage[mem=8G]"
###BSUB -R "select[gpu80gb]"
#BSUB -R "span[hosts=1]"
#BSUB -W 01:00
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
python3 create_model_rankings.py --model SBERT_1e
echo '=================== Executing script: Succeded ==================='
