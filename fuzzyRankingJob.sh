#BSUB -J Fuzzy_ranking
#BSUB -o out/Fuzzy_ranking_%J.out
#BSUB -e out/Fuzzy_ranking_%J.err
#BSUB -q HPC
#BSUB -n 4
#BSUB -R "rusage[mem=8G]"
#BSUB -R "span[hosts=1]"
#BSUB -W 04:00
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
python3 create_model_rankings.py --model Fuzzy
echo '=================== Executing script: Succeded ==================='
