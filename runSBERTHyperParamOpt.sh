#!/bin/sh
### General options
### –- specify queue: gpuv100, gputitanxpascal, gpuk40, gpum2050 --
#BSUB -q gpuv100
### -- set the job Name --
#BSUB -J SBERT_HP
##BSUB -gpu "num=1:mode=exclusive_process"
##BSUB -n 4
#BSUB -R "span[hosts=1]"
### -- set walltime limit: hh:mm --  maximum 24 hours for GPU-queues right now
#BSUB -W 24:00
### -- request 32GB of memory
#BSUB -R "rusage[mem=40GB]"
### -- set the email address --
# please uncomment the following line and put in your e-mail address,
# if you want to receive e-mail notifications on a non-default address
## BSUB -u fasco@dtu.dk
### -- send notification at start --
## BSUB -B
### -- send notification at completion--
#BSUB -N
### -- Specify the output and error file. %J is the job-id --
### -- -o and -e mean append, -oo and -eo mean overwrite --
#BSUB -o out/output_file-%J.out
#BSUB -e out/error_file-%J.err
# -- end of LSF options --
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
python3 SBERT_hyperparam_opt.py
echo '=================== Executing script: Succeded ===================
