#!/bin/sh
### General options
### –- specify queue: gpuv100, gputitanxpascal, gpuk40, gpum2050 --
#BSUB -q hpc
### -- set the job Name --
#BSUB -J reduce
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
#BSUB -u fasco@dtu.dk
### -- send notification at start --
#BSUB -B
### -- send notification at completion--
#BSUB -N
### -- Specify the output and error file. %J is the job-id --
### -- -o and -e mean append, -oo and -eo mean overwrite --
#BSUB -o output_file-%J.out
#BSUB -e error_file-%J.err
# -- end of LSF options --

source venv/bin/activate

python3 reduce_dataset.py
