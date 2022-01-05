#!/bin/zsh

#$ -l h_gpu=1
#$ -l m_mem_free=45G
#$ -cwd
#$ -V
#$ -e error_log_$JOB_ID
#$ -o out_log_$JOB_ID
#$ -l h_rt=12:00:00
###$ -l hostname=maxg01
###$ -l cuda_name=Tesla-V100-SXM2-16GB

export CUDA_VISIBLE_DEVICES=0

python run_auccpvloss.py "$@"
retVal=$?
if [ $retVal -ne 0 ]; then
    echo "Error"
	exit 100
fi
