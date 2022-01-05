#!/bin/zsh

#$ -l m_mem_free=30G
#$ -cwd
#$ -V
#$ -e error_log_$JOB_ID
#$ -o out_log_$JOB_ID
#$ -l h_rt=24:00:00

python run_auccpvloss.py "$@"
retVal=$?
if [ $retVal -ne 0 ]; then
    echo "Error"
	exit 100
fi
