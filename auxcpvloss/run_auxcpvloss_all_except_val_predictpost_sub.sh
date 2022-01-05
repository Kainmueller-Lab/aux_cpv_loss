#!/bin/zsh

jobid2=$(qsub -terse ./run_auccpvloss_cpu.sh -d validate_checkpoints "$@")
echo "submitting ${jobid2}"

# jobid3=$(qsub -terse -hold_jid $jobid2 ./run_auccpvloss_gpu.sh -d predict label evaluate --test-checkpoint best "$@")
# echo "submitting ${jobid3}"
