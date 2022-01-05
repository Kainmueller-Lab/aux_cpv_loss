#!/bin/zsh

jobid1=$(qsub -terse ./run_auccpvloss_gpu.sh --term_after_predict -d validate_checkpoints "$@")
echo "submitting ${jobid1}"

jobid2=$(qsub -terse -hold_jid $jobid1 ./run_auccpvloss_cpu.sh -d validate_checkpoints "$@")
echo "submitting ${jobid2}"

jobid3=$(qsub  -terse -hold_jid $jobid2 ./run_auccpvloss_gpu.sh -d predict label evaluate --test-checkpoint best "$@")
echo "submitting ${jobid3}"


# jobid3=$(qsub ./run_auccpvloss_gpu.sh -d predict label evaluate --test-checkpoint best "$@")
# echo "submitting ${jobid3}"
