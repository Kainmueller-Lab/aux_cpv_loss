#!/bin/zsh

jobid=$(qsub -terse ./run_auccpvloss_gpu.sh --term_after_predict "$@")
echo "submitting ${jobid}"

hj="-hold_jid ${jobid}"
qsub ${hj} ./run_auccpvloss_cpu.sh "$@"
