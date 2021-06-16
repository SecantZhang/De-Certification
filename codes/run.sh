set -ex

CUDA_DEVICES=$1

ymd=`date +"%y%m%d"`
WORK_PATH="/home/zxz147/projects/De-Certification"
LOG_PATH="/home/zxz147/projects/De-Certification/experiments/logs/$(date +"%y-%m-%d")-exps"

mkdir -p $LOG_PATH
cd ${WORK_PATH}

init_class=0
target_class=1
smoothed="True"
attack_method="swim" # "orig" "orig_cert" "cert" "swim"
smooth_N=100
smooth_N0=10
failure_prob=0.01
targeted="True"
noise_level="0.12" # 0.12, 0.50
report_acc="False" # "True" "False" "True_Cert"

CUDA_VISIBLE_DEVICES=${CUDA_DEVICES} python -u codes/attack_boundary.py \
    -ic ${init_class} \
    -tc ${target_class} \
    -s ${smoothed} \
    -a ${attack_method} \
    -sn ${smooth_N} \
    -sn0 ${smooth_N0} \
    -f ${failure_prob} \
    -t ${targeted} \
    -n ${noise_level} \
    -r ${report_acc} 2>&1 | tee ${LOG_PATH}/${ymd}-s${smooth_N}-a${attack_method}-sn${smooth_N}-sn0${smooth_N0}-f${failure_prob}-t${targeted}-n${noise_level}-r${report_acc}-1.log