set -ex

CUDA_DEVICES=$1

ymd=`date +"%y%m%d"`
WORK_PATH="/home/zxz147/projects/De-Certification"
LOG_PATH="/home/zxz147/projects/De-Certification/experiments/logs/$(date +"%y-%m-%d")-exps"

mkdir -p $LOG_PATH
cd ${WORK_PATH}

smoothed="True"
attack_method="cert"
smooth_N=100
smooth_N0=10
failure_prob=0.01
targeted="False"
noise_level="0.12" # 0.12, 0.50
report_acc="False"

CUDA_VISIBLE_DEVICES=${CUDA_DEVICES} python codes/attack_boundary.py \
    -s ${smoothed} \
    -a ${attack_method} \
    -sn ${smooth_N} \
    -sn0 ${smooth_N0} \
    -f ${failure_prob} \
    -t ${targeted} \
    -n ${noise_level} \
    -r ${report_acc} 2>&1 | tee ${LOG_PATH}/${ymd}-s${smooth_N}-a${attack_method}-sn${smooth_N}-sn0${smooth_N0}-f${failure_prob}-t${targeted}-n${noise_level}-r${report_acc}-6.log