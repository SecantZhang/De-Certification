set -xe

# Define tempstamp function for log output. 
timestamp() {
    date +"%Y-%m-%d %H:%M:%S"
}

# Define log function for structuring log output. 
log() {
    echo "[$(timestamp)] $1"
}

div() {
    if [ $1 -eq 0 ]
    then
        echo "============================================================================="
    elif [ $1 -eq 1 ]
    then
        echo "-----------------------------------------------------------------------------"
    else
        echo "Invalid sep param."
        exit 125
    fi
}

CUDA_VISIBLE_DEVICES=$1
PHASE=$2

WORK_DIR="/home/zxz147/projects/De-Certification"
OUTPUT_DIR="/data/zxz147/de-certification/rand-smoothing/trained-models/cifar10-rand"
PREDICTION_DIR=$OUTPUT_DIR/prediction_output
CERTIFICATION_DIR=$OUTPUT_DIR/certification_output

mkdir -p $OUTPUT_DIR $PREDICTION_DIR $CERTIFICATION_DIR

DATASET="cifar10"
MODEL_ARCH="cifar_resnet110"
noise_sd=0.5

cd $WORK_DIR

# Training of the smoothed classifier with gaussian perturbations. 
phase_01() {
    batch=400

    CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python codes/smoothing/train.py $DATASET $MODEL_ARCH $OUTPUT_DIR/$DATASET-$MODEL_ARCH-b$batch-n$noise_sd \
        --gpu $CUDA_VISIBLE_DEVICES \
        --batch $batch \
        --noise $noise_sd
}

phase_02() {
    checkpoint_dir="/home/zxz147/git_clones/smoothing/models/cifar10/resnet110/noise_0.50/checkpoint.pth.tar"
    CUDA_VISIBLE_DEVICE=$CUDA_VISIBLE_DEVICES python codes/smoothing/predict.py $DATASET $checkpoint_dir 0.50 $PREDICTION_DIR/$DATASET-$MODEL_ARCH-n$noise_sd-pred-out.tsv \
        --alpha 0.001 \
        --N 1000 \
        --skip 100 \
        --batch 400
}

phase_03() {
    checkpoint_dir="/home/zxz147/git_clones/smoothing/models/cifar10/resnet110/noise_0.50/checkpoint.pth.tar"
    CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python codes/smoothing/certify.py $DATASET $checkpoint_dir 0.50 $CERTIFICATION_DIR/$DATASET-$MODEL_ARCH-n$noise_sd-cert-out-test.tsv \
        --alpha 0.001 \
        --N0 100 \
        --N 100000 \
        --skip 100 \
        --batch 400
}

if [ $PHASE -eq 1 ]
then
    phase_01
elif [ $PHASE -eq 2 ]
then 
    phase_02
elif [ $PHASE -eq 3 ]
then 
    phase_03
else
    echo "Invalid phase param specified. Program exited."
    exit 125
fi