#!/usr/bin/env bash
export MXNET_CPU_WORKER_NTHREADS=8
export MXNET_CUDNN_AUTOTUNE_DEFAULT=0
export MXNET_ENGINE_TYPE=ThreadedEnginePerDevice

# Disable the openmp tuning, as perseus is using one process
# per GPU, the openmp tuning function assume each process own
# all the CPU resources which cause very long time tuning and
# is harmful for performance actually.
export MXNET_USE_OPERATOR_TUNING=0
export MXNET_USE_NUM_CORES_OPERATOR_TUNING=4
# Force perseus to use hybrid allreduce
export PERSEUS_ALLREDUCE_MODE=1
# Set maximum perseus stream count to 6
export PERSEUS_ALLREDUCE_STREAMS=6

export PERSEUS_ALLREDUCE_DTYPE=2
export PERSEUS_ALLREDUCE_NANCHECK=1

DATA_DIR=/root/faces_ms1m_112x112/

NETWORK=r101
JOB=softmax1e3
MODELDIR="../save-results/model-$NETWORK-$JOB"
mkdir -p "$MODELDIR"
PREFIX="$MODELDIR/model"
LOGFILE="$MODELDIR/log"

KVSTORE='dist-sync-perseus'
#TRAIN_MODE='data-parallel'
TRAIN_MODE='data-and-model-parallel'
TARGET='lfw'
LR=0.02
PER_BATCH_SIZE=32
MARGIN_M=0.0
MARGIN_S=64
EASY_MARGIN=0
EMB_SIZE=512
END_EPOCH=50
NUM_CLASSES=85000
PREFETCH=0

CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7' /root/anaconda/envs/mxnet_1.5.1.post0_cu10.0_py27/bin/python -u ./train.py --data-dir $DATA_DIR --network "$NETWORK" \
    --loss-type 4 \
    --prefix "$PREFIX" \
    --per-batch-size $PER_BATCH_SIZE \
    --kvstore $KVSTORE \
    --train-mode $TRAIN_MODE \
    --target $TARGET \
    --lr $LR \
    --margin-m $MARGIN_M \
    --margin-s $MARGIN_S \
    --easy-margin $EASY_MARGIN \
    --emb-size $EMB_SIZE \
    --end-epoch $END_EPOCH \
    --num-classes $NUM_CLASSES \
    --prefetch $PREFETCH
#    > "$LOGFILE" 2>&1 &

