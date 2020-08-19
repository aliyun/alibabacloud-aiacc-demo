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

DATA_DIR=/ncluster/data/faces_ms1m_112x112/

NETWORK=r100
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
NUM_CLASSES=80000
LOCAL_RUN=0

cmd=CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7' /root/anaconda/envs/mxnet_1.5_cu10.0_py27/bin/python -u ./train.py --data-dir $DATA_DIR --network "$NETWORK" \
    --loss-type 4 \
    --prefix "$PREFIX" \
    --per-batch-size $PER_BATCH_SIZE \
    --kvstore $KVSTORE \
    --train-mode $TRAIN_MODE \
    --target $TARGET \
    --lr $LR \
    --per-batch-size $PER_BATCH_SIZE \
    --margin-m $MARGIN_M \
    --margin-s $MARGIN_S \
    --easy-margin $EASY_MARGIN \
    --emb-size $EMB_SIZE \
    --end-epoch $END_EPOCH \
    --num-classes $NUM_CLASSES \
    --local-run $LOCAL_RUN 
#    > "$LOGFILE" 2>&1 &
cmd2=CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7' /root/anaconda/envs/mxnet_1.5_cu10.0_py27/bin/python -m pdb -u ./train.py --data-dir $DATA_DIR --network "$NETWORK" \
    --loss-type 4 \
    --prefix "$PREFIX" \
    --per-batch-size $PER_BATCH_SIZE \
    --kvstore $KVSTORE \
    --train-mode $TRAIN_MODE \
    --target $TARGET \
    --lr $LR \
    --per-batch-size $PER_BATCH_SIZE \
    --margin-m $MARGIN_M \
    --margin-s $MARGIN_S \
    --easy-margin $EASY_MARGIN \
    --emb-size $EMB_SIZE \
    --end-epoch $END_EPOCH \
    --num-classes $NUM_CLASSES \
    --local-run $LOCAL_RUN 
if [ $OMPI_COMM_WORLD_RANK -eq 0 ] ; then
    cmd2
else
    echo "common..."
    cmd
fi
