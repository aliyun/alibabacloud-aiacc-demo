#!/bin/bash
#export MXNET_CPU_WORKER_NTHREADS=8
export MXNET_CUDNN_AUTOTUNE_DEFAULT=0
export MXNET_ENGINE_TYPE=ThreadedEnginePerDevice

# Disable the openmp tuning, as perseus is using one process
# per GPU, the openmp tuning function assume each process own
# all the CPU resources which cause very long time tuning and
# is harmful for performance actually.
export MXNET_USE_OPERATOR_TUNING=0
export MXNET_USE_NUM_CORES_OPERATOR_TUNING=1
export OMP_NUM_THREADS=1

export MXNET_GPU_WORKER_NSTREAMS=1

# Force perseus to use hybrid allreduce
export PERSEUS_ALLREDUCE_MODE=0
# Set maximum perseus stream count to 6
export PERSEUS_ALLREDUCE_DTYPE=1
export PERSEUS_ALLREDUCE_NANCHECK=0

#export NCCL_DEBUG=INFO
#export NCCL_P2P_DISABLE=1

# mxnet profiler
#export MXNET_PROFILER_AUTOSTART=1

EPOCH=26
BS=16
NORM_LAYER='syncbn'  # bn syncbn perseussyncbn
#NORM_LAYER='perseussyncbn'
SAVE_PATH='save-parameter/'
DATASET="coco"
#KVSTORE='nccl'  #local device nccl
#KVSTORE='device'
#KVSTORE='dist-device-sync'
KVSTORE='dist-sync-perseus'
#export CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7' 

GPUS='0,1,2,3,4,5,6,7' # only for nccl
#/root/anaconda/envs/mxnet_1.5.1.post0_cu10.0_py36/bin/python -u ./train_faster_rcnn.py \
python -u ./train_faster_rcnn.py \
    --epochs $EPOCH \
    --batch-size $BS \
    --kv-store $KVSTORE \
    --save-prefix $SAVE_PATH \
    --dataset $DATASET \
    --gpus $GPUS \
    --norm-layer $NORM_LAYER \
    custom-model 
