#!/bin/bash

GPUS_PER_NODE=2
MASTER_ADDR=192.168.117.116
MASTER_PORT=6066
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))
DATA_PATH=./   

DETS=H1
CHECKPOINT_PATH=demo

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"

python -m torch.distributed.launch $DISTRIBUTED_ARGS \
       pretrain_gw.py \
       --tensor-model-parallel-size 1 \
       --pipeline-model-parallel-size 1 \
       --num-layers 16 \
       --hidden-size 1024 \
       --num-attention-heads 16 \
       --micro-batch-size 8 \
       --segment-length 256 \
       --dets $DETS \
       --seq-length 128 \
       --max-position-embeddings 128 \
       --train-iters 30000 \
       --save $CHECKPOINT_PATH \
       --load $CHECKPOINT_PATH \
       --data-path $DATA_PATH \
       --data-impl mmap \
       --split 949,50,1 \
       --distributed-backend nccl \
       --lr 0.0001 \
       --lr-decay-style linear \
       --min-lr 1.0e-5 \
       --lr-decay-iters 9900 \
       --weight-decay 1e-2 \
       --clip-grad 1.0 \
       --lr-warmup-fraction .002 \
       --log-interval 1 \
       --save-interval 10000 \
       --eval-interval 1 \
       --dataloader-type cyclic \
       --fp16 \
       --no-binary-head
