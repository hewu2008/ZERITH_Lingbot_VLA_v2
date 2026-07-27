#!/bin/bash

set -x

export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1 
export HF_DATASETS_OFFLINE=1 
export TRANSFORMERS_OFFLINE=1 
export HF_HUB_DISABLE_TELEMETRY=1 
export DISABLE_TELEMETRY=1 

if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
  NPROC_PER_NODE=$(nvidia-smi -L | wc -l)
else
  NPROC_PER_NODE=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
fi
echo "Using NPROC_PER_NODE=$NPROC_PER_NODE GPUs"
NNODES=${NNODES:=1}
NPROC_PER_NODE=${NPROC_PER_NODE:=$NPROC_PER_NODE}
NODE_RANK=${NODE_RANK:=0}
MASTER_ADDR=${MASTER_ADDR:=0.0.0.0}
MASTER_PORT=${MASTER_PORT:=62500}


python -m torch.distributed.run \
  --nnodes=$NNODES \
  --nproc-per-node $NPROC_PER_NODE \
  --node-rank $NODE_RANK \
  --master-addr=$MASTER_ADDR \
  --master-port=$MASTER_PORT \
  tasks/vla/train_lingbotvla.py ./configs/vla/zerith/zerith.yaml \
  --train.enable_gradient_checkpointing=true \
  --train.enable_fp32=true \
  --train.use_lora=true \
  --train.lora_rank=8 \
  --train.lora_alpha=8 \
  --train.freeze_non_lora=true \
  --train.lr=1e-4 \
  --train.lr_min=5e-6 \
  --train.micro_batch_size=1 \
  --train.global_batch_size=8 \
  --train.gradient_accumulation_steps=8 \
  --train.max_steps=40000 \
  --train.save_steps=4000 \
  --train.lora_grad_norm_log_interval=1 \
  --train.enable_resume=true \
  --train.output_dir output/zerith_lora \
  2>&1 | tee output/log_zerith_lora.txt