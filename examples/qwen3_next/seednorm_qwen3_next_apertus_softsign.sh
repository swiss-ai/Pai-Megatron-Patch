#!/bin/bash
#SBATCH --account=infra01
#SBATCH --job-name=qwen3-next
#SBATCH --time=11:59:00
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --mem=460000
#SBATCH --output=/iopsstor/scratch/cscs/%u/Pai-Megatron-Patch/logs/slurm/training/SeedNorm_softsign-%x-%j.out
#SBATCH --error=/iopsstor/scratch/cscs/%u/Pai-Megatron-Patch/logs/slurm/training/SeedNorm_softsign-%x-%j.err
#SBATCH --no-requeue
#SBATCH --environment=/iopsstor/scratch/cscs/jsun/Pai-Megatron-Patch/examples/qwen3_next/pytorch_env.toml

# ------------------------------------------------------------------
# 0.  Environment & bookkeeping
# ------------------------------------------------------------------
set -e
echo "START TIME: $(date)"

MEGATRON_PATCH_PATH=/iopsstor/scratch/cscs/$USER/Pai-Megatron-Patch
export PYTHONPATH=${MEGATRON_PATCH_PATH}:${MEGATRON_PATCH_PATH}/backends/megatron/Megatron-LM:$PYTHONPATH
export CUDA_DEVICE_MAX_CONNECTIONS=1
export HF_TOKEN= # ADD HF_TOKEN 
export HUGGING_FACE_HUB_TOKEN= # ADD HF_TOKEN 

# ------------------------------------------------------------------
# 1.  SLURM-driven distributed variables
# ------------------------------------------------------------------
NUM_NODES=$SLURM_NNODES
NODE_RANK=$SLURM_NODEID
GPUS_PER_NODE=$SLURM_GPUS_ON_NODE
MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n1)
MASTER_PORT=25679

# ------------------------------------------------------------------
# 2.  Your original parameters (unchanged)
# ------------------------------------------------------------------
TP=1
PP=1
EP=4
ETP=1
CP=1
MBS=1
GBS=256
SEQ_LEN=4096
SFT=false
TRAIN_TOKENS=8388608000
WARMUP_TOKENS=524288000
TENSORBOARD_DIR=/mnt/data/tensorboard/test_qwen3_next_pretrain
TRAIN_ITERS=$(( TRAIN_TOKENS / GBS / SEQ_LEN ))
LR_WARMUP_ITERS=$(( WARMUP_TOKENS / GBS / SEQ_LEN ))
LR_DECAY_ITERS=$(( TRAIN_TOKENS / GBS / SEQ_LEN ))

WANDB_ENTITY= # Add wandb name
WANDB_PROJECT=qwen3next
WANDB_EXP_NAME=qwen3_next_benchmark_sss_gating_xssslur2_seednorm_softsign

DATASETS="/capstor/store/cscs/swissai/a06/datasets_tokenized/megatron/sai/swissai-fineweb-filterrobots-merge/"
DATA_PATH=$(python3 ${MEGATRON_PATCH_PATH}/backends/megatron/Megatron-LM/scripts/tools/create_data_config.py -p ${DATASETS})

# ------------------------------------------------------------------
# 3.  Argument arrays identical to your benchmark_qwen3.sh
# ------------------------------------------------------------------
DISTRIBUTED_ARGS=(
    --nnodes $NUM_NODES
    --node_rank $NODE_RANK
    --nproc_per_node $GPUS_PER_NODE
    --master_addr $MASTER_ADDR
    --master_port $MASTER_PORT
)

MODEL_ARGS=(
    --transformer-impl transformer_engine
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --num-layers 8
    --hidden-size 2048
    --ffn-hidden-size 7680
    --num-attention-heads 16
    --group-query-attention
    --num-query-groups 2
    --hybrid-attention-ratio 0.125
    --hybrid-mlp-ratio 0.5
    --hybrid-override-pattern M-M-M-*-
    --is-hybrid-model
    --normalization SeeDNorm
    --qk-layernorm
    --norm-epsilon 1e-6
    --disable-bias-linear
    --use-rotary-position-embeddings
    --rotary-base 10000000
    --rotary-percent 0.25
    --seq-length ${SEQ_LEN}
    --max-position-embeddings ${SEQ_LEN}
    --position-embedding-type rope
    --untie-embeddings-and-output-weights
    --moe-router-load-balancing-type aux_loss
    --moe-grouped-gemm
    --moe-permute-fusion
    --moe-router-dtype fp32
    --moe-aux-loss-coeff 0.001
    --moe-router-score-function softmax
    --moe-router-topk 10
    --moe-ffn-hidden-size 768
    --moe-shared-expert-intermediate-size 768
    --num-experts 64
    --kv-channels 256
    --apply-layernorm-1p
    --xssslur2
    --sss-gating
    --no-persist-layer-norm
    --seednorm-activation softsign 
)

TRAINING_ARGS=(
    --use-mcore-models
    --micro-batch-size ${MBS}
    --global-batch-size ${GBS}
    --train-iters ${TRAIN_ITERS}
    --weight-decay 0.01
    --adam-beta1 0.9
    --adam-beta2 0.95
    --init-method-std 0.006
    --clip-grad 1.0
    --bf16
    --lr 1.0e-4
    --lr-decay-style cosine
    --min-lr 1.0e-5
    --lr-decay-iters ${LR_DECAY_ITERS}
    --lr-warmup-iters ${LR_WARMUP_ITERS}
    --data-path ${DATA_PATH}
    --dataset MMAP
    --num-workers 32
    --distributed-timeout-minutes 60
    --exit-duration-in-mins 220
    --no-save-optim
    --manual-gc
    --manual-gc-interval 10
    --no-load-optim
    --no-load-rng
    --auto-detect-ckpt-format
    --tensorboard-dir ${TENSORBOARD_DIR}
    --log-timers-to-tensorboard
    --log-memory-to-tensorboard
    --log-validation-ppl-to-tensorboard
    --log-throughput
    --log-interval 1
    --wandb-entity ${WANDB_ENTITY}
    --wandb-project ${WANDB_PROJECT}
    --wandb-exp-name ${WANDB_EXP_NAME}
    --wandb-save-dir $MEGATRON_PATCH_PATH/wandb
    --data-cache-path /iopsstor/scratch/cscs/$USER/datasets/cache
    --patch-tokenizer-type Qwen3Tokenizer
    --load Qwen/Qwen3-0.6B
    --extra-vocab-size 293
    --clip-grad 0.1  # ademamix
    --adam-beta2 0.999  # ademamix
    --ademamix-alpha 8  # ademamix
    --ademamix-beta3 0.9999  # ademamix
    --ademamix-beta3-warmup 100000  # ademamix
    --ademamix-alpha-warmup 100000  # ademamix
    --optimizer ademamix  # ademamix
)

[ ${SFT} = true ] && TRAINING_ARGS+=(
    --eod-mask-loss
    --calculate-per-token-loss
    --train-mode finetune
)

INFRA_ARGS=(
    --enable-experimental
    --tensor-model-parallel-size ${TP}
    --pipeline-model-parallel-size ${PP}
    --expert-model-parallel-size ${EP}
    --context-parallel-size ${CP}
    --expert-tensor-parallel-size ${ETP}
    --use-distributed-optimizer
    --sequence-parallel
    --attention-backend auto
    --cross-entropy-loss-fusion
    --cross-entropy-fusion-impl te
    --moe-token-dispatcher-type alltoall
)

DATA_ARGS=(
    --split 100,0,0
    --seq-length ${SEQ_LEN}
    --reset-attention-mask
    --eod-mask-loss
    --num-workers 32
    --goldfish-loss
    --goldfish-k 50
    --goldfish-h 50
    --eval-interval 5000000
    --save-interval 5000000
)

# ------------------------------------------------------------------
# 4.  Launch
# ------------------------------------------------------------------
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n1)
export MASTER_PORT=25680
export WORLD_SIZE=$SLURM_NTASKS

# --- let SLURM start the processes -------------------------------
TRAINING_CMD="python3 -u /iopsstor/scratch/cscs/jsun/Pai-Megatron-Patch/examples/qwen3_next/pretrain_qwen3_next.py \
    ${MODEL_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${INFRA_ARGS[@]} \
    ${DATA_ARGS[@]}"

srun --cpu-bind=none --mem-bind=none \
     bash -c "RANK=\$SLURM_PROCID LOCAL_RANK=\$SLURM_LOCALID $TRAINING_CMD"
