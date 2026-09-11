#!/usr/bin/env bash
# ./scripts/exp/train_v1011c.sh
set -e
cd /home/jaydv/code/prox_learning/submodules/act
export PYTHONPATH=$PWD

WANDB_PROJECT="${WANDB_PROJECT:-PC_ACT_experiments}"
EXP="${EXP:-PACT_READOUT}" # ACT, PACT_RAW, PACT_READOUT
SEED="${SEED:-1}"
TASK="${TASK:-pact_place_corridor_v10_11c_100}"
CHUNK_SIZE="${CHUNK_SIZE:-50}"
BATCH_SIZE="${BATCH_SIZE:-8}"
LR="${LR:-1e-5}"
EPOCHS="${EPOCHS:-2000}"

# Brace every var. Bare $TASK_ is an empty name, not "$TASK" + "_".
RUN="${TASK}_${EXP}_s${SEED}_bs${BATCH_SIZE}_cs${CHUNK_SIZE}_lr${LR}_e${EPOCHS}"

echo "$RUN"

if [ "$EXP" = ACT ]; then
  extra=
elif [ "$EXP" = PACT_RAW ]; then
  extra="--use_proximity --prox_feature raw --prox_layout per_sensor --prox_pool min"
elif [ "$EXP" = PACT_READOUT ]; then
  extra="--use_proximity --prox_feature surface_embedding --prox_layout per_sensor --prox_pool min --prox_tokens_per_sensor 1 --prox_encoder_ckpt /home/jaydv/code/prox_learning/experiments_output/default/surface_encoder_train/pact_place_corridor_v5/pact_surface_embedding_encoder_v1.pt --finetune_prox_encoder --prox_policy_tap readout"
else
  echo "set EXP=ACT, PACT_RAW, or PACT_READOUT"; exit 1
fi

python imitate_episodes.py \
  --task_name "$TASK" --policy_class ACT --ckpt_dir ckpts \
  --run_dir "ckpts/$TASK/$RUN" \
  --kl_weight 10 --chunk_size "$CHUNK_SIZE" --hidden_dim 512 --dim_feedforward 3200 \
  --batch_size "$BATCH_SIZE" --lr "$LR" --seed "$SEED" --num_epochs "$EPOCHS" \
  --wandb_project "$WANDB_PROJECT" --wandb_run_name "$RUN" \
  $extra \
  "$@"
