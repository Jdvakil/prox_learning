#!/usr/bin/env bash
# ./scripts/exp/train_v5_ext.sh
# v5 ext: pact_place_corridor/data/v5 (193 eps, same corridor v2 scene as hallway v5; trained wrist-only). Eval: eval_v5_ext.sh (eval_act.py --task hallway).
set -e
cd /home/jaydv/code/prox_learning/submodules/act
export PYTHONPATH=$PWD

WANDB_PROJECT=PC_ACT_experiments
EXP="${EXP:-PACT_READOUT}"  # ACT, PACT_RAW, PACT_READOUT
SEED="${SEED:-0}"
TASK=pact_place_corridor_v5_ext
CHUNK_SIZE=50
BATCH_SIZE=8
LR=1e-5
EPOCHS=2000
# Brace every var. Bare $TASK_ is an empty name, not "$TASK" + "_".
RUN="${TASK}_${EXP}_s${SEED}_bs${BATCH_SIZE}_cs${CHUNK_SIZE}_lr${LR}_e${EPOCHS}"

echo "$RUN"

RUN_DIR="ckpts/$TASK/$RUN"
if [ -f "$RUN_DIR/policy_best.ckpt" ]; then
  echo "DONE $RUN_DIR has policy_best.ckpt. Change SEED to train another."; exit 0
fi
if [ -d "$RUN_DIR" ] && [ -n "$(ls -A "$RUN_DIR")" ]; then
  echo "REFUSE $RUN_DIR exists without policy_best.ckpt (running or crashed). Move it aside first."; exit 3
fi


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
