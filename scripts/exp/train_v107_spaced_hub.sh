#!/usr/bin/env bash
# ./scripts/exp/train_v107_spaced_hub.sh
# v107_spaced hub: data/v107_spaced in pact_pick_n_place_v2 (200 eps, exo + wrist). Not the 210-ep batman T-107 set (train_v107_spaced.sh). Eval: eval_v107_spaced_hub.sh.
set -e
cd /home/jaydv/code/prox_learning/submodules/act
export PYTHONPATH=$PWD

WANDB_PROJECT=PC_ACT_experiments
EXP="${EXP:-PACT_READOUT}"  # ACT, PACT_RAW, PACT_READOUT
SEED="${SEED:-0}"
TASK=pact_pick_n_place_v2_v107_spaced
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
