#!/usr/bin/env bash

set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "Usage: $0 <model>" >&2
    exit 2
fi

model="$1"

if [[ "$model" != "llama" ]]; then
    echo "Unsupported model: $model (expected: llama)" >&2
    exit 2
fi

sparsities=(0.25 0.4375 0.57812 0.68359 0.7627)

for run in {1..3}; do
    for index in "${!sparsities[@]}"; do
        iteration=$((index + 1))
        sparsity="${sparsities[$index]}"

        echo "Starting Run${run}, iteration ${iteration}, sparsity ${sparsity}"

        CUDA_VISIBLE_DEVICES=0 python3 code/cofi/run-sheard.py \
            --model_name llama \
            --path_to_pretrained LLAMA/models/pretrained/ \
            --teacher_model_dir "/workspace/CCE_NLI/LLAMA/models/lottery_ticket/Run${run}/" \
            --data_debug 100 \
            --output_dir "/workspace/CCE_NLI/LLAMA/models/CoFi/Run${run}/${iteration}_Pruning_Iter" \
            --logging_steps 100 \
            --task_name SNLI \
            --model_name_or_path knowledgator/Sheared-LLaMA-encoder-1.3B \
            --ex_name "SNLI_sparsity${sparsity}" \
            --do_train \
            --do_eval \
            --max_seq_length 128 \
            --per_device_train_batch_size 32 \
            --per_device_eval_batch_size 32 \
            --learning_rate 2e-5 \
            --reg_learning_rate 1.0 \
            --num_train_epochs 10 \
            --overwrite_output_dir \
            --save_steps 0 \
            --eval_steps 500 \
            --evaluation_strategy steps \
            --seed 57 \
            --pruning_type head+head_layer+intermediate+mlp+final_mlp_hidden+hidden \
            --pretrained_pruned_model None \
            --target_sparsity "$sparsity" \
            --freeze_embeddings \
            --do_distill \
            --do_layer_distill \
            --distill_ce_loss_alpha 0.1 \
            --distill_loss_alpha 0.9 \
            --distill_temp 4 \
            --scheduler_type linear \
            --layer_distill_version 4 \
            --prepruning_finetune_epochs 1 \
            --lagrangian_warmup_epochs 2 \
            --device cuda
    done
done
