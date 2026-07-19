#!/usr/bin/env bash

set -euo pipefail

if (( $# < 1 || $# > 3 )); then
    echo "Usage: $0 <llama|bert> [start_sparsity] [gpu]" >&2
    exit 2
fi

model="$1"
start_sparsity="${2:-0.25}"
gpu="${3:-0}"

if [[ ! "$gpu" =~ ^[0-9]+$ ]]; then
    echo "Invalid GPU: $gpu" >&2
    exit 2
fi
case "$model" in
    llama)
        model_dir="LLAMA"
        model_name_or_path="knowledgator/Sheared-LLaMA-encoder-1.3B"
        reg_learning_rate="0.1"
        prune_modules="head+head_layer+intermediate+mlp+final_mlp_hidden+hidden"
        ;;
    bert)
        model_dir="BERT"
        model_name_or_path="bert-base-uncased"
        reg_learning_rate="0.01"
        prune_modules="head+head_layer+intermediate+mlp+final_mlp_hidden+hidden"
        ;;
    bowman)
        model_dir="BOWMAN"
        model_name_or_path="bowman"
        reg_learning_rate="0.1"
        prune_modules="final_mlp_hidden"
        ;;
    *)
        echo "Unsupported model: $model (expected llama, bert, or bowman)" >&2
        exit 2
        ;;
esac

sparsities=(0.25 0.4375 0.57812 0.68359 0.7627)
seeds=(57 42 68)

start_index=-1

for index in "${!sparsities[@]}"; do
    if [[ "${sparsities[$index]}" == "$start_sparsity" ]]; then
        start_index="$index"
        break
    fi
done

if (( start_index == -1 )); then
    echo "Invalid start sparsity: $start_sparsity" >&2
    echo "Valid values: ${sparsities[*]}" >&2
    exit 2
fi

for run in {2..3}; do
    seed="${seeds[$((run - 1))]}"

    for index in "${!sparsities[@]}"; do
        if (( index < start_index )); then
            continue
        fi

        iteration=$((index + 1))
        sparsity="${sparsities[$index]}"

        output_dir="/workspace/CCE_NLI/${model_dir}/models/CoFi/Run${run}/${iteration}_Pruning_Iter"

       

        echo "Starting Run${run}"
        echo "Iteration: ${iteration}"
        echo "Sparsity: ${sparsity}"
        echo "Seed: ${seed}"
        echo "Physical GPU: ${gpu}"
        echo "Output: ${output_dir}"

        
            
        # fientuning 
        CUDA_VISIBLE_DEVICES="$gpu" python3 code/cofi/run-sheard.py \
            --model_name "$model" \
            --using_untrained_student False \
            --path_to_pretrained "${model_dir}/models/pretrained/" \
            --teacher_model_dir "/workspace/CCE_NLI/${model_dir}/models/lottery_ticket/Run1/" \
            --data_debug 100 \
            --output_dir "$output_dir" \
            --logging_steps 100 \
            --task_name SNLI \
            --model_name_or_path "$model_name_or_path" \
            --ex_name "SNLI_sparsity${sparsity}" \
            --do_train \
            --do_eval \
            --max_seq_length 128 \
            --per_device_train_batch_size 32 \
            --per_device_eval_batch_size 32 \
            --learning_rate 2e-5 \
            --reg_learning_rate "$reg_learning_rate" \
            --num_train_epochs 3 \
            --overwrite_output_dir \
            --save_steps 0 \
            --eval_steps 500 \
            --evaluation_strategy steps \
            --seed "$seed" \
            --pruning_type "$prune_modules" \
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
        
        CUDA_VISIBLE_DEVICES="$gpu" python3 code/cofi/run-sheard.py \
            --model_name "$model" \
            --using_untrained_student False \
            --path_to_pretrained "${model_dir}/models/pretrained/" \
            --teacher_model_dir "/workspace/CCE_NLI/${model_dir}/models/lottery_ticket/Run1/" \
            --data_debug 100 \
            --output_dir "$output_dir" \
            --logging_steps 100 \
            --task_name SNLI \
            --model_name_or_path "$model_name_or_path" \
            --ex_name "SNLI_sparsity${sparsity}" \
            --do_train \
            --do_eval \
            --max_seq_length 128 \
            --per_device_train_batch_size 32 \
            --per_device_eval_batch_size 32 \
            --learning_rate 2e-5 \
            --reg_learning_rate "$reg_learning_rate" \
            --num_train_epochs 3 \
            --overwrite_output_dir \
            --save_steps 0 \
            --eval_steps 500 \
            --evaluation_strategy steps \
            --seed "$seed" \
            --pretrained_pruned_model "${output_dir}/model_best.pth" \
            --target_sparsity "$sparsity" \
            --device cuda

    done
done