#!/bin/bash
sparsities=(0.25 0.4375 0.57812 0.68359 0.7627)
model_name=llama
encoder_name=knowledgator/Llama-encoder-1.0B
mkdir -p logs

for i in "${!sparsities[@]}"; do
    gpu=$((i % 2))         # alternates: 0, 1, 0, 1, 0
    sparsity=${sparsities[$i]}

    echo "Launching sparsity $sparsity on GPU $gpu"
    CUDA_VISIBLE_DEVICES=$gpu ./code/cofi/run.sh \
        ${model_name} ${encoder_name} ${sparsity} \
        "structured_heads+structured_mlp+hidden+layer+final_mlp_hidden" $((i+1)) \
        > logs/sparsity_${sparsity}.log 2>&1 &
    pids+=($!)

    # every 2 jobs, wait for both to finish before launching next pair
    if (( (i+1) % 2 == 0 )); then
        echo "Waiting for pair to finish..."
        wait "${pids[@]}"
        pids=()
    fi
done

# wait for any leftover (5th job)
wait "${pids[@]}"
echo "All done"