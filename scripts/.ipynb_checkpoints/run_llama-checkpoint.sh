#!/bin/bash

# Optional: Print the absolute path of analyze.py
python -c "import os; print(os.path.abspath('analyze.py'))"

for i in {4..6}; do
    echo "Processing iteration $i"
    
    if [[ $i -eq 4 ]]; then
        echo "Running lottery ticket training for iteration $i (first run - no restart)"
        python3 code/lotteryTicket/snli_lottery_ticket_training.py \
            --model_type llama \
            --filename "Run0.25_$i" \
            --finetune_epochs 3 \
            --start_idx 0 \
            --pretrained_ckpt LLAMA/models/pretrained/llama_MAIN_pretrained_inits.pth \

 
    else
        echo "Running lottery ticket training for iteration $i (restart from iteration 4)"
        python3 code/lotteryTicket/snli_lottery_ticket_training.py \
            --model_type llama \
            --filename "Run0.25_$i" \
            --finetune_epochs 3 \
            --restart_from_ckpt "LLAMA/models/lottery_ticket/Run0.25_4/0_Pruning_Iter/model_best.pth" \
            --start_idx 1 \
            --pretrained_ckpt LLAMA/models/pretrained/llama_MAIN_pretrained_inits.pth
    fi
    
    # Optional: WANDA training
    # echo "Running WANDA training for iteration $i"
    # python3 code/wanda/snli_wanda_training.py \
    #     --model_type llama \
    #     --filename "Run0.25_3" \
    #     --prune_method wanda \
    #     --ckpt "LLAMA/models/lottery_ticket/Run0.25_$i/0_Pruning_Iter/model_best.pth"
    
    #echo "Running pruning explanation for iteration $i"
    #python3 code/pruning_explanation.py \
    #    --model_type llama \
    #    --pruning_method lottery_ticket \
    #    --ckpt "LLAMA/models/lottery_ticket/Run0.25_$i/0_Pruning_Iter/model_best.pth" \
    #    --filename "Run0.25_$i"
    
    
done

echo "All iterations complete!"
