#!/bin/bash
python -c "import os; print(os.path.abspath('analyze.py'))"


for i in {3..4}; do

    python3 code/lotteryTicket/snli_lottery_ticket_training.py \
        --model_type llama \
        --filename "Run0.25_$i" \
        --finetune_epochs 10 \
        --restart_from_ckpt "LLAMA/models/lottery_ticket/Run0.25_2/0_Pruning_Iter/model_best.pth" \
        --start_idx 1 \
        --ckpt LLAMA/models/lottery_ticket/Run0.25_2/0_Pruning_Iter/model_best.pth

    # python3 code/wanda/snli_wanda_training.py \
    #     --model_type llama \
    #     --filename "Run0.25_$i" \
    #     --prune_method wanda \
    #     --ckpt "LLAMA/models/lottery_ticket/Run0.25_$i/0_Pruning_Iter/model_best.pth"

    python3 code/pruning_explanation.py \
        --model_type llama \
        --pruning_method wanda \
        --ckpt "LLAMA/models/wanda/Run0.25_2/0_Pruning_Iter/model_best.pth" \
        --filename "Run0.25_2"


   


done
