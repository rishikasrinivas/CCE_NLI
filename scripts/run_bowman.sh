#!/bin/bash
python -c "import os; print(os.path.abspath('analyze.py'))"


for i in {2..4}; do
    python3 code/lotteryTicket/snli_lottery_ticket_training.py \
        --model_type bowman \
        --ckpt BOWMAN/models/lottery_ticket/Run0.25/0_Pruning_Iter/model_best.pth \
        --filename "Run0.25_$i" \
        --finetune_epochs 5 \
        --restart_from_ckpt BOWMAN/models/lottery_ticket/Run0.25/0_Pruning_Iter/model_best.pth \
        --start_idx 1
        

    # python3 code/wanda/snli_wanda_training.py \
    #     --model_type llama \
    #     --filename "Run0.25_$i" \
    #     --prune_method wanda \
    #     --ckpt "LLAMA/models/lottery_ticket/Run0.25_$i/0_Pruning_Iter/model_best.pth"

    python3 code/pruning_explanation.py \
        --model_type bowman \
        --pruning_method lottery_ticket \
        --ckpt "BOWMAN/models/Run0.25/lottery_ticket/0_Pruning_Iter/model_best.pth" \
        --filename "Run0.25_$i"

    # python3 code/wanda/snli_wanda_training.py \
    #     --model_type llama \
    #     --filename "Run0.25_$i" \
    #     --prune_method wanda \
    #     --ckpt "LLAMA/models/lottery_ticket/Run0.25_$i/0_Pruning_Iter/model_best.pth"

done
