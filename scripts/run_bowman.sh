#!/bin/bash
python -c "import os; print(os.path.abspath('analyze.py'))"

for i in {5..5}; do
    if [[ $i -eq 3 ]]; then
        python3 code/lotteryTicket/snli_lottery_ticket_training.py \
            --model_type bowman \
            --filename "Run0.25_$i" \
            --finetune_epochs 3 \
            --start_idx 0 \
            --pretrained_ckpt BOWMAN/models/pretrained/bowman_MAIN_pretrained_inits.pth
    else
        python3 code/lotteryTicket/snli_lottery_ticket_training.py \
            --model_type bowman \
            --filename "Run0.25_$i" \
            --finetune_epochs 3 \
            --restart_from_ckpt "BOWMAN/models/lottery_ticket/Run0.25_3/0_Pruning_Iter/model_best.pth" \
            --start_idx 1 \
            --pretrained_ckpt BOWMAN/models/pretrained/bowman_MAIN_pretrained_inits.pth
    fi

    # Optional: WANDA training
    

    python3 code/pruning_explanation.py \
        --model_type bowman \
        --pruning_method lottery_ticket \
        --ckpt "BOWMAN/models/lottery_ticket/Run0.25_5/0_Pruning_Iter/model_best.pth" \
        --filename "Run0.25_5"
done
