#!/bin/bash
python -c "import os; print(os.path.abspath('analyze.py'))"


for i in {4..5}; do
    if [[ $i -eq 4 ]]; then
        python3 code/lotteryTicket/snli_lottery_ticket_training.py \
            --model_type bert \
            --filename "Run0.25_$i" \
            --finetune_epochs 3 \
            --start_idx 0 \
            --pretrained_ckpt BERT/models/pretrained/bert_MAIN_pretrained_inits.pth
    else
        python3 code/lotteryTicket/snli_lottery_ticket_training.py \
            --model_type bert \
            --filename "Run0.25_$i" \
            --finetune_epochs 3 \
            --restart_from_ckpt "BERT/models/lottery_ticket/Run0.25_4/0_Pruning_Iter/model_best.pth" \
            --start_idx 1 \
            --pretrained_ckpt BERT/models/pretrained/bert_MAIN_pretrained_inits.pth
    fi

    # Optional: WANDA training
    # python3 code/wanda/snli_wanda_training.py \
    #     --model_type llama \
    #     --filename "Run0.25_$i" \
    #     --prune_method wanda \
    #     --ckpt "LLAMA/models/lottery_ticket/Run0.25_$i/0_Pruning_Iter/model_best.pth"

    python3 code/pruning_explanation.py \
        --model_type bert \
        --pruning_method CoFi \
        --ckpt "BERT/models/CoFi/Run0.25_6/0_Pruning_Iter/model_best.pth" \
        --filename "Run0.25_6"
done
