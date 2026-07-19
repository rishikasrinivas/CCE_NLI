#!/bin/bash
python -c "import os; print(os.path.abspath('analyze.py'))"


for i in {5..7}; do
    
    python3 code/pruning_explanation.py \
        --model_type bert \
        --pruning_method lottery_ticket \
        --ckpt "BERT/models/lottery_ticket/Run1/0_Pruning_Iter/model_best.pth" \
        --filename "Run1"
done
