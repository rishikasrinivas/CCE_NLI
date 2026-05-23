#!/bin/bash
python -c "import os; print(os.path.abspath('analyze.py'))"


for i in {5..7}; do
    
    python3 code/pruning_explanation.py \
        --model_type llama \
        --pruning_method lottery_ticket \
        --ckpt "LLAMA/models/lottery_ticket/Run0.25_${i}/0_Pruning_Iter/model_best.pth" \
        --filename "Run0.25_${i}"
done
