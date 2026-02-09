#!/bin/bash
python -c "import os; print(os.path.abspath('analyze.py'))"


for i in {6..8}; do
    
    python3 code/pruning_explanation.py \
        --model_type bert \
        --pruning_method CoFi \
        --ckpt "BERT/models/CoFi/Run0.25_${i}/0_Pruning_Iter/model_best.pth" \
        --filename "Run0.25_${i}"
done
