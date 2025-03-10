#!/bin/bash

# Command 1
python -c "print('Hello from command 1')"

# Command 2
python3 code/activation.py --model_type bowman --save_activs_dir activations/bowman/lottery_ticket/Run2FIXEDLTH --prune_metrics_dir models/snli/prune_metrics/lottery_ticket/bowman/Run2FIXEDLTH 

python3 code/activation.py --model_type bowman --save_activs_dir activations/bowman/wanda/Run2FIXEDLTH --prune_metrics_dir models/snli/prune_metrics/wanda/bowman/Run2FIXEDLTH

python3 code/activation.py --model_type bert --save_activs_dir activations/bert/lottery_ticket/Run2FIXEDLTH --prune_metrics_dir models/snli/prune_metrics/lottery_ticket/bert/Run2FIXEDLTH

python3 code/activation.py --model_type bert --save_activs_dir activations/bert/wanda/Run2FIXEDLTH --prune_metrics_dir models/snli/prune_metrics/wanda/bert/Run2FIXEDLTH

python3 code/activation.py --model_type bert --save_activs_dir activations/bert/random --prune_metrics_dir models/snli/Run2Random
python3 code/activation.py --model_type bowman --save_activs_dir activations/bowman/random --prune_metrics_dir models/snli/Run2Random


python3 code/activation.py --model_type bert --save_activs_dir activations/bert/osccar/Run2FIXEDLTH --prune_metrics_dir models/snli/prune_metrics/osscar