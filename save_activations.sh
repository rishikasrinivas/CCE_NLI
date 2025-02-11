#!/bin/bash

# Command 1
python -c "print('Hello from command 1')"

# Command 2
python3 code/activation.py --model_type bowman --save_masks_dir activations/bowman/lottery_ticket/RunFIXEDLTH --prune_metrics_dir models/snli/prune_metrics/lottery_ticket/bowman/RunFIXEDLTH 

python3 code/activation.py --model_type bowman --save_masks_dir activations/bowman/wanda/Run1 --prune_metrics_dir models/snli/prune_metrics/wanda/bowman/Run1

python3 code/activation.py --model_type bert --save_masks_dir activations/bert/lottery_ticket/Run1 --prune_metrics_dir models/snli/prune_metrics/lottery_ticket/bert/Run1

python3 code/activation.py --model_type bert --save_masks_dir activations/bert/wanda/Run1 --prune_metrics_dir models/snli/prune_metrics/wanda/bert/Run1

python3 code/activation.py --model_type bert --save_masks_dir activations/bert/random --prune_metrics_dir models/snli
python3 code/activation.py --model_type bowman --save_masks_dir activations/bowman/random --prune_metrics_dir models/snli

# Command 3
python -c "import pandas as pd; print(pd.__version__)"

# Command 4
python script4.py arg1 arg2
