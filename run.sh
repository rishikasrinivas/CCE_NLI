#!/bin/bash

python3 code/lotteryTicket/snli_lottery_ticket_training.py --prune_metrics_dir models/snli/prune_metrics/lottery_ticket/bowman/Run2FIXEDLTH --model_type bowman 

python3 code/activation.py --model_type bowman --save_activs_dir activations/bowman/lottery_ticket/Run2FIXEDLTH --prune_metrics_dir models/snli/prune_metrics/lottery_ticket/bowman/Run2FIXEDLTH

 python3 code/pruning_explanation.py --expls_mask_root_dir exp/bowman/lottery_ticket/Run2FIXEDLTH --prune_metrics_dir models/snli/prune_metrics/lottery_ticket/bowman/Run2FIXEDLTH --model_type bowman --pruning_method lottery_ticket --ckpt models/snli/prune_metrics/lottery_ticket/bowman/Run2FIXEDLTH/0_Pruning_Iter/model_best.pth --activations_root_dir activations/bowman/lottery_ticket/Run2FIXEDLTH


python3 code/wanda/snli_wanda_training.py --prune_metrics_dir models/snli/prune_metrics/wanda/bowman/Run2FIXEDLTH --model_type bowman --ckpt models/snli/Run2_bowman_random_inits.pth

python3 code/activation.py --model_type bowman --save_activs_dir activations/bowman/wanda/Run2FIXEDLTH --prune_metrics_dir models/snli/prune_metrics/wanda/bowman/Run2FIXEDLTH

python3 code/pruning_explanation.py --expls_mask_root_dir exp/bowman/wanda/Run2FIXEDLTH --prune_metrics_dir models/snli/prune_metrics/wanda/bowman/Run2FIXEDLTH --model_type bowman --pruning_method wanda --ckpt models/snli/prune_metrics/wanda/bowman/Run2FIXEDLTH/0_Pruning_Iter/model_best.pth --activations_root_dir activations/bowman/wanda/Run2FIXEDLTH
 
python3 code/pruning_explanation.py --expls_mask_root_dir exp/bert/osscar/Run2FIXEDLTH --prune_metrics_dir models/snli/prune_metrics/osscar/bert/Run2FIXEDLTH --model_type bert --pruning_method osscar --ckpt models/snli/prune_metrics/osscar/bert/Run2FIXEDLTH/0_Pruning_Iter/model_best.pth --activations_root_dir activations/bert/osscar/Run2FIXEDLTH
 

