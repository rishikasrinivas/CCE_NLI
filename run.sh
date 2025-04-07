#!/bin/bash
python -c "import os; print(os.path.abspath('analyze.py'))"
python3 code/analyze.py --model_type bert --filename Run3


python3 code/lotteryTicket/snli_lottery_ticket_training.py --prune_metrics_dir BERT/models/lottery_ticket/Run3 --model_type bert --ckpt BERT/models/Random/Run3/bert_random_inits.pth

python3 code/activation.py --model_type bert --save_activs_dir BERT/activations/lottery_ticket/Run3 --prune_metrics_dir BERT/models/lottery_ticket/Run3

python3 code/pruning_explanation.py --model_type bert --pruning_method lottery_ticket --ckpt BERT/models/Run3/lottery_ticket/0_Pruning_Iter/model_best.pth --filename Run3


python3 code/wanda/snli_wanda_training.py --prune_metrics_dir BERT/models/wanda/Run3 --model_type bert --ckpt BERT/models/Run3/lottery_ticket/0_Pruning_Iter/model_best.pth

python3 code/activation.py --model_type bert --save_activs_dir BERT/activations/wanda/Run3 --prune_metrics_dir BERT/models/wanda/Run3/0_Pruning_Iter/model_best.pth

python3 code/pruning_explanation.py --model_type bert --pruning_method wanda --ckpt BERT/models/Run3/wanda/0_Pruning_Iter/model_best.pth --filename Run3
 


