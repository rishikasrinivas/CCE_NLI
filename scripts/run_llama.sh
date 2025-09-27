#!/bin/bash
python -c "import os; print(os.path.abspath('analyze.py'))"


python3 code/lotteryTicket/snli_lottery_ticket_training.py  --model_type llama --filename Run0.25_2 --i 2 --finetune_epochs 3
python3 code/wanda/snli_wanda_training.py --model_type llama --filename Run0.25_2 --prune_method wanda --ckpt  LLAMA/models/lottery_ticket/Run0.25_2/0_Pruning_Iter/model_best.pth



python3 code/pruning_explanation.py --model_type llama --pruning_method lottery_ticket --ckpt LLAMA/models/Run0.25_2/lottery_ticket/0_Pruning_Iter/model_best.pth --filename Run0.25_2




python3 code/pruning_explanation.py --model_type llama --filename Run0.25_2 --pruning_method wanda --ckpt LLAMA/models/Run0.25_2/lottery_ticket/0_Pruning_Iter/model_best.pth