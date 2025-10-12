#!/bin/bash
python -c "import os; print(os.path.abspath('analyze.py'))"

python3 code/lotteryTicket/snli_lottery_ticket_training.py  --model_type bowman --filename Run0.25_newtraining --finetune_epochs 10 
python3 code/pruning_explanation.py --model_type bowman --pruning_method lottery_ticket --ckpt BOWMAN/models/Run0.25_newtraining /lottery_ticket/0_Pruning_Iter/model_best.pth --filename Run0.25_newtraining 


python3 code/wanda/snli_wanda_training.py --model_type bowman --filename Run0.25_newtraining --prune_method wanda --ckpt BOWMAN/models/lottery_ticket/Run0.25_newtraining /0_Pruning_Iter/model_best.pth

python3 code/pruning_explanation.py --model_type bowman --filename Run0.25  --pruning_method wanda --ckpt BOWMAN/models/Run0.25/lottery_ticket/0_Pruning_Iter/model_best.pth