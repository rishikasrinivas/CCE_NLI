# CCE_NLI

```
git clone https://github.com/rishikasrinivas/CCE_NLI.git
cd CCE_NLI
mkdir DataLoaders
./downloads.sh
```
upload model_best.pth to directory CCE_NLI and snli_1.0 to CCE_NLI/data (both uploaded here: https://drive.google.com/drive/folders/1D9onWZBu8aJWnRABIkyIPmeIYHUb50Ky?usp=sharing)


To Run Explanations:

```
pip install pyparsing==2.4.2
python3 code/pruning_explanation.py --model_type bert --pruning_method lottery_ticket --ckpt BERT/models/Run3/lottery_ticket/0_Pruning_Iter/model_best.pth --filename Run3
 python3 code/pruning_explanation.py --model_type bert --filename Run3 --pruning_method wanda
 python3 code/pruning_explanation.py --model_type bowman --filename Run1 --pruning_method wanda
```

Lottery Ticket Pruning

```
    mkdir LLAMA
    mkdir models
 python3 code/lotteryTicket/snli_lottery_ticket_training.py  --model_type bert --filename test_lth_bert
 python3 code/lotteryTicket/snli_lottery_ticket_training.py  --model_type llama --filename test_lth_llama 
 python3 code/lotteryTicket/snli_lottery_ticket_training.py  --model_type bowman --filename Run1 
 
```

Wanda Pruning

```
python3 code/wanda/snli_wanda_training.py --model_type bert --filename test_bert  --prune_method wanda --ckpt BERT/models/random/llama_random_inits.pth
python3 code/wanda/snli_wanda_training.py --model_type llama --filename test_llama  --prune_method wanda --ckpt lama_random_inits.pth
python3 code/wanda/snli_wanda_training.py --model_type bowman --filename Run1  --prune_method wanda --ckpt BOWMAN/models/lottery_ticket/Run1/0_Pruning_Iter/model_best.pth
```
