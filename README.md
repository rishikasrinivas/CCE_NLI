# CCE_NLI

```
git clone 
cd CCE_NLI
./downloads.sh
```
upload model_best.pth to directory CCE_NLI and SNLI_1.0 to CCE_NLI/data (both uploaded here: https://drive.google.com/drive/folders/1D9onWZBu8aJWnRABIkyIPmeIYHUb50Ky?usp=sharing)


To Run Explanations:

```
pip install pyparsing==2.4.2
python3 code.pruning_explanation.py --model_type bert --filename test_exp
```

Lottery Ticket Pruning

```
 python3 code/lotteryTicket/snli_lottery_ticket_training.py  --model_type bert --filename test_lth 
```

Wanda Pruning

```
python3 code/wanda/snli_wanda_training.py --model_type bert --filename test_wanda  --prune_method wanda --ckpt model_best.pth
```
