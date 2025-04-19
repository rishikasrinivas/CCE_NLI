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
