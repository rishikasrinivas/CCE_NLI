# CCE_NLI



```pip install -U spacy```

```python -m spacy download en_core_web_sm```

```pip install -r requirements.txt```


To Run Explanations:

```
pip install pyparsing==2.4.2
python3 code/pruning_explanation.py --expls_mask_root_dir <dir to store explanations and masks> --prune_metrics_dir <dir to store ckpts> --model_type <bowman/bert> --pruning_method <lottery_ticket or wanda> --ckpt <ckpt to load model from>
```

Lottery Ticket Pruning

```
python3 code/lotteryTicket/snli_lottery_ticket_training.py --model_type [bert or bowman] --ckpt [path to bert initial weights] --finetune_epochs=[epochs to fine-tune for] --prune_metrics_dir [directory to store pruning checkpts]


```

Wanda Pruning

```
python3 code/wanda/snli_wanda_training.py --model_type <bert/bowman> --ckpt <initially trained weights file path> --prune_metrics_dir <dir to store ckpts> --offset <int for which iteration to resume pruning from>
```
