# CCE_NLI


To Retrain:

```pip install -U spacy```

```python -m spacy download en_core_web_sm```

```python3 code/snli_train.py```

To Run Explanations:

```
pip install pyparsing==2.4.2
python3 code/analyze.py
```

Lottery Ticket Pruning

```
python3 code/lotteryTicket/snli_lottery_ticket_training.py --model_type [bert or bowman] --ckpt [path to bert initial weights] --finetune_epochs=[epochs to fine-tune for] --prune_metrics_dir [directory to store pruning checkpts]
```
