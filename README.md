# CCE_NLI



```pip install -U spacy```

```python -m spacy download en_core_web_sm```

```pip install -r requirements.txt```

mkdir -p BERT
tar -xvf BOWMAN.tar.gz -C BOWMAN/

To Run Explanations:

```
pip install pyparsing==2.4.2
python3 code.pruning_explanation.py --model_type [bowman or bert] --filename [folder name in which to store masks/expls/weights]
```

Lottery Ticket Pruning

```
python3 code/lotteryTicket/snli_lottery_ticket_training.py --model_type [bert or bowman] --ckpt [path to bert initial weights] --finetune_epochs=[epochs to fine-tune for] --prune_metrics_dir [directory to store pruning checkpts]


```

Wanda Pruning

```
python3 code/wanda/snli_wanda_training.py --model_type <bert/bowman> --ckpt <initially trained weights file path> --prune_metrics_dir <dir to store ckpts> --offset <int for which iteration to resume pruning from>
```
