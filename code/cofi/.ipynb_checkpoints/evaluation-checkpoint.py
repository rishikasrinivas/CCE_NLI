import inspect
import os
import pdb
import random
import sys
import time
from torch.nn.utils.rnn import pad_sequence
import torch

import datasets
import numpy as np
import torch
import transformers
from matplotlib import pyplot as plt
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.sampler import SequentialSampler
from transformers import AutoTokenizer, EvalPrediction, GlueDataset
from transformers import GlueDataTrainingArguments as DataTrainingArguments
from transformers.data.data_collator import (DataCollator,
                                             DataCollatorWithPadding,
                                             default_data_collator)
from transformers.trainer_pt_utils import nested_concat, nested_numpify
from transformers.trainer_utils import EvalPrediction
import train_utils
from models.modeling_bert import (CoFiBertForQuestionAnswering,
                                  CoFiBertForSequenceClassification)
from models.modeling_roberta import CoFiRobertaForSequenceClassification
from utils.cofi_utils import *
from utils.qa_utils import *
from utils.utils import *
import data
from data import snli as snli
import evaluate

task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "snli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}
def custom_padding_collator(features):
    # Initialize batch dict
    batch = {
        "pre_input_ids": [],
        "pre_attention_mask": [],
        "hyp_input_ids": [],
        "hyp_attention_mask": [],
        "labels": []
    }

    # Separate and pad each feature
    for feature in features:
        for key in batch.keys():
            if key != "labels":
                batch[key].append(torch.tensor(feature[key]))
            else:
                batch[key].append(feature["label"])

    # Pad sequences
    padded_batch = {}
    for key in batch:
        if key != "labels":
            padded_batch[key] = pad_sequence(
                batch[key],
                batch_first=True,
                padding_value=1  # Use your SNLI's padding index
            )
        else:
            padded_batch[key] = torch.tensor(batch[key])

    return padded_batch
def _remove_unused_columns(dataset: "datasets.Dataset", description):
    # Inspect model forward signature to keep only the arguments it accepts.
    signature = inspect.signature(model.forward)
    signature_columns = list(signature.parameters.keys())
    # Labels may be named label or label_ids, the default data collator handles that.
    signature_columns += ["label", "label_ids"]
    columns = [k for k in signature_columns if k in dataset.column_names]
    ignored_columns = list(set(dataset.column_names) - set(signature_columns))
    dset_description = "" if description is None else f"in the {description} set "
    print(
        f"The following columns {dset_description} don't have a corresponding argument in `{model.__class__.__name__}.forward` and have been ignored: {', '.join(ignored_columns)}."
    )
    dataset.set_format(type=dataset.format["type"], columns=columns)


def get_dataloader(dataset, batch_size):
    dataloader = DataLoader(dataset,
                            sampler=SequentialSampler(dataset),
                            batch_size=batch_size,
                            collate_fn=default_data_collator)
    return dataloader

def post_processing_function(examples, features, predictions):
    # Post-processing: we match the start logits and end logits to answers in the original context.
    predictions = postprocess_qa_predictions(
        examples=examples,
        features=features,
        predictions=predictions,
    )
    # Format the result to the format the metric expects.
    formatted_predictions = [{"id": k, "prediction_text": v}
                             for k, v in predictions.items()]
    references = [{"id": ex["id"], "answers": ex[answer_column_name]}
                  for ex in datasets["validation"]]
    return EvalPrediction(predictions=formatted_predictions, label_ids=references)


def evaluate_(model, dataloader):
    metrics = {}
    total_infer_times = 0
    task_name='snli'
    t = 5
    if task_name in ["rte", "stsb", "cola", "mrpc"]:
        t = 20
    assert t > 1

    total_examples = 0
    for i in range(t):

        preds = None
        label_ids = None
        total_infer_time = 0
        for num_batch, inputs in enumerate(dataloader):
            labels = inputs["labels"] if "labels" in inputs else None
            for key in inputs:
                inputs[key] = inputs[key].cuda()
            with torch.no_grad():
                a = time.time()
                if task_name == "squad":
                    output = model(**inputs)
                    logits = output["start_logits"], output["end_logits"]
                else:
                    logits = model(**inputs)["logits"][2]
                    print(logits)
                torch.cuda.synchronize()
                b = time.time()
                total_infer_time += (b-a)
                if i == 0:
                    total_examples += len(logits)
                    preds = logits if preds is None else nested_concat(
                        preds, logits)
                    label_ids = labels if label_ids is None else nested_concat(
                        label_ids, labels)
        if label_ids is not None:
            final_label_ids = nested_numpify(label_ids)
        if preds is not None:
            final_preds = nested_numpify(preds)

        if i == 0:
            metrics["num_examples"] = total_examples

        if i > 0:
            total_infer_times += total_infer_time
    if task_name == 'squad':
        dataset.set_format(
            type=dataset.format["type"], columns=list(dataset.features.keys()))
        eval_preds = post_processing_function(
            eval_examples, dataset, final_preds)
        metrics = compute_metrics(eval_preds)
    else:
        metrics = compute_metrics(EvalPrediction(
            predictions=final_preds, label_ids=final_label_ids))
    total_infer_time = round(total_infer_times / (t-1), 4)
    metrics["seconds/example"] = total_infer_times / (t-1) / total_examples
    return metrics


def prepare_validation_features(examples):
    # Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. This results
    # in one example possible giving several features when a context is long, each of those features having a
    # context that overlaps a bit the context of the previous feature.
    max_length = 384
    doc_stride = 128
    tokenized_examples = tokenizer(
        examples[question_column_name if pad_on_right else context_column_name],
        examples[context_column_name if pad_on_right else question_column_name],
        truncation="only_second" if pad_on_right else "only_first",
        max_length=max_length,
        stride=doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length"
    )

    # Since one example might give us several features if it has a long context, we need a map from a feature to
    # its corresponding example. This key gives us just that.
    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")

    # For evaluation.py, we will need to convert our predictions to substrings of the context, so we keep the
    # corresponding example_id and we will store the offset mappings.
    tokenized_examples["example_id"] = []

    for i in range(len(tokenized_examples["input_ids"])):
        # Grab the sequence corresponding to that example (to know what is the context and what is the question).
        sequence_ids = tokenized_examples.sequence_ids(i)
        context_index = 1 if pad_on_right else 0

        # One example can give several spans, this is the index of the example containing this span of text.
        sample_index = sample_mapping[i]
        tokenized_examples["example_id"].append(examples["id"][sample_index])

        # Set to None the offset_mapping that are not part of the context so it's easy to determine if a token
        # position is part of the context or not.
        tokenized_examples["offset_mapping"][i] = [
            (o if sequence_ids[k] == context_index else None)
            for k, o in enumerate(tokenized_examples["offset_mapping"][i])
        ]

    return tokenized_examples


def glue_preprocess_function(examples):
    # Tokenize the texts
    sentence1_key, sentence2_key = task_to_keys[task_name]
    max_seq_length = 128
    padding = "max_length"
    args = (
        (examples[sentence1_key],) if sentence2_key is None else (
            examples[sentence1_key], examples[sentence2_key])
    )

    result = tokenizer(*args, padding=padding,
                       max_length=max_seq_length, truncation=True)
    if task_name == "mnli" and model_name_or_path.startswith("princeton-nlp/"):
        # legacy issue of using GLUEDataset
        label_to_id = {1:2, 0:1, 2:0}
        labels = [label_to_id[i] for i in examples["label"]]
        result["label"] = labels
    return result


def warmup():
    time1 = time.time()
    input = torch.randn(128, 1024).cuda()
    linear = torch.nn.Linear(1024, 1024).cuda()
    for i in range(10000):
        input = linear(input)

    time2 = time.time()
    print(round(time2 - time1, 2), "seconds for warmup")

def get_glue_metric():
    metric = load_metric("glue", task_name)
    is_regression = task_name == "stsb"

    # You can define your custom compute_metrics function. It takes an `EvalPrediction` object (a namedtuple with a
    # predictions and label_ids field) and has to return a dictionary string to float.
    def compute_metrics(p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.squeeze(preds) if is_regression else np.argmax(preds, axis=1)
        result = metric.compute(predictions=preds, references=p.label_ids)
        if len(result) > 1:
            result["combined_score"] = np.mean(list(result.values())).item()
        return result
    return compute_metrics

if __name__ == '__main__':
    # warmup
    warmup()

    # data
    model_name='bert'
    train,val, _,_ = train_utils.create_dataloaders(max_data=100, debug=True)
    label_list = list(set(train.labels))
    vocab= {'stoi': train.stoi, 'itos': train.itos}
    
    num_labels = len(set(train.labels))
    bs = 128

    model_name_or_path = 'bert-base-uncased'
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path, use_fast=True, truncation_size="right")
    
    t_name='snli'
    # ======= Load the model params ========
    config = AutoConfig.from_pretrained(
        model_name_or_path,
        num_labels=num_labels,
        finetuning_task='snli',
        cache_dir=None,
        revision='main',
        use_auth_token=False,
    )
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        cache_dir=None,
        use_fast=True,
        revision='main',
        use_auth_token=False,
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token  = tokenizer.eos_token 
        assert model_name == 'llama'
        
    # Padding strategy
    padding = "max_length"

    # Some models have set the order of the labels to use, so let's make sure we do use it.
    label_to_id = None

    LABEL_STOI = {"entailment": 0, "neutral": 1, "contradiction": 2}
    LABEL_ITOS = {v: k for k, v in LABEL_STOI.items()}
    
    label_to_id = LABEL_STOI
    max_seq_length=128
   
    if max_seq_length > tokenizer.model_max_length:
        logger.warning(
            f"The max_seq_length passed ({max_seq_length}) is larger than the maximum length for the"
            f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
        )
    max_seq_length = min(max_seq_length, tokenizer.model_max_length)
    
    
    # tokenize sentences so the dataloader contains tokenized inputs prior to entering training (to align with orginal code)
    
    def preprocess_function(examples):
        result = {}
        def indices_to_bert_tokens(indices):
            batch_size, seq_len = indices.shape
      
            words = []
            for i in range(batch_size):
                sentence = []
                for idx in indices[i]:
                    if idx.item() in vocab['itos']:
                        word = vocab['itos'][idx.item()]
                        if word not in ("[PAD]", "<pad>", "PAD"): 
                            sentence.append(word)
                    else:
                        break
                words.append(sentence)

            return tokenizer(words, is_split_into_words=True, return_tensors="pt", padding=True, truncation=True)
        
        s1_pad, s1len, s2_pad, s2len, labels = snli.pad_collate([
            (torch.tensor(p), plen, torch.tensor(h), hlen, l)
            for p, plen, h, hlen, l in zip(
                examples["premise"],
                examples["premise_len"],
                examples["hypothesis"],
                examples["hypothesis_len"],
                examples["label"]
            )
        ])

        # Transpose as needed
        s1_transposed = s1_pad.transpose(1, 0)
        s2_transposed = s2_pad.transpose(1, 0)

        # Convert to BERT tokens
        s1_tokens = indices_to_bert_tokens(s1_transposed)
        s2_tokens = indices_to_bert_tokens(s2_transposed)

        # Build the result with proper batch dimension
        result = {
            "pre_input_ids": s1_tokens["input_ids"].cpu().numpy(),
            "pre_attention_mask": s1_tokens["attention_mask"].cpu().numpy(),
            "hyp_input_ids": s2_tokens["input_ids"].cpu().numpy(),
            "hyp_attention_mask": s2_tokens["attention_mask"].cpu().numpy(),
            "label": [l for l in labels]
        }
            
        return result

    #load training and val data and tokenize
  
    hf_train = Dataset.from_dict({
        "premise": train.s1s,  # Replace with your actual attributes
        "premise_len": train.s1lens,  # Replace with your actual attributes
        "hypothesis": train.s2s,
        "hypothesis_len": train.s2lens,
        "label": train.labels
    })

    train = hf_train.map(
        preprocess_function,
        batched=True,
        load_from_cache_file=True,
        remove_columns=["premise", "premise_len", "hypothesis", "hypothesis_len"],
        desc="Running tokenizer on dataset",
    ) #! get dataset
    train = train.filter(
        lambda example: example["label"] != -1,
        desc="Filtering out samples with label -1"
    )

    hf_val = Dataset.from_dict({
        "premise": val.s1s,  # Replace with your actual attributes
        "premise_len": val.s1lens,  # Replace with your actual attributes
        "hypothesis": val.s2s,
        "hypothesis_len": val.s2lens,
        "label": val.labels
    })
    val = hf_val.map(
        preprocess_function,
        batched=True,
        load_from_cache_file=True,
        remove_columns=["premise", "premise_len", "hypothesis", "hypothesis_len"],
        desc="Running tokenizer on val dataset",
    ) #! get dataset

    val = val.filter(
        lambda example: example["label"] != -1,
        desc="Filtering out samples with label -1"
    )
    # Get the metric function
    metric = evaluate.load("accuracy")
    # You can define your custom compute_metrics function. It takes an `EvalPrediction` object (a namedtuple with a
    # predictions and label_ids field) and has to return a dictionary string to float.
    def compute_metrics(p: EvalPrediction):
     
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
       
        preds = np.argmax(preds, axis=1)
        labels = p.label_ids
        if hasattr(preds, 'device'):  # If it's a torch tensor
            preds = preds.detach().cpu().numpy()
            labels = labels.detach().cpu().numpy()

        
        return {"accuracy": (preds == labels).astype(np.float32).mean().item()}

        
    eval_dataloader = DataLoader (
            val,
            batch_size=16,
            shuffle=False,
            pin_memory=False,
            num_workers=0,
            collate_fn=custom_padding_collator,
            drop_last=True 
        )
    model_class = CoFiBertForSequenceClassification
    model_name_or_path = 'out_bert_old/SNLI/CoFi/SNLI_sparsity0.20'
    zs_llm = load_zs(os.path.join(model_name_or_path, 'zs_llm.pt'))
    zs_mlp = load_zs(os.path.join(model_name_or_path, 'zs_mlp.pt'))


    zs = zs_llm | zs_mlp
    # for compressed models
    if zs is None:
        model = model_class.from_pretrained(
            pretrained_model_name_or_path= os.path.join("out_bert_old/SNLI/CoFi/SNLI_sparsity0.20", "model.safetensors"), # if llm part of student model is alr trained itll be here otherwise a default model will be loaded and finetuned
            from_tf=False,
            teacher=True,
            config=config,
            train_data=train,
            max_data=max_data
        
        ) 
    # for full models with compression vectors zs
    else:
        model = load_model(model_name_or_path, model_class, zs)
    model.config.id2label = LABEL_ITOS
    model.config.label2id = LABEL_STOI
    model = model.cuda()
    model = model.eval()

    model.config.output_hidden_states = False
    model.config.output_attentions = False

    metrics = evaluate_(model, eval_dataloader)
    model_size = calculate_parameters(model)
    full_model_size = calculate_parameters(model_class(model.config))
    sparsity = 1 - round(model_size / full_model_size, 3)

    print(f"Task: snli")
    print(f"Model path: {model_name_or_path}")
    print(f"Model size: {model_size}")
    print(f"Sparsity: {sparsity}")
    for key in metrics:
        print(f"{key}: {round(metrics[key], 6 if 'seconds' in key else 4)}")
    print()
