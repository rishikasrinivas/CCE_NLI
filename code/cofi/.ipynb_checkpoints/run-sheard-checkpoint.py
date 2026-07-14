import logging
import os
import sys
import time
import random
from copy import deepcopy
import collections
import datasets
import numpy as np
import torch
import transformers
import evaluate
import json
from datasets import load_dataset, DatasetDict
from transformers import AutoConfig, AutoTokenizer, EvalPrediction, default_data_collator, DataCollatorWithPadding
from transformers import (HfArgumentParser, TrainingArguments, PretrainedConfig,
                          glue_output_modes, glue_tasks_num_labels, set_seed)
import sys
print(os.listdir(os.getcwd()))
sys.path.append('code/')
from args import AdditionalArguments, DataTrainingArguments
from cofi.utils.cofi_utils import *
from models.cofi_models.mha_l0_module import L0Module_Sheared
from models.cofi_models.modeling_bert import CoFiBertForSequenceClassification
from models.cofi_models.modeling_llama import CoFiLlamaForSequenceClassification
from models.cofi_models.modeling_bowman import CoFiBowmanEntailmentClassifier, TextEncoder
from cofi.trainer.trainer import CoFiTrainer 
from cofi.utils.utils import *
from models.cofi_models.model_args import ModelArguments
import train_utils
#import wandb
import data.snli as snli
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

logger = logging.getLogger("llm.txt")

def load_pruned_structure_then_weights(model_cls, model_dir, zs_path, config, tokenizer, device, **kwargs):
    # 1. Build fresh base/dense model
    model = model_cls(config).to('cpu')

    # 2. Load zs and apply structure pruning
    zs = torch.load(zs_path, map_location='cpu')

    if model.model_name == "bowman":
        update_LSTM_params(model, zs)
        prune_hidden_mlp(zs, model)
    elif model.model_name == "llama":
        update_llama_params(model, zs)
        model = prune_model_with_z(zs, model)
    else:
        update_bert_params(model, zs)
        model = prune_model_with_z(zs, model)

    # 3. Load weights AFTER structure exists
    ckpt_path = os.path.join(model_dir, "model_best.pth")
    ckpt = torch.load(ckpt_path, map_location='cpu')
    result = model.load_state_dict(ckpt["state_dict"], strict=False)

    print("Missing keys:", result.missing_keys)
    print("Unexpected keys:", result.unexpected_keys)

    return model.to(device)

def main():
    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments, AdditionalArguments))

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args, additional_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args, additional_args = parser.parse_args_into_dataclasses()

    training_args.dataloader_num_workers=4
    #print("training args ", training_args)
    os.makedirs(training_args.output_dir, exist_ok=True)
    os.makedirs(data_args.path_to_pretrained, exist_ok=True)
    print(f"model: {model_args}\n data:{data_args}, {training_args}, {additional_args}")
     # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # save args
    torch.save(data_args, os.path.join(
        training_args.output_dir, "data_args.bin"))
    torch.save(model_args, os.path.join(
        training_args.output_dir, "model_args.bin"))
    torch.save(additional_args, os.path.join(
        training_args.output_dir, "additional_args.bin"))

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # print all arguments
    log_all_parameters(logger, model_args, data_args,
                       training_args, additional_args)

    # ====== Data Loading ===========
 
    device = data_args.device
    max_data = None if data_args.data_debug > 0 else None
    train,val,dl = train_utils.create_dataloaders(model_type= additional_args.model_name, pruning_method='CoFi', max_data=None, debug=False)
    #train,val,dl = train_utils.create_dataloaders(model_type=additional_args.model_name, pruning_method='CoFi', max_data=60000, debug=True)
    label_list = list(set(train.labels))
    vocab= {'stoi': train.stoi, 'itos': train.itos}

    if not os.path.exists("../DataLoaders/vocab.json"):
        with open("../DataLoaders/vocab.json", "w") as f:
            json.dump(vocab, f, indent=4) 
    
    # Labels
    is_regression=False
    if is_regression:
        num_labels = 1
    else:
        num_labels = len(set(train.labels))
    
    t_name=data_args.task_name
    trained_teacher=None
    teacher_model=None
    config=None
    print("model pretrained path ", model_args.model_name_or_path)
    if model_args.model_name_or_path.startswith("bert"):
        Teach_Model = CoFiBertForSequenceClassification 
        Student_Model = CoFiBertForSequenceClassification 
    elif model_args.model_name_or_path.startswith('knowledgator'):
        Teach_Model = CoFiLlamaForSequenceClassification
        Student_Model = CoFiLlamaForSequenceClassification
    # ======= Load the model params ========
    base_model_path = os.path.join(data_args.teacher_model_dir, '0_Pruning_Iter', 'model_best.pth')
    pretrained_path = os.path.join(data_args.path_to_pretrained, f'{additional_args.model_name}_MAIN_pretrained_inits.pth')
    print(f"Loding teacher from {base_model_path}")
    if additional_args.model_name in ['bert', 'llama']:
        config = AutoConfig.from_pretrained(
            model_args.model_name_or_path,
            num_labels=num_labels,
            finetuning_task=t_name,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )
        
        if not os.path.exists(os.path.join(data_args.teacher_model_dir, 'config.json')):
            config.save_pretrained(os.path.join(training_args.output_dir, 'config.json'))
            print(f"Saved config to ", os.path.join(data_args.teacher_model_dir, 'config.json'))
        
        
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=model_args.cache_dir,
            use_fast=model_args.use_fast_tokenizer,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token  = tokenizer.eos_token 
            assert additional_args.model_name == 'llama'
        # set up configuration for distillation
        if additional_args.do_distill:
            config.output_attentions = True
            config.output_hidden_states = True
            
        
        if additional_args.do_distill:
        
            teacher_model, trained_teacher = Teach_Model.from_pretrained(
                pretrained_model_name_or_path=base_model_path, #if teacher model alr exists, load that (and that will be at this filepath here) but if teacher model doesnt alr exist another default model will be loaded and trained later (Training checks for same path)
                ckpt= pretrained_path,
                config=config,
                device=device,
                hf_name = model_args.model_name_or_path

            )
            config.do_layer_distill = additional_args.do_layer_distill #! True
            #config.output_hidden_states = True
            
    else:
        tokenizer_teacher = TextEncoder(len(vocab['stoi']))
        tokenizer= TextEncoder(len(vocab['stoi']))
        Teach_Model = CoFiBowmanEntailmentClassifier(tokenizer_teacher, training_args.device)
        Student_Model = CoFiBowmanEntailmentClassifier(tokenizer, training_args.device)
        teacher_model, trained_teacher = Teach_Model.from_pretrained(
                pretrained_model_name_or_path=base_model_path,
                encoder=tokenizer_teacher,
                device=device,
                hf_name = model_args.model_name_or_path

            )
    if teacher_model:
        teacher_model.eval()


    #load an untrained student model which we need to initially finetune before pruning
    if additional_args.pretrained_pruned_model is not None:
        student_path = additional_args.pretrained_pruned_model
    else:
        student_path = os.path.join("/".join(training_args.output_dir.split("/")[:-2]), 'student', 'student_model.pth')
    
    print(f'Loading student model from : {student_path} to {device}')
    
    '''student_model, trained_student = Student_Model.from_pretrained(
        pretrained_model_name_or_path= student_path, # if student model is alr trained itll be here otherwise a default model will be loaded and finetuned
        config=config,
        encoder=tokenizer,
        ckpt= pretrained_path,
        device=device,
        hf_name = model_args.model_name_or_path,
        
    ) #! inside the function, we get the original struct  #! CofiBertForSequenceClassification
    #load other stff fromstudent right here?????
    
    
    '''
    if additional_args.pretrained_pruned_model is not None: #from args menas the model is already pruned:
        student_model = load_pruned_structure_then_weights(
            Student_Model,
            model_dir=training_args.output_dir,
            zs_path=os.path.join(training_args.output_dir, "zs.pt"),
            config=config,
            tokenizer=tokenizer,
            device=device,
            hf_name=model_args.model_name_or_path,
        )
        trained_student = True
       
        
    else:
        pass
        student_model, trained_student = Student_Model.from_pretrained(
            pretrained_model_name_or_path= student_path, # if student model is alr trained itll be here otherwise a default model will be loaded and finetuned
            config=config,
            encoder=tokenizer,
            ckpt= pretrained_path,
            device=device,
            hf_name = model_args.model_name_or_path,

        )
    print(f'Loaded student model from : {student_path} to {student_model.device}, trained? {trained_student}')
    LABEL_STOI = {"entailment": 0, "neutral": 1, "contradiction": 2}
    LABEL_ITOS = {v: k for k, v in LABEL_STOI.items()}
    if config:
        student_model.config.label2id = {f"LABEL_{i}":i for i in range(num_labels)}
         # Some models have set the order of the labels to use, so let's make sure we do use it.
        student_model.config.id2label = LABEL_ITOS
        student_model.config.label2id = LABEL_STOI
    label_to_id = LABEL_STOI
    
    # initialize the layer transformation matrix to be an identity matrix
    assert not additional_args.do_layer_distill if  additional_args.pretrained_pruned_model is not None else additional_args.do_layer_distill, f'Using pruned model: {additional_args.pretrained_pruned_model} - distillatino is on {additional_args.do_layer_distill}'
    
    if additional_args.do_layer_distill:
        initialize_layer_transformation(student_model)

    #logger.info(model)
    logger.info(f"Model size: {calculate_parameters(student_model)}")

    zs = None
    
    
    if additional_args.pretrained_pruned_model is not None:
        print(
            f"Model Size after pruning: {calculate_parameters(student_model)}")

    l0_module = None
    
    assert additional_args.pruning_type is None if additional_args.pretrained_pruned_model is not None else additional_args.pruning_type is not None #if ur using a pruned model, pruning_type should be None
    if additional_args.pruning_type is not None:
        l0_module = L0Module_Sheared(config=config,target_sparsity=additional_args.target_sparsity, pruning_modules= additional_args.pruning_type, device=device)
        '''else:
            l0_module = L0Module(config=config,
                                 model_name=additional_args.model_name,
                                 droprate_init=additional_args.droprate_init,
                                 temperature=additional_args.temperature,
                                 target_sparsity=additional_args.target_sparsity,
                                 pruning_type=additional_args.pruning_type,
                                
                                 args=training_args,
                                full_model_size=calculate_parameters(teacher_model)).to(device)'''
            


    # Padding strategy
    if data_args.pad_to_max_length:
        padding = "max_length"
    else:
        # We will pad later, dynamically at batch creation, to the max sequence length in each batch
        padding = False

   
    
    print("===Mapping =====",LABEL_STOI)
    if data_args.max_seq_length > tokenizer.model_max_length:
        logger.warning(
            f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the"
            f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
        )
    max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)
    

    # Get the metric function
    metric = evaluate.load("accuracy")
    # You can define your custom compute_metrics function. It takes an `EvalPrediction` object (a namedtuple with a
    # predictions and label_ids field) and has to return a dictionary string to float.
    def compute_metrics(p: EvalPrediction):
     
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
       
        preds = np.squeeze(preds) if is_regression else np.argmax(preds, axis=1)
        labels = p.label_ids
        if hasattr(preds, 'device'):  # If it's a torch tensor
            preds = preds.detach().cpu().numpy()
            labels = labels.detach().cpu().numpy()

    
        if data_args.task_name is not None:
            result = metric.compute(predictions=preds, references=labels)
            if len(result) > 1:
                result["combined_score"] = np.mean(list(result.values())).item()
            return result
        elif is_regression:
            return {"mse": ((preds - labels) ** 2).mean().item()}
        else:
            return {"accuracy": (preds == labels).astype(np.float32).mean().item()}

    # Data collator will default to DataCollatorWithPadding when the tokenizer is passed to Trainer, so we change it if
    # we already did the padding.
    if data_args.pad_to_max_length:
        data_collator = default_data_collator
    elif training_args.fp16:
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
    else:
        data_collator = None
   

    trainer = CoFiTrainer(
        model_name = additional_args.model_name,
        model=student_model,
        dataset=train,
        config=config,
        args=training_args,
        additional_args=additional_args,
        full_train_dataset=dl['train'] if training_args.do_train else None,
        full_eval_dataset=dl['val'] if training_args.do_eval else None,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        data_collator=data_collator,
        l0_module=l0_module,
        teacher_model=teacher_model,
        teacher_model_dir = data_args.teacher_model_dir,
        device=device,
        trained_teacher=trained_teacher
    )

    if training_args.do_train:
        trainer.train(using_trained_student=trained_student)
        
        if additional_args.target_sparsity > 0:
            tokenizer.save_pretrained(training_args.output_dir)
       
        #print(trainer.evaluate())
        
    

    


if __name__ == "__main__":
    # wandb.init(project='Cofi')
    os.environ["WANDB_DISABLED"] = "true"
    t_start = time.time()
    main()
    t_end = time.time()
    logger.info(f"Training took {round(t_end - t_start, 2)} seconds.")



   
