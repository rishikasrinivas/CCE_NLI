import math
import os
import sys
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import data.snli as snli
import numpy as np
import torch
import torch.nn.functional as F
from packaging import version
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from torch.utils.data.distributed import DistributedSampler
from tqdm.auto import tqdm, trange
from transformers import Trainer
from transformers.data.data_collator import DataCollator
from transformers.modeling_utils import PreTrainedModel
from transformers.optimization import get_linear_schedule_with_warmup
from torch.optim import AdamW
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.trainer import Trainer
from transformers.trainer_pt_utils import nested_concat, nested_numpify
from transformers.trainer_utils import (PREFIX_CHECKPOINT_DIR, EvalPrediction,
                                        EvaluationStrategy, PredictionOutput,
                                        TrainOutput)
from transformers.utils import logging
import logging

from transformers.training_args import TrainingArguments
import data.snli as snli 
from args import AdditionalArguments
from cofi.utils.cofi_utils import *
from cofi.utils.utils import *
import torch.nn as nn
import torch.optim as optim
import train_utils
#import wandb
 
from transformers.utils import logging
import util
logger = logging.get_logger(__name__)
 
glue_tasks = {"cola": "matthews_correlation",
              "mnli": "mnli/acc",
              "snli": "accuracy",
              "mrpc": "accuracy",
              "sst2": "accuracy",
              "stsb": "corr",
              "qqp": "accuracy",
              "qnli": "accuracy",
              "rte": "accuracy",
              "sst2_aug": "accuracy",
              "rte_aug": "accuracy",
              "mrpc_aug": "accuracy",
              "qnli_aug": "accuracy",
              "stsb_aug": "corr",}
from torch.nn.utils.rnn import pad_sequence
import torch

def llm_padding_collator(features):
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

def bowman_padding_collator(features):

    batch = {
        "s1": [],
        "s1len": [],
        "s2": [],
        "s2len": [],
        "labels": []
    }

    # Separate and pad each feature
    for feature in features:
        for key in batch.keys():
            if key not in ["labels", "s1len", "s2len"]:
                batch[key].append(torch.tensor(feature[key]))
            else:
                batch[key].append(feature[key])

    # Pad sequences
    padded_batch = {}
    for key in batch:
        if key not in ["labels", "s1len", "s2len"]:
            padded_batch[key] = pad_sequence(
                batch[key],
                padding_value=1  # Use your SNLI's padding index
            )
        else:
            padded_batch[key] = torch.tensor(batch[key])


    return padded_batch
        
class Eval_Counter():
    def __init__(self):
        self.epoch = 0
        self.global_step = 0
        self.best_eval_score = 0
        self.near_sparsity_eval_times = 0
        self.level_best_score = {0.85: 0, 0.8: 0, 0.7: 0,
                                 0.6: 0, 0.75: 0, 0.9: 0, 0.95: 0, 0.65: 0}

    def round_nearest(self, x, a):
        return round(round(x / a) * a, -int(math.floor(math.log10(a))))

    def update(self, epoch, global_step, eval_score):
        best_so_far = False
        print("Best eval score is ", self.best_eval_score)
        if eval_score > self.best_eval_score:
            self.epoch = epoch
            self.global_step = global_step
            self.best_eval_score = eval_score
            best_so_far = True
        return best_so_far

    def clear(self):
        self.eval_score = 0


class CoFiTrainer(Trainer):
    def __init__(
            self,
            model_name, 
            dataset=None,
            model: PreTrainedModel = None,
            config=None, 
            device: str =  'cuda',
            args: TrainingArguments = None,
            additional_args: AdditionalArguments = None,
            data_collator: Optional[DataCollator] = None,
            
            full_train_dataset: Optional[Dataset] = None,
            full_eval_dataset: Optional[Dataset] = None,
            subset_train_dataset: Optional[Dataset] = None,
            subset_val_dataset: Optional[Dataset] = None,
            tokenizer: Optional[PreTrainedTokenizerBase] = None,
            model_init: Callable[[], PreTrainedModel] = None,
            compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
            l0_module=None,
            teacher_model=None,
            teacher_model_dir = None,
            **kwargs,
    ):

        Trainer.__init__(self, model, args, data_collator, subset_train_dataset , subset_val_dataset, tokenizer, model_init, compute_metrics=compute_metrics, **kwargs)
        self.num_workers = 4
        self.model=model
        
        
        self.dataset=dataset
        self.additional_args = additional_args
        self.finetuned_teacher = False
        self.l0_module = l0_module
        self.prepruning_finetune_steps = 100
        self.start_prune = False
        self.teacher_model_dir=teacher_model_dir
        self.config=config
        self.full_train_data=full_train_dataset
        self.full_val_data=full_eval_dataset
        self.subset_train_data=subset_train_dataset
        self.subset_val_data=subset_val_dataset
        
        self.student_optimizer=None
        self.teacher_optimizer=None
        self.l0_optimizer = None
        self.lagrangian_optimizer = None
        self.eval_counter = Eval_Counter()
        self.start_saving_best = True if self.additional_args.pruning_type is None else False
        self.model_name = model_name
        self.pruned_sparsity = 0.0

        self.teacher_model = teacher_model
        if self.teacher_model is not None:
            self.teacher_model = self.teacher_model.to(self.args.device)
        print("BEFORE ENTERING CODE===="*80)
        for n,p in self.model.named_parameters():
            if n=='encoder.rnn.weight_ih_l0':
                print("student ", n,p)
        for n,p in self.teacher_model.named_parameters():
            if n=='encoder.rnn.weight_ih_l0':
                print("student ", n,p)
        print("===="*80)
            
        log_level = args.get_process_log_level()
        logging.set_verbosity(log_level)
        logger.setLevel(log_level)
        
        self.full_eval_dataloader = self.full_val_data
        print(self.model_name)
        
        self.full_train_dataloader = self.full_train_data
        
        self.eval_dataloader = self.subset_val_data
        print(self.model_name)
        
        self.train_dataloader = self.subset_train_data
        
        self.device=device

    def create_optimizer_and_scheduler(self, num_training_steps: int, build_l0_optimizer:bool=True, student=True):
        def log_params(param_groups, des):
            for i, grouped_parameters in enumerate(param_groups):
                logger.info(
                    f"{des}, number of params: {sum(p.nelement() for p in grouped_parameters['params'])}, weight_decay: {grouped_parameters['weight_decay']}, lr: {grouped_parameters['lr']}")

        if self.student_optimizer is None or self.teacher_optimizer is None:
            no_decay = ["bias", "LayerNorm.weight"]
            

            
            if self.student_optimizer is None:
                student_main_model_params = [
                    {
                        "params": [p for n, p in self.model.named_parameters() ],
                        "weight_decay": self.args.weight_decay,
                        "lr": self.args.learning_rate
                    },
                ]
                if self.model_name != 'bowman':
                    self.student_optimizer = AdamW(self.model.parameters(), lr=2e-5, eps=1e-8)  # AdamW optimizer is recommended for BERTAdamW(
                else:
                    self.student_optimizer = optim.Adam(self.model.parameters())
                log_params(student_main_model_params, "student main params")

            if self.teacher_optimizer is None:
            
                teacher_main_model_params = [
                    {
                        "params": [p for n, p in self.teacher_model.named_parameters()],
                        "weight_decay": self.args.weight_decay,
                        "lr": self.args.learning_rate
                    },
                ]
          
                if self.model_name != 'bowman':
                    self.teacher_optimizer = AdamW(self.teacher_model.parameters(),  lr=2e-6, eps=1e-8)  # AdamW optimizer is recommended for BERTAdamW(
                else:
                    self.teacher_optimizer = optim.Adam(self.teacher_model.parameters())
                log_params(teacher_main_model_params, "teacher main params")

            
               

            if build_l0_optimizer and self.l0_module is not None:
                l0_params = [{
                    "params": [p for n, p in self.l0_module.named_parameters() if "lambda" not in n],
                    "weight_decay": 0.0,
                    "lr": self.additional_args.reg_learning_rate
                }]
                
                log_params(l0_params, "l0 reg params")
                self.l0_optimizer = AdamW(l0_params,
                                          betas=(self.args.adam_beta1,
                                                 self.args.adam_beta2),
                                          eps=self.args.adam_epsilon, )

                lagrangian_params = [{
                    "params": [p for n, p in self.l0_module.named_parameters() if "lambda" in n],
                    "weight_decay": 0.0,
                    "lr": -self.additional_args.reg_learning_rate
                }]
                log_params(lagrangian_params, "l0 reg lagrangian params")
                self.lagrangian_optimizer = AdamW(lagrangian_params,
                                                    betas=(self.args.adam_beta1,
                                                            self.args.adam_beta2),
                                                    eps=self.args.adam_epsilon)

        if self.lr_scheduler is None:
            if self.additional_args.scheduler_type == "linear":
                self.lr_scheduler = get_linear_schedule_with_warmup(
                    self.student_optimizer, num_warmup_steps=self.args.warmup_steps, num_training_steps=num_training_steps
                )
            else:
                self.lr_scheduler = None
                

    
    def train(self):
        
        
 
        num_update_steps_per_epoch = len(
            self.train_dataloader) // self.args.gradient_accumulation_steps
        num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1) #! 12272
        

        if self.l0_module is not None:
            lagrangian_warmup_steps = self.additional_args.lagrangian_warmup_epochs * num_update_steps_per_epoch #!24544 = 2*12272
            # self.prepruning_finetune_steps = self.additional_args.prepruning_finetune_epochs * num_update_steps_per_epoch
            self.l0_module.set_lagrangian_warmup_steps(lagrangian_warmup_steps)
            logger.info(f"Prepruning finetune steps: {self.prepruning_finetune_steps}")
            logger.info(f"Lagrangian warmup steps: {lagrangian_warmup_steps}")

        if self.args.max_steps > 0:
            self.t_total = self.args.max_steps
            num_train_epochs = self.args.max_steps // num_update_steps_per_epoch + int(
                self.args.max_steps % num_update_steps_per_epoch > 0
            )
        else:
            self.t_total = int(num_update_steps_per_epoch *
                               self.args.num_train_epochs)
            num_train_epochs = self.args.num_train_epochs
            self.args.max_steps = self.t_total

        self.create_optimizer_and_scheduler(num_training_steps=self.t_total, build_l0_optimizer = self.start_prune)

        model = self.model
        
                                
        total_train_batch_size = 1

        logger.info("***** Running training *****")

        logger.info("  Num examples = %d", self.num_examples(self.train_dataloader))
        logger.info("  Num Epochs = %d", num_train_epochs)
        logger.info("  Instantaneous batch size per device = %d",
                    self.args.per_device_train_batch_size)
        logger.info(
            "  Total train batch size (w. parallel, distributed & accumulation) = %d", total_train_batch_size)
        logger.info("  Gradient Accumulation steps = %d",
                    self.args.gradient_accumulation_steps)
        logger.info("  Total optimization steps = %d", self.t_total)

        self.global_step = 0
        self.epoch = 0
        self.total_flos = 0

        epochs_trained = 0

        tr_loss = torch.tensor(0.0).to(self.args.device)
        reg_loss = torch.tensor(0.0).to(self.args.device)
        lag_loss = torch.tensor(0.0).to(self.args.device)

        logging_loss_scalar = 0.0
        logging_reg_loss_scalar = 0.0
        logging_lag_loss_scalar = 0.0

        model.zero_grad()
        if self.l0_module is not None:
            self.l0_module.zero_grad()

        self.student_optimizer.zero_grad()
        if self.l0_optimizer is not None:
            self.l0_optimizer.zero_grad()
        if self.lagrangian_optimizer is not None:
            self.lagrangian_optimizer.zero_grad()

        disable_tqdm = self.args.disable_tqdm or not self.is_local_process_zero()
        train_pbar = trange(epochs_trained, int(
            np.ceil(num_train_epochs)), desc="Epoch", disable=disable_tqdm)

        #Train the teacher model first
        if not self.finetuned_teacher:
            print("===="*80)
            for n,p in self.model.named_parameters():
                if n=='encoder.rnn.weight_ih_l0':
                    print("student ", n,p)
            for n,p in self.teacher_model.named_parameters():
                if n=='encoder.rnn.weight_ih_l0':
                    print("student ", n,p)
            print("===="*80)
            
            self.teacher_model = self.finetune_teacher(self.teacher_model)
            self.finetuned_teacher = True

        self.evaluate()
        # training
        print(f"Training for {num_train_epochs} epochs")
        for epoch in range(epochs_trained, int(np.ceil(num_train_epochs))): #! 20 epoch
            
            print(f"Starting epoch {epoch}")
            epoch_start = time.time()

            if isinstance(self.train_dataloader, DataLoader) and isinstance(self.train_dataloader.sampler, DistributedSampler):
                self.train_dataloader.sampler.set_epoch(epoch)
 
            epoch_iterator = self.train_dataloader

            # Reset the past mems state at the beginning of each epoch if necessary.
            if self.args.past_index >= 0:
                self._past = None
       
            epoch_pbar = tqdm(epoch_iterator, desc="Iteration",
                              disable=disable_tqdm)
            self.eval_counter.clear()

            for step, inputs in enumerate(epoch_iterator):
                #print(f"Can only start pruning at {self.global_step} == {self.prepruning_finetune_steps}")
                #print(f"right now, glboal step = {self.global_step} and self.prepruning_finetune_steps = {self.prepruning_finetune_steps}" )
                
                if self.prepruning_finetune_steps > 0 and self.global_step == self.prepruning_finetune_steps: #! before pruning, run 12272 steps
                   
                    logger.warning("started pruning")
                    self.start_prune = True
                    self.student_optimizer = None
                    self.lr_scheduler = None
                    lr_steps = self.t_total - self.global_step

                    # reset the optimizer
                    self.create_optimizer_and_scheduler(lr_steps, self.start_prune)
                    logger.info("Starting l0 regularization!")
            
                
                if self.start_prune:
                    zs = self.l0_module.forward(training=True) #! get the zs
                    
                    self.fill_inputs_with_zs(zs, inputs) #! use the zs
                
                    
 
                
                loss_terms =  self.training_step(model, inputs)
                tr_loss_step = loss_terms["loss"]
                lag_loss_step = loss_terms["lagrangian_loss"]
          

                tr_loss += tr_loss_step
                lag_loss += lag_loss_step if lag_loss_step is not None else 0.0

                self.total_flos += self.floating_point_ops(inputs)
                
                if (step + 1) % self.args.gradient_accumulation_steps == 0 or (
                        len(epoch_iterator) <= self.args.gradient_accumulation_steps
                        and(step + 1) == len(epoch_iterator)
                ):
                    
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), self.args.max_grad_norm)
                    
                    self.student_optimizer.step()

                    if self.l0_module is not None and self.l0_optimizer is not None:
                        self.l0_optimizer.step()
                        self.lagrangian_optimizer.step()

                    if self.lr_scheduler is not None:
                        self.lr_scheduler.step()

                    if self.l0_module is not None:
                        self.l0_module.constrain_parameters()

                    model.zero_grad()
                    if self.l0_module is not None:
                        self.l0_module.zero_grad()
                    self.student_optimizer.zero_grad()
                    if self.l0_optimizer is not None:
                        self.l0_optimizer.zero_grad()
                    if self.lagrangian_optimizer is not None:
                        self.lagrangian_optimizer.zero_grad()

                    self.global_step += 1
                    self.epoch = epoch + (step + 1) / len(epoch_iterator)

                    if (self.args.logging_steps > 0 and self.global_step % self.args.logging_steps == 0) or (
                            self.global_step == 1 and self.args.logging_first_step
                    ):
                        logs: Dict[str, float] = {}
                        tr_loss_scalar = tr_loss.item()
                        reg_loss_scalar = reg_loss.item()
                        lag_loss_scalar = lag_loss.item()

                        logs["loss"] = (
                            tr_loss_scalar - logging_loss_scalar) / self.args.logging_steps
                        logs["reg_loss"] = (
                            reg_loss_scalar - logging_reg_loss_scalar) / self.args.logging_steps
                        logs["lag_loss"] = (
                            lag_loss_scalar - logging_lag_loss_scalar) / self.args.logging_steps

                        # backward compatibility for pytorch schedulers
                        if self.lr_scheduler is not None:
                            lr = self.lr_scheduler.get_last_lr()[0] if version.parse(
                                torch.__version__) >= version.parse("1.4") else self.lr_scheduler.get_lr()[0]
                        else:
                            lr = self.args.learning_rate

                        logs["learning_rate"] = lr
                        logging_loss_scalar = tr_loss_scalar
                        logging_reg_loss_scalar = reg_loss_scalar
                        logging_lag_loss_scalar = lag_loss_scalar

                        self.log(logs)
                        print(f"Global step: {self.global_step}, eval_steps: {self.args.eval_steps}")

                    if self.global_step % self.args.eval_steps == 0:
                        logger.warning("evaluating")
                        self.evaluate()

                epoch_pbar.update(1)

                if self.args.max_steps > 0 and self.global_step >= self.args.max_steps or  self.pruned_sparsity >= self.additional_args.target_sparsity:
                    break

            epoch_end = time.time()
            # wandb.log({'epoch':epoch})
            logger.info(
                f"Epoch {epoch} finished. Took {round(epoch_end - epoch_start, 2)} seconds.")

            epoch_pbar.close()
            train_pbar.update(1)

            if self.args.max_steps > 0 and self.global_step >= self.args.max_steps or self.pruned_sparsity >= self.additional_args.target_sparsity:
                break

        train_pbar.close()

        if self.args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of training
            delattr(self, "_past")

        # wandb.log({'global_step':self.global_step,'training_loss':tr_loss.item() / self.global_step})
       
        return TrainOutput(self.global_step, tr_loss.item() / self.global_step, None)
    from accelerate.utils import tqdm
    import torch.nn as nn
    def prediction_loop(self, dataloader: DataLoader, description: str, prediction_loss_only: Optional[bool] = None) -> PredictionOutput:
        prediction_loss_only = (
            prediction_loss_only if prediction_loss_only is not None else self.args.prediction_loss_only
        )
        
        
        
        print("Is iterable:", hasattr(dataloader, '__iter__'))
        # disable output hidden states and attention during evaluation
        if self.model_name != 'bowman':
            self.model.config.output_hidden_states = False
            self.model.config.output_attentions = False

        model = self.model
        

        # multi-gpu eval

        batch_size = dataloader.batch_size
        
        # Initialize containers
        # losses/preds/labels on GPU/TPU (accumulated for eval_accumulation_steps)
        losses_host = None
        preds_host = None
        labels_host = None
        # losses/preds/labels on CPU (final containers)
        all_losses = None
        all_preds = None
        all_labels = None
        model.eval()

        if self.args.past_index >= 0:
            self._past = None

        disable_tqdm = not self.is_local_process_zero() or self.args.disable_tqdm

        zs = None
        if self.start_prune:
            self.l0_module.eval()
            zs = self.l0_module.forward(training=False)
        
        if zs is not None:
            pruned_model_size_info = self.l0_module.calculate_model_size(zs)

    
        for ii, inputs  in enumerate(dataloader): #here instead do for ii, (s1,s1l,s2,s2l,labels) in enumerate(dataloader) using the algined w our model appraoch
           
            if zs is not None:
                if ii == 0:
                    logger.info(f"Putting zs {zs.keys()} into inputs:")
                self.fill_inputs_with_zs(zs, inputs) #! use the zd
            
            loss, logits, labels = self.prediction_step(
                model, inputs, prediction_loss_only) #pass instead of "inputs"
           
            batch_size = inputs[list(inputs.keys())[0]].shape[0]
            #13x16x62x768 #logits[1][0] is premise hidden states logits[1][1] is hyp hidden similarly logits[2] is attn mask [0] for pre [1] for hyp
            if logits is not None:
                   # print("shape of arbitrary hidden state 2 in first logtis ", preds_host[2][0][2].shape)
                preds_host = logits[0][2] if preds_host is None else nested_concat(
                    preds_host, logits[0][2])
            if labels is not None:
                labels_host = labels if labels_host is None else nested_concat(
                    labels_host, labels)
            if loss is not None:
                if type(loss) == float:
                    losses = [loss] * batch_size
                    if losses_host is None:
                        losses_host = losses
                    else:
                        losses_host.extend(losses)
                else:
                    losses = loss.repeat(batch_size)
                    losses_host = losses if losses_host is None else torch.cat(
                        (losses_host, losses), dim=0)
        

        if self.args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of the evaluation.py loop
            delattr(self, "_past")

        if losses_host is not None:
            if not torch.is_tensor(losses_host):
                losses_host = torch.tensor(losses_host)
            losses = nested_numpify(losses_host)
            all_losses = losses if all_losses is None else np.concatenate(
                (all_losses, losses), axis=0)
        if preds_host is not None:
            logits = nested_numpify(preds_host)
            
            all_preds = logits if all_preds is None else nested_concat(
                all_preds, logits, padding_index=-100)
            print(all_preds.shape)
        if labels_host is not None:
            labels = nested_numpify(labels_host)
            all_labels = labels if all_labels is None else nested_concat(
                all_labels, labels, padding_index=-100)
            print("all labels ", all_labels.shape)
            
  
        if self.compute_metrics is not None and all_preds is not None and all_labels is not None:
            
            metrics = self.compute_metrics(EvalPrediction(
                predictions=all_preds, label_ids=all_labels))
            #print(metrics)
        else:
            metrics = {}

        if all_losses is not None and len(all_losses) > 0:
            metrics["eval_loss"] = np.mean(all_losses)

        if zs is not None:
            lag_loss, expected_sparsity, target_sparsity = self.l0_module.lagrangian_regularization(
                self.global_step - self.prepruning_finetune_steps)

            expected_sparsity = round(expected_sparsity.item(), 5)
            metrics.update(pruned_model_size_info)
            metrics["expected_sparsity"] = expected_sparsity
            metrics["target_sparsity"] = target_sparsity

            if (not self.start_saving_best) and (target_sparsity - self.additional_args.target_sparsity >= -self.additional_args.sparsity_epsilon):
                self.start_saving_best = True
                logger.info(f"Starting saving the best from epoch {int(self.epoch)} and step {self.global_step}")
        #only fro llm
        if self.model_name != 'bowman':
            model.config.output_hidden_states = True
            model.config.output_attentions = True

        return PredictionOutput(predictions=all_preds, label_ids=all_labels, metrics=metrics)

    def evaluate(self, eval_dataset: Optional[Dataset] = None) -> Tuple[Dict[str, float], List]:
        

   
        output = self.prediction_loop(
            self.eval_dataloader, description="Evaluation")
        

        self.log(output.metrics)
        # wandb.log(output.metrics)
        output.metrics["step"] = self.global_step
        logger.info(f"Evaluating: {output.metrics}")
  
        eval_score = 0
        
        name = glue_tasks['snli']
        print("Name is ", name)
        if isinstance(name, str):
            if name in output.metrics:
                eval_score = output.metrics[name]
        else:
            for na in name:
                if na in output.metrics:
                    eval_score = output.metrics[na]
                    break
        print("Eval score is ", eval_score)
        print(output.metrics)

        # logger.info(f"starting saving best: {self.global_step} {self.start_saving_best}")
    
        if self.start_saving_best:
            self.pruned_sparsity= output.metrics['pruned_model_sparsity']
            print(f"======{self.pruned_sparsity >= self.additional_args.target_sparsity}========")
            if self.pruned_sparsity >= self.additional_args.target_sparsity:
                
                best_so_far = self.eval_counter.update(self.epoch, self.global_step, eval_score)
                print(f"Best so far: {best_so_far}, eval: {eval_score}")
                if best_so_far:
                    print("SAVING MODEL")

                    
                    
                    logger.warning(f"Saving the best model so far: [Epoch {int(self.epoch)} | Step: {self.global_step} | Model size: {output.metrics['remaining_params'] if 'remaining_params' in output.metrics else 'Full' } | Score: {round(eval_score, 5)}]")
                    self.save_model(self.model, self.args.output_dir)
        return output.metrics

    def save_model(self, model, output_dir: Optional[str] = None):
        print("SAVING MODEL")
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        #torch.save(self.l0_module, os.path.join(output_dir, f"l0_module.pt"))

        #zs = self.l0_module.forward(training=False)
        #torch.save(zs, os.path.join(output_dir, f"zs.pt"))

        #self.model.save_pretrained(output_dir)
        if self.l0_module is not None:
                        zs = self.l0_module.forward(training=False)
                        torch.save(zs, os.path.join(output_dir, f"zs.pt"))
                        torch.save(self.l0_module, os.path.join(
                            output_dir, f"l0_module.pt"))
        if self.config:
            self.config.save_pretrained(os.path.join(output_dir))


        # Assuming 'model' is your PyTorch or Hugging Face model
        util.save_checkpoint(
            train_utils.serialize(model, self.model_name, self.dataset),
            is_best=True,
            exp_dir=output_dir,
        )
       

    def calculate_layer_distillation_loss(self, teacher_outputs, student_outputs, zs):
        def normalize(rep):
            return F.layer_norm(rep, rep.shape[-1:])
        #print(f"In calculate_layer_distillation_loss in trainer.py\nteacher_outputs: {teacher_outputs}\nstudent_outputs: {student_outputs}")
        layer_loss=0
        mse_loss = torch.nn.MSELoss(reduction="mean")
        if self.additional_args.do_layer_distill: #! only do layer distill
            mlp_z = None
            head_layer_z = None
            # logger.info(f"zs={zs}")
            if "mlp_z" in zs:
                mlp_z = zs["mlp_z"].detach().cpu()
            if "head_layer_z" in zs:
                head_layer_z = zs["head_layer_z"].detach().cpu()

            
            
            teacher_pre_final_layer_reps, teacher_final_layer_reps = teacher_outputs.logits[0], teacher_outputs.logits[1]
            if self.model_name!='bowman':
                teacher_pre_layer_output = teacher_outputs.hidden_states[0][1:] #! hidden states, with a length of 12. Every has a shape of [32, 65, 768] for pre and hyp
                teacher_hyp_layer_output = teacher_outputs.hidden_states[1][1:] #! hidden states, with a length of 12. Every has a shape of [32, 65, 768] for pre and hyp


                student_pre_final_layer_reps, student_final_layer_reps = student_outputs.logits[0], student_outputs.logits[1]
                student_pre_layer_output = student_outputs.hidden_states[0][1:] 
                student_hyp_layer_output = student_outputs.hidden_states[1][1:] 
            else:
                teacher_pre_layer_output = teacher_outputs.hidden_states[0] #! hidden states, with a length of 12. Every has a shape of [32, 65, 768] for pre and hyp
                teacher_hyp_layer_output = teacher_outputs.hidden_states[1]#! hidden states, with a length of 12. Every has a shape of [32, 65, 768] for pre and hyp


                student_pre_final_layer_reps, student_final_layer_reps = student_outputs.logits[0], student_outputs.logits[1]
                student_pre_layer_output = student_outputs.hidden_states[0]
                student_hyp_layer_output = student_outputs.hidden_states[1]

            #potential issues shape:
            #assert student_pre_layer_output[0].shape == student_hyp_layer_output[0].shape, f"Assertion failed, student premise shape is {student_pre_layer_output[0].shape} but hyp shape is {student_hyp_layer_output[0].shape}"

            # distilliting existing layers
            
            #orig: a single pre + hyp concatentation passed into bert and each layers hidden state given
            #ours: 2 inputs so outputs are hidden state pre hidden state hyp so for each layer msle lossfor pres and hyps
            if self.additional_args.layer_distill_version == 2 or self.model_name=='bowman':
                for layer_num, (t_layer_o, s_layer_o) in enumerate(zip(teacher_pre_layer_output, student_pre_layer_output)):
                    s_layer_o = self.model.layer_transformation(s_layer_o)
                    l = mse_loss(normalize(t_layer_o), normalize(s_layer_o))
                    if l==0:
                        print("Sam outputs")
                    #if mlp_z is None or mlp_z[layer_num] > 0:
                    layer_loss += l
                for layer_num, (t_layer_o, s_layer_o) in enumerate(zip(teacher_hyp_layer_output, student_hyp_layer_output)):
                    s_layer_o = self.model.layer_transformation(s_layer_o)
                    l = mse_loss(normalize(t_layer_o), normalize(s_layer_o))
                    #if mlp_z is None or mlp_z[layer_num] > 0:
                    layer_loss += l
          
            # distilling layers with a minimal distance
            elif self.additional_args.layer_distill_version > 2:
                l = []
                if self.additional_args.layer_distill_version > 4:
                    specified_teacher_layers = [i for i in range(12)]
                    if self.additional_args.layer_distill_version ==5:
                        specified_teacher_layers = sorted(random.sample(specified_teacher_layers, 4))
                    elif self.additional_args.layer_distill_version ==6:
                        result_layers_T= []
                        skip_window = len(specified_teacher_layers)//4
                        for i in range(0, len(specified_teacher_layers), skip_window):
                            result_layers_T.append(random.sample(specified_teacher_layers[i:i+skip_window], 1)[0])
                        specified_teacher_layers = result_layers_T
                    specified_teacher_layers[0] = max(2, specified_teacher_layers[0])
                else:
                    if hasattr(self.model, 'bert'): #bert
                        specified_teacher_layers = [2, 5, 8, 11]
                    elif hasattr(self.model, 'model'): #llama
                        
                        specified_teacher_layers = [2,5,8,11]
                        
                #logger.warning(f"sampled teacher layers: {specified_teacher_layers}")
                
                
                #here not yet accounting for mlp. pulling teaacher output for 2,5,8,11 in bert layers 
                transformed_s_layer_o_pre = [self.model.layer_transformation(
                s_pre_layer_o) for s_pre_layer_o in student_pre_layer_output]
                
                transformed_s_layer_o_hyp = [self.model.layer_transformation(
                s_hyp_layer_o) for s_hyp_layer_o in student_hyp_layer_output]
                    #rn its [layer1 out, layer2 out, etc]
                        #change to [(layer1 pre, layer1 hyp), (layer2 pre, layer2 hyp), etc...]
                
                specified_teacher_layer_reps_pre = [
                    teacher_pre_layer_output[i] for i in specified_teacher_layers] #! teacher: 4x[32,113,768]
                
                
                specified_teacher_layer_reps_hyp = [
                    teacher_hyp_layer_output[i] for i in specified_teacher_layers]
                        #rn its [layer1 out, layer2 out, etc]
                        #change to [(layer1 pre, layer1 hyp), (layer2 pre, layer2 hyp), etc...]

                device = transformed_s_layer_o_hyp[0].device
       
                for t_pre_layer_o, t_hyp_layer_o in zip(specified_teacher_layer_reps_pre,specified_teacher_layer_reps_hyp) :
                    
                    for s_pre_layer_o, s_hyp_layer_o in zip(transformed_s_layer_o_pre, transformed_s_layer_o_hyp): #! student: 12x[32,113,768]
                        l.append(
                            mse_loss(normalize(t_pre_layer_o), normalize(s_pre_layer_o)) + 
                            mse_loss(normalize(t_hyp_layer_o), normalize(s_hyp_layer_o))
                        )#mse(t_layer_pre, s_layer_pre), mse(t_layer_hyp, s_layer_hyp) #cant u add them since its just yhe loss
                
                #now dp lloss for mlp
                
                layerwiseloss = torch.stack(l).reshape(
                    len(specified_teacher_layer_reps_pre), len(student_pre_layer_output)) #! [4,12] cannot do w list of tuples
                
     

                existing_layers = None
                if head_layer_z is not None:
                    existing_layers = head_layer_z != 0
                    existing_layers = existing_layers.to(layerwiseloss.device)

                layer_loss = mse_loss(normalize(student_pre_final_layer_reps), normalize(teacher_pre_final_layer_reps)) + mse_loss(normalize(student_final_layer_reps), normalize(teacher_final_layer_reps))
                #! no ordering restriction specified
                if self.additional_args.layer_distill_version == 3:
                    alignment = torch.argmin(layerwiseloss, dim=1)
                #! added the ordering restriction -> to choose the min loss in 4 student layers
                elif self.additional_args.layer_distill_version in (3, 4, 5, 6):
                    last_aligned_layer = 12 if self.model_name=='bert' else 22
                    alignment = []
                    for search_index in range(len(specified_teacher_layers)-1, -1, -1):
                        indexes = layerwiseloss[search_index].sort()[1]
                        if existing_layers is not None:
                            align = indexes[(
                                indexes < last_aligned_layer) & existing_layers]
                      
                        else:
                            align = indexes[indexes < last_aligned_layer]
                        if len(align) > 0:
                            align = align[0]
                        else:
                            align = last_aligned_layer
                        alignment.append(align)
                        last_aligned_layer = align
                    alignment.reverse()
                    alignment = torch.tensor(alignment).to(device)
                else:
                    logger.info(
                        f"{self.additional_args.layer_distill_version} version is not specified.")
                    sys.exit()

                layerwise = torch.arange(len(specified_teacher_layers)).to(device)
                layer_loss += layerwiseloss[layerwise, alignment].sum() #! layerwise: teacher (specified layers) / alignment: student (min loss layers) / layerwiseloss: [4,12]
                if self.global_step % 100 == 0:
                    logger.info(f"v{self.additional_args.layer_distill_version} Global step: {self.global_step}, Alignment: " + str(alignment))
            return layer_loss
        else:
            return None
    def calculate_distillation_loss(self, teacher_outputs, student_outputs, zs):
        layer_loss = self.calculate_layer_distillation_loss(teacher_outputs, student_outputs, zs)
        distill_loss = layer_loss
        distill_loss = distill_loss.cpu()

        ce_distill_loss = F.kl_div(
            input=F.log_softmax(
                student_outputs[1][2] / self.additional_args.distill_temp, dim=-1), #! logits: [32,3]
            target=F.softmax(
                teacher_outputs[1][2]  / self.additional_args.distill_temp, dim=-1), #! distill_temp: 2.0
            reduction="batchmean") * (self.additional_args.distill_temp ** 2)
        
        loss = self.additional_args.distill_ce_loss_alpha * ce_distill_loss
        if distill_loss is not None:
            loss += self.additional_args.distill_loss_alpha * distill_loss
        #print("Layer loss: ", layer_loss, "Total loss" ,loss, "KL Div",  ce_distill_loss)

        return distill_loss, ce_distill_loss, loss


    def store_results(self,teacher):


        initial_acc = train_utils.run_eval(teacher,self.full_eval_dataloader,self.model_name, pruning_method='cofi')
        import json
        file_path='./initial_accs.json'

        # Sample Python dictionary
        results = {
            self.model_name: initial_acc
            
        }

        # Write to JSON file

        # Step 1: Load existing data
        if os.path.exists(file_path):
            with open(file_path, "r") as f:
                try:
                    data = json.load(f)
                except json.JSONDecodeError:
                    data = []  # If file is empty or invalid, start fresh
        else:
            data = []

        # Step 2: Append new data
        if isinstance(data, list):
            data.append(results)
        else:
            # Handle case where JSON root is not a list (optional)
            data = [data, results]

        # Step 3: Write updated data back to file
        with open(file_path, "w") as f:
            json.dump(data, f, indent=4)

        print("Data written to initial_accs.json")
    def shortens_inputs(self, inputs):
        if self.model_name=='bowman': return inputs
        max_length = inputs["pre_attention_mask"].sum(-1).max().item()
        inputs["pre_input_ids"] = inputs["pre_input_ids"][:, :max_length]
        inputs["pre_attention_mask"] = inputs["pre_attention_mask"][:, :max_length]
        
        max_length = inputs["hyp_attention_mask"].sum(-1).max().item()
        inputs["hyp_input_ids"] = inputs["hyp_input_ids"][:, :max_length]
        inputs["hyp_attention_mask"] = inputs["hyp_attention_mask"][:, :max_length]
        if "token_type_ids" in inputs:
            inputs["token_type_ids"] = inputs["token_type_ids"][:, :max_length]
      
    def finetune_teacher(self,teacher):
        save_teacher_dir= f'0_Pruning_Iter'
        teacher_model_path=os.path.join(self.teacher_model_dir, save_teacher_dir)
        #Load from safensors (not uniform for all pruning iters though)
        '''if save_teacher_dir in os.listdir(self.teacher_model_dir) and 'model.safetensors' in os.listdir(teacher_model_path):
            print(f"Reloading Finetuning SNLI teacher model ")
            state_dict = load_file(os.path.join(teacher_model_path,'model.safetensors'))
            self.teacher_model.load_state_dict(state_dict)'''
            
        # Load rfrm pth (deal)
        if save_teacher_dir in os.listdir(self.teacher_model_dir) and 'model_best.pth' in  os.listdir(teacher_model_path):
            print(f"Reloading Finetuning SNLI teacher model ")
            print(f"Loading from {teacher_model_path}")
            print(os.listdir(self.teacher_model_dir))
            
            state_dict = torch.load(os.path.join(teacher_model_path,'model_best.pth'))['state_dict']
            self.teacher_model.load_state_dict(state_dict, strict=False)
            for n,p in self.teacher_model.named_parameters():
                p.requires_grad = False
            self.store_results(self.teacher_model)
            
             
            for (n,ns), (t,nt) in zip(self.model.named_parameters(), self.teacher_model.named_parameters()):
                print(n,ns,nt)
            return self.teacher_model
        
        os.makedirs(teacher_model_path , exist_ok=True)
        #use full data just for training teacher model
        dataloaders = {
            'train': self.full_train_dataloader,
            'val':self.full_eval_dataloader,
        }
        print(f"training using full loaders", len(self.full_train_dataloader.dataset))
       
        criterion = nn.CrossEntropyLoss()
        
        self.teacher_model = train_utils.finetune_pruned_model(model=teacher,model_type=self.model_name, optimizer=self.teacher_optimizer, pruning_method='cofi', criterion=criterion, dataloaders = dataloaders, finetune_epochs=5, prune_metrics_dir=teacher_model_path,device = self.device)
        weights = teacher.state_dict()
       
        
        util.save_checkpoint(
            train_utils.serialize(self.teacher_model, self.model_name, self.dataset),
            is_best=True,
            exp_dir=teacher_model_path
        )
        
        print(f"Saving to {teacher_model_path}")
        for (n,ns), (t,nt) in zip(self.model.named_parameters(), self.teacher_model.named_parameters()):

            if torch.equal(ns,nt):
                print(f"Layers {n}, {t} are the same")
        self.save_model(self.teacher_model, teacher_model_path)
        return self.teacher_model
    
    
   

    def training_step(self, model: torch.nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> List[torch.Tensor]:
        model.train()
        if self.l0_module is not None:
            self.l0_module.train()
        #inputs; {labels: tensor, input_ids: tensor, token_type_ids: tensor, attention_mask: tensor
        inputs = self._prepare_inputs(inputs)
        distill_loss = None
        distill_ce_loss = None
        if self.teacher_model is not None:
            with torch.no_grad():
                # only retain inputs of certain keys
                #next 3 lines need to be adjusted for bowman

                if self.model_name == 'bowman':
                    teacher_inputs_keys = ["s1", "s1len", "s2", "s2len", "labels"] 
    
                
                else:
                    teacher_inputs_keys = ["hyp_input_ids", "hyp_attention_mask", "pre_input_ids", "pre_attention_mask", "token_type_ids", "position_ids", "labels",
                                       "output_attentions", "output_hidden_states", "return_dict"]
                    
                teacher_inputs = {key: inputs[key]
                                  for key in teacher_inputs_keys if key in inputs}
                self.shortens_inputs(teacher_inputs)
                
                teacher_outputs = self.teacher_model(**teacher_inputs)
          
            self.shortens_inputs(inputs)
            
            student_outputs = self.model(**inputs) #! get the two outputs
     
            
            zs = {key: inputs[key] for key in inputs if "_z" in key} #! extract the zs: keys are 'pre_input_ids', 'pre_attention_mask', 'hyp_input_ids', 'hyp_attention_mask', 'labels', 'head_z', 'intermediate_z', 'hidden_z', 'mlp_z', 'head_layer_z'
            distill_loss, distill_ce_loss, loss = self.calculate_distillation_loss(
                teacher_outputs, student_outputs, zs)
        else:
            loss = self.compute_loss(model, inputs)

        lagrangian_loss = None
        if self.start_prune:
            lagrangian_loss, _, _ = \
                self.l0_module.lagrangian_regularization(
                    self.global_step - self.prepruning_finetune_steps)
            loss += lagrangian_loss

        if self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps

        loss.backward()
        
        # wandb.log({"loss": loss.detach(),
        #         "lagrangian_loss": lagrangian_loss.detach() if lagrangian_loss is not None else None,
        #         "distill_layer_loss": distill_loss.detach() if distill_loss is not None else None,
        #         "distill_ce_loss": distill_ce_loss.detach() if distill_ce_loss is not None else None})
        
        return {"loss": loss.detach(),
                "lagrangian_loss": lagrangian_loss.detach() if lagrangian_loss is not None else None,
                "distill_layer_loss": distill_loss.detach() if distill_loss is not None else None,
                "distill_ce_loss": distill_ce_loss.detach() if distill_ce_loss is not None else None}

    def fill_inputs_with_zs(self, zs, inputs):
        for key in zs:
            inputs[key] = zs[key]
