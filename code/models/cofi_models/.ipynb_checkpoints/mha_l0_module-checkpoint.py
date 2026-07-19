import math
import sys

import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F

from argparse import Namespace as NS
from typing import Any, List

import numpy as np
limit_a, limit_b, epsilon = -.1, 1.1, 1e-6

class Mask(nn.Module):
    def __init__(self, 
                 name: str,
                 mask_shape: List, 
                 num_params_per_mask: int, 
                 mask_output_shape: List, 
                 target_sparsity: float,
                 target_mask_size: int,
                 device: str,
                 eval_target_model: bool=True) -> None:
        super().__init__()
        self.name = name
        self.num_params_per_mask = num_params_per_mask
        self.mask_output_shape = mask_output_shape
        self.target_sparsity=target_sparsity

        self.droprate_init = 0.5
        self.temperature = 2./3.
        self.magical_number = 0.8
        self.device = device
        
        self.z_loga = self.initialize_mask(mask_shape) 
        self.mask_size = self.z_loga.shape[-1] # the full size of each unit
        self.target_mask_size = target_mask_size
        self.eval_target_model = eval_target_model
        
    def get_size(self):
        return self.mask_output_shape 
    def param_init_fn(self, module):
        """ Initialize the parameters for masking variables. """
        mean = math.log(1 - self.droprate_init) - math.log(self.droprate_init)
        mean = 5
        if isinstance(module, nn.Parameter):
            module.data.normal_(mean, 1e-2)
        else:
            for tensor in module.parameters():
                tensor.data.normal_(mean, 1e-2)
        
    def initialize_mask(self, mask_shape: List):
        """ Initialize the parameters for masking variables. """
        z_loga = nn.Parameter(torch.ones(*mask_shape, device=self.device))
        self.param_init_fn(z_loga)
        return z_loga

    def cdf_qz(self, z_loga: torch.Tensor = None):
        """Implements the CDF of the 'stretched' concrete distribution"""
        if z_loga is None:
            z_loga = self.z_loga
        xn = (0 - limit_a) / (limit_b - limit_a)
        logits = math.log(xn) - math.log(1 - xn)
        return torch.sigmoid(logits * self.temperature - z_loga).clamp(min=epsilon, max=1 - epsilon)
    
    def get_eps(self, size: List):
        """Uniform random numbers for the concrete distribution"""
        eps = torch.FloatTensor(size).uniform_(epsilon, 1-epsilon)
        eps = Variable(eps) # is it a must?
        return eps
    
    def quantile_concrete(self, eps: torch.Tensor):
        y = torch.sigmoid((torch.log(eps) - torch.log(1 - eps) + self.z_loga) / self.temperature)
        return y * (limit_b - limit_a) + limit_a
    
    def sample_z(self):
        eps = self.get_eps(torch.FloatTensor(*self.z_loga.shape)).to(self.z_loga.device)
        z = self.quantile_concrete(eps)
        z = F.hardtanh(z, min_val=0, max_val=1).reshape(*self.mask_output_shape)
        return z
    
    def _deterministic_z(self, z_loga):
        # Following https://github.com/asappresearch/flop/blob/e80e47155de83abbe7d90190e00d30bfb85c18d5/flop/hardconcrete.py#L8 line 103
        if self.target_mask_size is None or not self.eval_target_model:
            expected_score = 1 - self.cdf_qz(z_loga)
            expected_num_nonzeros = expected_score.sum()
            expected_num_zeros = z_loga.nelement() - expected_num_nonzeros.item()
        else:
            expected_num_zeros = self.mask_size - self.target_mask_size 
        try:
            num_zeros = round(expected_num_zeros)
        except:
            print("num of zeros is nan....")
            sys.exit()
        soft_mask = torch.sigmoid(z_loga / self.temperature * self.magical_number)
        if num_zeros > 0:
            if soft_mask.ndim == 0:
                soft_mask = torch.tensor(0).to(self.z_loga.device)
            else:
                _, indices = torch.topk(soft_mask, k=num_zeros, largest=False)
                soft_mask[indices] = 0.
        return soft_mask
    
    def deterministic_z(self):
        if self.z_loga.ndim == 1:
            z = self._deterministic_z(self.z_loga).reshape(*self.mask_output_shape)
        else:
            z_loga = self.z_loga.reshape(-1, self.z_loga.shape[-1])
            z = []
            for i in range(z_loga.shape[0]):
                z_ = self._deterministic_z(z_loga[i])
                z.append(z_)
            z = torch.stack(z).reshape(*self.mask_output_shape)
        return z
    
    def forward(self):
        print(self.training)
        func = self.sample_z if self.training else self.deterministic_z
        z = func(self.z_loga).reshape(self.mask_output_shape)
        return z            
            
    def constrain_parameters(self):
        self.z_loga.data.clamp_(min=math.log(1e-2), max=math.log(1e2))

    def calculate_expected_score_sparsity(self):
        score = 1 - self.cdf_qz()
        sparsity = 1 - score.sum(-1) / self.mask_size
        return score, sparsity
 
class L0Module_Sheared(nn.Module):
    def __init__(self, config, device, target_sparsity, pruning_modules):

        super(L0Module_Sheared, self).__init__()
        self.config=config
        
        if 'bowman' in config._name_or_path:
            self.n_matrix_mlp = 0
        elif "bert" in config._name_or_path:
            self.n_matrix_mlp = 2
        elif 'llama' in config._name_or_path.lower():
            self.n_matrix_mlp = 3
        
        
        self.pruning_modules = pruning_modules.split("+") 
        self.set_model_info(config, n_matrix_mlp=self.n_matrix_mlp) 
        
        target_model_cfg= None
        self.target_model_info = None
        # l0 config
        
        
    
        self.start_sparsity = 0.0
        self.lagrangian_warmup_steps = 0
        self.device = device

        self.eval_target_model = False
        
        # l0 params
        self.lambdas = {}
        self.lambdas["lambda_1"] = torch.nn.Parameter(torch.tensor(0.0, device=device))
        self.lambdas["lambda_2"] = torch.nn.Parameter(torch.tensor(0.0, device=device))
        self.masks = {}
        for pruning_module in self.pruning_modules:
            self.initialize_one_module(pruning_module)
        self.masks = torch.nn.ModuleDict(self.masks)
        self.lambdas = torch.nn.ParameterDict(self.lambdas)
        
        # config after initialization
        self.prunable_model_size = self.calculate_prunable_model_size()
        if target_model_cfg is not None:
            self.prunable_target_model_size = self.calculate_prunable_model_size(self.target_model_info)
            self.target_sparsity = 1 - self.prunable_target_model_size / self.prunable_model_size
        else:
            self.target_sparsity =target_sparsity
    
        
        print("********** Initializing L0 Module **********") 
        for pruning_module in self.pruning_modules:
            print(f"***** {pruning_module} *****")
            print(f"z.shape", self.masks[pruning_module].z_loga.shape)
            print(f"size", self.masks[pruning_module].mask_size)
        print(f"prunable model size: {self.prunable_model_size}")
        
    def set_lagrangian_warmup_steps(self, lagrangian_warmup):
        self.lagrangian_warmup_steps = lagrangian_warmup
        
    def set_model_info(self, cfg, n_matrix_mlp):
        
        
        #bert llama only
        self.hidden_size = cfg.hidden_size if cfg is not None else 512
        if "bert" in self.config._name_or_path.lower() or 'llama' in self.config._name_or_path.lower():
            self.intermediate_size = cfg.intermediate_size
            self.num_attention_heads = cfg.num_attention_heads
            self.mlp_num_per_layer = 1
            self.dim_per_head = self.hidden_size // self.num_attention_heads 
            self.num_layers = cfg.num_hidden_layers
            self.vocab_size = cfg.vocab_size

        # also bowman
        
        self.final_mlp_hidden = 1024
        self.out_params=3 
        
        #bert llama only
        if "bert" in self.config._name_or_path.lower() or 'llama' in self.config._name_or_path.lower():
            self.params_per_head_layer = self.hidden_size * self.hidden_size * 4
            self.params_per_head =  self.params_per_head_layer // self.num_attention_heads
            self.params_per_mlp_layer = self.hidden_size * self.intermediate_size * self.n_matrix_mlp
            self.params_per_intermediate_dim = self.params_per_mlp_layer // self.intermediate_size
        
        
        self.params_finalmlp_layer = self.hidden_size * 4 * self.final_mlp_hidden + (self.final_mlp_hidden* self.out_params)  
        self.params_finalmlp_layer=self.params_finalmlp_layer
        self.params_per_hidden_dim_final_mlp = self.params_finalmlp_layer // self.final_mlp_hidden
        self.full_model_size = self.calculate_prunable_model_size()
        
        
    def calculate_prunable_model_size(self):
        
        prunable_model_size = 0
        
        if "head_layer" in self.pruning_modules or "head" in self.pruning_modules:
            prunable_head_layer_size = self.params_per_head_layer * self.num_layers
            prunable_model_size += prunable_head_layer_size
        if "mlp" in self.pruning_modules or "intermediate" in self.pruning_modules:
            prunable_mlp_size = self.params_per_mlp_layer * self.num_layers
            prunable_model_size += prunable_mlp_size
        prunable_model_size += self.params_finalmlp_layer
        
        if "hidden" in self.pruning_modules:
            return prunable_mlp_size + prunable_head_layer_size + self.params_finalmlp_layer

        return prunable_model_size
        
    def initialize_one_module(self, module_name: str):
        func_name = f"initialize_{module_name}"
        try:
            method = getattr(self, func_name)
        except AttributeError:
            raise NotImplementedError("Instance `{}` does not implement `{}`".format(self, func_name))
        method()
            
    def initialize_hidden(self):
        mask_shape = [self.hidden_size]
        num_params_per_mask=self.hidden_size * 4 + self.hidden_size * 4 * 2
        
        target_hidden_sparsity = None; pd=None; target_mask_size=None; 
        if self.target_model_info is not None:
            target_hidden_sparsity = 1 - self.target_model_info.hidden_size / self.hidden_size
            target_mask_size = self.target_model_info.hidden_size
            pd = {"lambda_1_hidden": torch.nn.Parameter(torch.tensor(0.0, device=self.device)),
                  "lambda_2_hidden": torch.nn.Parameter(torch.tensor(0.0, device=self.device))}
            self.lambdas.update(pd)
        
        hidden_mask = Mask(name="hidden",
                           mask_shape=mask_shape,
                           num_params_per_mask=num_params_per_mask,
                           mask_output_shape=[self.hidden_size],
                           target_sparsity=target_hidden_sparsity,
                           target_mask_size=None,
                           device=self.device,
                           eval_target_model=self.eval_target_model)
        self.masks["hidden"] = hidden_mask

    def initialize_head(self):
        mask_shape = [self.num_layers, self.num_attention_heads]
        num_params_per_mask = self.params_per_head
        mask_output_shape = [self.num_layers, 1, self.num_attention_heads, 1] 
        
        target_head_sparsity = None; pd = {} ; target_mask_size=None; 
        if self.target_model_info is not None:
            target_head_sparsity = 1 - self.target_model_info.num_attention_heads / self.num_attention_heads
            target_mask_size = self.target_model_info.num_attention_heads
            pd = {"lambda_1_head": torch.nn.Parameter(torch.tensor(0.0, device=self.device)),
                  "lambda_2_head": torch.nn.Parameter(torch.tensor(0.0, device=self.device))}
            self.lambdas.update(pd)
        head_mask = Mask(name="head",
                         mask_shape=mask_shape,
                         num_params_per_mask=num_params_per_mask,
                         mask_output_shape=mask_output_shape,
                         target_sparsity=target_head_sparsity,
                         target_mask_size=None,
                           device=self.device,
                           eval_target_model=self.eval_target_model)
        self.masks["head"] = head_mask 

    def initialize_qk_head_dim(self): # only campatible when target model info is available
        mask_shape = [self.num_layers, self.num_attention_heads, self.dim_per_head]
        num_params_per_mask = 2 * self.hidden_size
        mask_output_shape = [self.num_layers, self.hidden_size] 
        
        target_qk_head_dim_sparsity = None; pd = {} 
        if self.target_model_info is not None:
            target_qk_head_dim_sparsity = 1 - self.target_model_info.hidden_size / self.hidden_size
            pd = {"lambda_1_qk_head_dim": torch.nn.Parameter(torch.tensor(0.0, device=self.device)),
                  "lambda_2_qk_head_dim": torch.nn.Parameter(torch.tensor(0.0, device=self.device))}
            self.lambdas.update(pd)
        qk_head_dim = Mask(name="qk_head_dim",
                         mask_shape=mask_shape,
                         num_params_per_mask=num_params_per_mask,
                         mask_output_shape=mask_output_shape,
                         target_sparsity=target_qk_head_dim_sparsity,
                         target_mask_size=None,
                         device=self.device)
        self.masks["qk_head_dim"] = qk_head_dim 
          
          
    def initialize_vo_head_dim(self): # only campatible when target model info is available
        mask_shape = [self.num_layers, self.num_attention_heads, self.dim_per_head]
        num_params_per_mask = 2 * self.hidden_size
        mask_output_shape = [self.num_layers, self.hidden_size] 
        
        target_vo_head_dim_sparsity = None; pd = {} 
        if self.target_model_info is not None:
            target_vo_head_dim_sparsity = 1 - self.hidden_size / self.hidden_size
            pd = {"lambda_1_vo_head_dim": torch.nn.Parameter(torch.tensor(0.0, device=self.device)),
                  "lambda_2_vo_head_dim": torch.nn.Parameter(torch.tensor(0.0, device=self.device))}
            self.lambdas.update(pd)
        vo_head_dim = Mask(name="vo_head_dim",
                         mask_shape=mask_shape,
                         num_params_per_mask=num_params_per_mask,
                         mask_output_shape=mask_output_shape,
                         target_sparsity=target_vo_head_dim_sparsity,
                           target_mask_size=None,
                         device=self.device)
        self.masks["vo_head_dim"] = vo_head_dim 
        
    def initialize_head_layer(self):
        mask_shape = [self.num_layers]
        num_params_per_mask=self.params_per_head *  self.num_attention_heads
        mask_output_shape = [self.num_layers] 
        
        target_head_layer_sparsity = None; pd = {}; target_mask_size=None; 
        if self.target_model_info is not None:
            target_head_layer_sparsity = 1 - self.target_model_info.num_layers / self.num_layers
            target_mask_size = self.target_model_info.num_layers
            pd = {"lambda_1_head_layer": torch.nn.Parameter(torch.tensor(0.0, device=self.device)),
                  "lambda_2_head_layer": torch.nn.Parameter(torch.tensor(0.0, device=self.device))}
            self.lambdas.update(pd)
        
        head_layer_mask = Mask(name="head_layer",
                              mask_shape=mask_shape,
                               num_params_per_mask=num_params_per_mask,
                               mask_output_shape=mask_output_shape,
                               target_sparsity=target_head_layer_sparsity,
                               target_mask_size=None,
                           device=self.device,
                           eval_target_model=self.eval_target_model)
        self.masks["head_layer"] = head_layer_mask
        
    def initialize_intermediate(self):
        mask_shape = [self.num_layers, self.intermediate_size]
        num_params_per_mask=self.params_per_intermediate_dim
        mask_output_shape = [self.num_layers, 1, 1, self.intermediate_size] 
        
        target_int_sparsity = None; pd = {}; target_mask_size=None; 
        if self.target_model_info is not None:
            target_int_sparsity = 1 - self.target_model_info.intermediate_size / self.intermediate_size
            target_mask_size = self.target_model_info.intermediate_size
            pd = {"lambda_1_intermediate": torch.nn.Parameter(torch.tensor(0.0, device=self.device)),
                  "lambda_2_intermediate": torch.nn.Parameter(torch.tensor(0.0, device=self.device))}
            self.lambdas.update(pd)
        
        int_mask = Mask(name="intermediate",
                        mask_shape=mask_shape,
                        num_params_per_mask=num_params_per_mask,
                        mask_output_shape=mask_output_shape,
                        target_sparsity=target_int_sparsity,
                        target_mask_size=None,
                           device=self.device,
                           eval_target_model=self.eval_target_model)
        self.masks["intermediate"] = int_mask
       

    def initialize_mlp(self):
        mask_shape = [self.num_layers]
        num_params_per_mask=self.params_per_mlp_layer
        mask_output_shape = [self.num_layers] 
        
        target_mlp_sparsity = None; pd = {}; target_mask_size=None; 
        if self.target_model_info is not None:
            target_mlp_sparsity = 1 - self.target_model_info.num_layers / self.num_layers
            target_mask_size = self.target_model_info.num_layers
            pd = {"lambda_1_mlp": torch.nn.Parameter(torch.tensor(0.0, device=self.device)),
                  "lambda_2_mlp": torch.nn.Parameter(torch.tensor(0.0, device=self.device))}
            self.lambdas.update(pd)
        
        mlp_mask = Mask(name="mlp",
                        mask_shape=mask_shape,
                        num_params_per_mask=num_params_per_mask,
                        mask_output_shape=mask_output_shape,
                        target_sparsity=target_mlp_sparsity,
                        target_mask_size=None,
                           device=self.device,
                           eval_target_model=self.eval_target_model)
        self.masks["mlp"] = mlp_mask 

    def initialize_layer(self):
        mask_shape = [self.num_layers]
        num_params_per_mask=self.params_per_head * self.num_attention_heads + self.params_per_mlp_layer
        mask_output_shape = [self.num_layers] 
        
        target_layer_sparsity = None; target_mask_size=None;  pd = {}
        if self.target_model_info is not None:
            target_layer_sparsity = 1 - self.target_model_info.num_layers / self.num_layers
            target_mask_size = self.target_model_info.num_layers
            pd = {"lambda_1_layer": torch.nn.Parameter(torch.tensor(0.0, device=self.device)),
                  "lambda_2_layer": torch.nn.Parameter(torch.tensor(0.0, device=self.device))}
            self.lambdas.update(pd)
        
        layer_mask = Mask(name="layer",
                         mask_shape=mask_shape,
                          num_params_per_mask=num_params_per_mask,
                          mask_output_shape=mask_output_shape,
                          target_sparsity=target_layer_sparsity,
                          target_mask_size=None,
                           device=self.device,
                           eval_target_model=self.eval_target_model) 
        self.masks["layer"] = layer_mask 
        
    def initialize_final_mlp_hidden(self): #also add final_layer_hid_mlp to self.types
        target_layer_sparsity = None; target_mask_size=None;  pd = {}
        mask_shape = [self.final_mlp_hidden]
        num_params_per_mask=self.params_finalmlp_layer
        mask_output_shape = [self.final_mlp_hidden] 
        
        final_mlp_mask = Mask(name="final_mlp_hidden",
                              mask_shape=mask_shape,
                               num_params_per_mask=num_params_per_mask,
                               mask_output_shape=mask_output_shape,
                              target_sparsity=target_layer_sparsity,
                              target_mask_size=None,
                           device=self.device)
        
        self.masks["final_mlp_hidden"] = final_mlp_mask
        
    
    def constrain_parameters(self):
        for key in self.masks:
            self.masks[key].constrain_parameters()

    def calculate_expected_score_sparsity(self):
        expected_scores = {}
        expected_sparsitys = {}
        for key in self.masks:
            score, sparsity = self.masks[key].calculate_expected_score_sparsity()
            expected_scores[key] = score
            expected_sparsitys[key] = sparsity
        return expected_scores, expected_sparsitys
    
    def transform_scores_for_head(self, expected_scores: dict):
        head_score = expected_scores["head"] # 12 * 12

        head_layer_score = None
        if "head_layer" in expected_scores:
            head_layer_score = expected_scores["head_layer"]
        elif "layer" in expected_scores:
            head_layer_score = expected_scores["layer"] # 12
        if head_layer_score is not None:
            head_layer_score = head_layer_score.view(-1, 1) # 12 * 1
       
        return head_layer_score, head_score

    def transform_scores_for_mlp(self, expected_scores: dict):
        mlp_score = None
        if "mlp" in expected_scores:
            mlp_score = expected_scores["mlp"] # 12
        elif "layer" in expected_scores:
            mlp_score = expected_scores["layer"] # 12
        if mlp_score is not None:
            mlp_score = mlp_score.unsqueeze(-1)
        
        intermediate_score = expected_scores["intermediate"] # 12 * 3072
        return mlp_score, intermediate_score


    def get_expected_num_params(self, expected_scores: dict): #! calculate the current parsity
        num_parameters = 0
       
        # 12 * 1 
        # 12 * 12
        if "bert" in self.config._name_or_path.lower() or 'llama' in self.config._name_or_path.lower():
            head_layer_score, head_score = self.transform_scores_for_head(expected_scores)
            mlp_score, int_score = self.transform_scores_for_mlp(expected_scores)

            head_score = (head_layer_score * head_score) # 12 * 12
            int_score = (mlp_score * int_score) # 12 * 3072

            qk_score = None
            if "qk_head_dim" in expected_scores:
                qk_head_dim_score = expected_scores["qk_head_dim"] # num_layer * hidden_size
                vo_head_dim_score = expected_scores["vo_head_dim"] # num_layer * hidden_size
                qk_head_dim_score = qk_head_dim_score.view(qk_head_dim_score.shape[0], -1) # 12 * 768
                vo_head_dim_score = vo_head_dim_score.view(vo_head_dim_score.shape[0], -1) # 12 * 768
                head_score = torch.repeat_interleave(head_score, self.dim_per_head, dim=1) # 12 * 768

                qk_score = head_score * qk_head_dim_score # 12 * 768
                vo_score = head_score * vo_head_dim_score # 12 * 768

            if "hidden" in expected_scores:
                hidden_score = expected_scores["hidden"] # 768 

                if qk_score is None:
                    num_parameters += torch.outer(hidden_score, head_score.reshape(-1)).sum() * self.masks.head.num_params_per_mask / self.hidden_size # 768 * 144
                    num_parameters += torch.outer(hidden_score, int_score.reshape(-1)).sum() * self.masks.intermediate.num_params_per_mask / self.hidden_size # 768 * 36864
                else:
                    num_parameters += torch.sum(torch.matmul(hidden_score.reshape(1, -1, 1), qk_score.unsqueeze(1))) * 2 # 12 * 768 * 768
                    num_parameters += torch.sum(torch.matmul(hidden_score.reshape(1, -1, 1), vo_score.unsqueeze(1))) * 2 # 12 * 768 * 768
                    num_parameters += torch.sum(torch.matmul(hidden_score.reshape(1, -1, 1), int_score.unsqueeze(1))) * 3 # 12 * 768 * 3072

            else:
                assert False, f'must prune hidden dims'
                num_parameters += torch.sum(head_score) * self.masks.head.num_params_per_mask
                num_parameters += torch.sum(int_score) * self.masks.intermediate.num_params_per_mask
            
        else:
            hidden_score = torch.ones(512).to(expected_scores["final_mlp_hidden"].device)
            
        if "final_mlp_hidden" in expected_scores:
            final_hidden_score = expected_scores["final_mlp_hidden"]  # (1024,)

            final_input_score = torch.cat([
                hidden_score,
                hidden_score,
                hidden_score,
                hidden_score,
            ])  # (4H,)

            num_parameters += torch.sum(
                torch.outer(final_input_score, final_hidden_score)
            )

            num_parameters += torch.sum(final_hidden_score) * 3
        else:
            num_parameters += self.params_finalmlp_layer
        return num_parameters
    
    def get_target_sparsity(self, pruned_steps: int, full_sparsity: float = None):
        target_sparsity = full_sparsity
        if getattr(self, "lagrangian_warmup_steps", 0) > 0:
            target_sparsity = (target_sparsity - self.start_sparsity) * min(1, pruned_steps / self.lagrangian_warmup_steps) + self.start_sparsity
        return target_sparsity


    def lagrangian_regularization(self, pruned_steps: int):
        def _lag_loss(expected_sparsity: torch.tensor, target_sparsity: float, lambda_1: torch.tensor, lambda_2: torch.tensor):
            lagrangian_loss = lambda_1 * (expected_sparsity - target_sparsity) + lambda_2 * (expected_sparsity - target_sparsity) ** 2 
            lagrangian_loss = lagrangian_loss.mean()
            return lagrangian_loss

        target_sparsity = self.get_target_sparsity(pruned_steps, self.target_sparsity)            
        expected_scores, expected_sparsitys = self.calculate_expected_score_sparsity()
        expected_size = self.get_expected_num_params(expected_scores) #! calculate \bar s
        expected_sparsity = 1 - expected_size / self.prunable_model_size
        
        return_v = {}
        if self.target_model_info is None:
            lagrangian_loss = _lag_loss(expected_sparsity, target_sparsity, self.lambdas["lambda_1"], self.lambdas["lambda_2"])
            return_v = {"expected_sparsity": expected_sparsity.item(), "target_sparsity": target_sparsity}
            for key in expected_sparsitys:
                return_v[f"expected_{key}_sparsity"] = expected_sparsitys[key].mean().item()
        else:
            lagrangian_loss = 0
            return_v = {}
            for pruning_module in self.pruning_modules:
                ts = self.get_target_sparsity(pruned_steps, self.masks[pruning_module].target_sparsity)
                expected_ts = expected_sparsitys[pruning_module] 
                lagrangian_loss += _lag_loss(expected_ts, ts, self.lambdas[f"lambda_1_{pruning_module}"], self.lambdas[f"lambda_2_{pruning_module}"])
                expected_ts = expected_ts.mean().item()
                return_v.update({"expected_{}_sparsity".format(pruning_module): expected_ts, "target_{}_sparsity".format(pruning_module): ts})
            return_v["expected_sparsity"] = expected_sparsity.item()
            return_v["target_sparsity"] = target_sparsity


        # return_v might not matter
        return lagrangian_loss, expected_sparsity, target_sparsity
 
    def get_z_from_zs(self, zs):
        numpified_zs = {} 
        for type in self.masks:
            
            z = zs.get(f"{type}_z", np.ones(self.masks[type].get_size()))
            if torch.is_tensor(z): 
                z = z.squeeze().detach().cpu().numpy() > 0
            numpified_zs[type] = z
        return numpified_zs
    
    def calculate_model_size_LLM(self, zs):
        numpified_zs = self.get_z_from_zs(zs)
        remaining_model_size=0
        results = {}
        hidden_z = numpified_zs.get("hidden", np.ones(self.hidden_size))
        if "bert" in self.config._name_or_path.lower() or 'llama' in self.config._name_or_path.lower():
            intermediate_z = numpified_zs.get("intermediate",np.ones((self.num_layers, self.intermediate_size)))
            mlp_z = numpified_zs.get("mlp",np.ones(self.num_layers)).reshape(-1, 1)
            head_z = numpified_zs.get("head",np.ones((self.num_layers, self.num_attention_heads)))

            head_layer_z = numpified_zs.get("head_layer",np.ones((self.num_layers,))).reshape(-1, 1) 
            
        
       


            remaining_hidden_dims = hidden_z.sum().item()
            remaining_intermediate_nums = intermediate_z.reshape(self.num_layers, self.intermediate_size).sum(-1).tolist()
            remaining_head_nums = head_z.reshape(self.num_layers, self.num_attention_heads).sum(-1).tolist()
            


            head_nums = np.outer((head_z * head_layer_z).reshape(-1), hidden_z).sum().item()

            """
            hidden_z: (2048,)
            intermediate_z:(24, 5504)
            mlp_z : (24, 1)
            head_z: (24, 16)
            head_layer_z: (24, 1)


            """
            intermediate_nums = np.outer((intermediate_z * mlp_z).reshape(-1), hidden_z).sum().item()
            print(f"remaining_hidden_dims: {remaining_hidden_dims}\nremaining_intermediate_nums: {remaining_intermediate_nums}")

            remaining_model_size = head_nums * self.dim_per_head * 4 + intermediate_nums * self.n_matrix_mlp 
            
            results["head_layers"] = head_layer_z.reshape(-1).astype(int).tolist()
            results["mlp_layers"] = mlp_z.reshape(-1).astype(int).tolist()
            results["hidden_dims"] = remaining_hidden_dims
            results["intermediate_dims"] = remaining_intermediate_nums
            results["head_nums"] = remaining_head_nums
        
        
        mlp_final_hidden = numpified_zs.get("final_mlp_hidden",np.ones(1024)) #should be 1024
        mlp_final_input = np.concatenate((hidden_z,hidden_z,hidden_z,hidden_z) )
        remaining_mlp_inp=mlp_final_input.sum().item()
        remaining_mlp_hidden=mlp_final_hidden.sum().item()
        
        final_mlp  = np.outer(mlp_final_input, mlp_final_hidden).sum().item()
        remaining_model_size += (final_mlp) + (remaining_mlp_hidden * self.out_params)
        pruned_model_size = self.prunable_model_size - remaining_model_size

        
        # Not multiplied with each other
        
        results["mlp_input_3072"] = remaining_mlp_inp
        results["mlp_input_1024"] = remaining_mlp_hidden
        results["pruned_params"] = pruned_model_size
        results["remaining_params"] = remaining_model_size
        results["pruned_model_sparsity"] = pruned_model_size / (self.prunable_model_size)
       

        return results

        
    def forward(self, calculate_lagrangian: bool = False, pruned_steps: int = 0, training=False):
        self.constrain_parameters()
        if calculate_lagrangian:
            return self.lagrangian_regularization(pruned_steps)
        
        zs = {f"{pruning_module}_z": [] for pruning_module in self.pruning_modules}
        
        if "layer" in self.pruning_modules:
            zs.pop("layer_z")
            zs["mlp_z"] = []
            zs["head_layer_z"] = []
  
        if self.training:
            for pruning_module in self.pruning_modules:
                mask = self.masks[pruning_module]
                z = mask.sample_z()
                zs[f"{pruning_module}_z"] = z
        else: # removed layerwise! 
            with torch.no_grad():
                for pruning_module in self.pruning_modules:
                    mask = self.masks[pruning_module]
                    z = mask.deterministic_z()
                    zs[f"{pruning_module}_z"] = z
        if "layer_z" in zs:
            zs["mlp_z"] = zs.pop("layer_z")
            zs["head_layer_z"] = zs["mlp_z"]
        return zs 

