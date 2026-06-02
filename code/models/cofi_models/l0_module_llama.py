# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import pdb

from re import L
import torch
import math
import numpy as np
import torch.nn.functional as F
from torch.nn.modules import Module
from torch.nn.parameter import Parameter
from torch.autograd import Variable
from transformers.utils import logging
import os
limit_a, limit_b, epsilon = -.1, 1.1, 1e-6
logger = logging.get_logger(__name__)

class L0Module_LLAMA(Module):
    def __init__(self,
                 config, 
                 args=None,
                 model_name=None,
                 droprate_init=0.5,
                 sparsity_scheduler="linear",
                 temperature=2./3.,
                 lagrangian_warmup=0,
                 start_sparsity=0.0,
                 target_sparsity=0.0,
                 full_model_size=None,
                 pruning_type="structured_heads+structured_mlp+hidden+layer",
                 magical_number=0.8, # from Wang et al. 2020
                 ):
        super(L0Module_LLAMA, self).__init__()
        self.final_mlp_hidden = 1024
        self.out_params = 3
        self.all_types = ["hidden_z", "intermediate_z", "mlp_z", "head_layer_z", "head_z", 'final_mlp_hidden_z']
        pruning_type = pruning_type.split("+")
        self.pruning_type = pruning_type
        assert ("mlp_layer" in pruning_type and "head_layer" in pruning_type and "layer" not in pruning_type) or \
            ("mlp_layer" not in pruning_type and "head_layer" not in pruning_type and "layer" in pruning_type) or \
                ("mlp_layer" not in pruning_type and "head_layer" not in pruning_type and "layer" not in pruning_type)
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size #hidden_size*4#config.ffn_dim#intermediate_size 
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.mlp_num_per_layer = 1
        self.dim_per_head = self.hidden_size // self.num_attention_heads #changed this from num attn heads to num kv heads 
        self.num_hidden_layers = config.num_hidden_layers
        
        self.vocab_size = config.vocab_size
       
        #modified
        kv_dim = self.num_key_value_heads * self.dim_per_head
        self.params_per_head_layer = (
            self.hidden_size * self.hidden_size      # q_proj
            + self.hidden_size * kv_dim             # k_proj
            + self.hidden_size * kv_dim             # v_proj
            + self.hidden_size * self.hidden_size    # o_proj
        )
        
        self.params_per_mlp_layer = self.hidden_size * self.intermediate_size * 3
        
        
        self.params_per_head =  self.params_per_head_layer // self.num_key_value_heads
        self.params_per_intermediate_dim = self.params_per_mlp_layer // self.intermediate_size
        
        # we ignore the parameters in normalization layers (it takes a very small amount)
        #self.full_model_size = (self.params_per_head_layer + self.params_per_mlp_layer) * self.num_hidden_layers
        self.prunable_model_size = 0 
        

        self.temperature = temperature
        self.droprate_init = droprate_init if droprate_init != 0. else 0.5
       
        
        # origin exp 0: mean=-10, layer init=0, close (layer_gate_init_open=False, layer_gate_open_0=False)
        self.types = []
        self.z_logas = {}
        self.parameters_per_dim = {}
        self.sizes = {}
        self.shapes = {}

        self.hidden_loga = None
        self.hidden_type = None

        #MLP
        self.params_finalmlp_layer = (self.hidden_size * 4 * self.final_mlp_hidden) + (self.final_mlp_hidden* self.out_params)  #weights &bias
        self.params_per_hidden_dim_final_mlp = self.params_finalmlp_layer // self.final_mlp_hidden
        
        types = self.pruning_type
        for type in types:
            self.initialize_one_module(type)
        self.shapes['mlp'] = [self.num_hidden_layers]
        self.shapes['head_layer'] = [self.num_hidden_layers]

        self.magical_number = magical_number

        self.lambda_1 = torch.nn.Parameter(torch.tensor(0.0))
        self.lambda_2 = torch.nn.Parameter(torch.tensor(0.0))

        self.lagrangian_warmup = lagrangian_warmup
        self.start_sparsity = start_sparsity
        self.target_sparsity = target_sparsity

     
        self.sparsity_scheduler = sparsity_scheduler
        
        

      

        logger.info("********** Initializing L0 Module **********") 
        for type in self.types:
            logger.info(f"***** {type} *****")
            logger.info(f"z.shape: {self.z_logas[type].shape}")
            logger.info(f"size: {self.sizes[type]}")
        logger.info(f"prunable model size: {self.prunable_model_size}")
        
        

    def set_lagrangian_warmup_steps(self, lagrangian_warmup):
        self.lagrangian_warmup = lagrangian_warmup

    def initialize_one_module(self, module_name):
        if module_name == "structured_mlp":
            self.initialize_structured_mlp()
        elif module_name == "structured_heads":
            self.initialize_structured_head()
        elif module_name == "hidden":
            self.initialize_hidden()
        elif module_name == "head_layer":
            self.initialized_layer_structured_heads()
        elif module_name == "layer":
            self.initialize_whole_mlp()
        elif module_name == 'final_mlp_hidden':
          
            self.initialize_final_hidden_layer_mlp()
            
    '''
    TYPE head
    parameters_per_dim: 2359296
    sizes: 4
    shapes: [22, 1, 4, 1, 1]
    TYPE intermediate
    parameters_per_dim: 6144
    sizes: 5632
    shapes: [22, 1, 1, 5632]
    TYPE hidden
    parameters_per_dim: 21504
    sizes: 2048
    shapes: [2048]
    TYPE mlp
    parameters_per_dim: 34603008
    sizes: 1
    shapes: [22]
    TYPE final_mlp_hidden
    parameters_per_dim: 8195
    sizes: 1024
    shapes: [1024]
    
    '''
    def add_one_module(self, z_loga, type, parameter_per_dim, size, shape): #! init the z_logas
        self.types.append(type)
        self.z_logas[type] = z_loga
        self.parameters_per_dim[type] = parameter_per_dim
        self.sizes[type] = size
        self.shapes[type] = shape
        print(f"TYPE {type}\nparameters_per_dim: {parameter_per_dim}\nsizes: {size}\nshapes: {shape}")

    def initialize_parameters(self, size, num_layer=None):
        if num_layer is not None:
            return Parameter(torch.Tensor(num_layer, size))
        else:
            return Parameter(torch.Tensor(size))

    #MODIFIED
    def initialize_hidden(self):
        self.hidden_loga = self.initialize_parameters(self.hidden_size)
        self.add_one_module(self.hidden_loga, type="hidden", 
                            parameter_per_dim= (
                                self.hidden_size      # q input
                                + 256                 # k input
                                + 256                 # v input
                                + self.hidden_size    # o output
                                + 3 * self.intermediate_size
                            ),
                            size=self.hidden_size, shape=[self.hidden_size])
        self.reset_loga(self.hidden_loga, mean=10)
        self.input_layer_mlp_loga = Parameter(torch.cat([self.hidden_loga for i in range(4)]))
        self.input_layer_mlp_loga.requires_grad=False  
        logger.info(f"Initialized hidden loga! Prunable_model_size = {self.prunable_model_size}")

    def initialize_structured_head(self, add_prunable_model_size=True):
        self.head_loga = self.initialize_parameters(self.num_key_value_heads, self.num_hidden_layers)
        self.reset_loga(self.head_loga, mean=10)
        self.add_one_module(self.head_loga, type="head", 
                            parameter_per_dim=self.params_per_head, size=self.num_key_value_heads,
                            shape=[self.num_hidden_layers, 1, self.num_key_value_heads, 1, 1])
        if add_prunable_model_size:
            self.prunable_model_size += self.params_per_head * self.num_hidden_layers * self.num_key_value_heads
       
        logger.info(f"Initialized structured heads! Prunable_model_size = {self.prunable_model_size}")


    def initialized_layer_structured_heads(self):
        n_layer = self.num_hidden_layers
        self.headlayer_loga = self.initialize_parameters(n_layer)
        self.reset_loga(self.headlayer_loga, mean=10)
        self.add_one_module(self.headlayer_loga, type="head_layer", 
                            parameter_per_dim=self.params_per_head * self.num_key_value_heads, size=1,
                            shape=[n_layer])
        logger.info(f"Initialized layerwise structured heads! Prunable_model_size = {self.prunable_model_size}")

    def initialize_structured_mlp(self):
        self.int_loga = self.initialize_parameters(self.intermediate_size, self.num_hidden_layers)

        self.add_one_module(self.int_loga, type="intermediate", 
                            parameter_per_dim=self.params_per_intermediate_dim, size=self.intermediate_size,
                            shape=[self.num_hidden_layers, 1, 1, self.intermediate_size])
        self.prunable_model_size += self.params_per_mlp_layer * self.num_hidden_layers
       
        self.reset_loga(self.int_loga)
        logger.info(f"Initialized structured mlp! Prunable_model_size = {self.prunable_model_size}")

    def initialize_whole_mlp(self):
        n_layer = self.num_hidden_layers
        self.intlayer_loga = self.initialize_parameters(n_layer)
        self.add_one_module(self.intlayer_loga, type="mlp", 
                            parameter_per_dim=self.params_per_mlp_layer, size=self.mlp_num_per_layer,
                            shape=[n_layer])
        self.reset_loga(self.intlayer_loga, mean=10)
        logger.info(f"Initialized whole mlps! Prunable_model_size = {self.prunable_model_size}")

        
    def initialize_final_hidden_layer_mlp(self): #also add final_layer_hid_mlp to self.types
      
        self.hidden_layer_mlp_loga = self.initialize_parameters(self.final_mlp_hidden) #3072,1024 this will prune the 1024
        self.reset_loga(self.hidden_layer_mlp_loga, mean=10)
        self.add_one_module(self.hidden_layer_mlp_loga, type="final_mlp_hidden", 
                            parameter_per_dim=self.params_per_hidden_dim_final_mlp, size=self.final_mlp_hidden,
                            shape=[self.final_mlp_hidden])
       
        logger.info(f"Initialized final layer hidden mlp! Prunable_model_size = {self.prunable_model_size}")
    
    
    def reset_loga(self, tensor, mean=None):
        if mean is None:
            mean = math.log(1 - self.droprate_init) - math.log(self.droprate_init)
        tensor.data.normal_(mean, 1e-2)

    def reset_qz_logas(self):
        for key in self.z_logas:
            if key in ["head_layer", "mlp", "head"]:
                self.reset_loga(self.z_logas[key], 10)
            else:
                self.reset_loga(self.z_logas[key])

    def constrain_parameters(self):
        def _constrain(tensor):
            tensor.data.clamp_(min=math.log(1e-2), max=math.log(1e2))
        for key in self.z_logas:
            _constrain(self.z_logas[key])

    def cdf_qz(self, x, loga):
        """Implements the CDF of the 'stretched' concrete distribution"""
        xn = (x - limit_a) / (limit_b - limit_a)
        logits = math.log(xn) - math.log(1 - xn)
        return torch.sigmoid(logits * self.temperature - loga).clamp(min=epsilon, max=1 - epsilon)

    def quantile_concrete(self, x, loga):
        y = torch.sigmoid((torch.log(x) - torch.log(1 - x) + loga) / self.temperature)
        return y * (limit_b - limit_a) + limit_a

    def get_num_parameters_for_one(self, loga, parameter_size):
        return torch.sum(1 - self.cdf_qz(0, loga)) * parameter_size

    def transform_scores_for_head(self):
        assert "head" in self.types

        if "head_layer" in self.types:
            all_head_score = 1 - self.cdf_qz(0, self.headlayer_loga)
        else:
            all_head_score = None
        head_score = 1 - self.cdf_qz(0, self.head_loga) # 32 * 32
       
        if all_head_score is not None:
            all_head_score = all_head_score.view(-1, 1) # 32 * 1
        head_score = head_score.unsqueeze(-1)   # 32 * 32 * 1
       
        return all_head_score, head_score

    def transform_scores_for_mlp(self):
        assert "intermediate" in self.types

        
        if "mlp" in self.types:
            all_int_score = 1 - self.cdf_qz(0, self.intlayer_loga)
        else:
            all_int_score = None
        int_score = 1 - self.cdf_qz(0, self.int_loga)  # 12 * 3072

        if all_int_score is not None:
            all_int_score = all_int_score.view(-1, 1) # 32 * 1
       
        return all_int_score, int_score

    def get_num_parameters_for_mlp(self):
        raise NotImplementedError
        if "layer" in self.types:
            intlayer_score = 1 - self.cdf_qz(0, self.layer_loga)
        elif "mlp_layer" in self.types:
            intlayer_score = 1 - self.cdf_qz(0, self.intlayer_loga)
        else:
            intlayer_score = None
        int_score = 1 - self.cdf_qz(0, self.int_loga) # 32 * 11008
        intlayer_score = intlayer_score.unsqueeze(-1)

        num_parameters = torch.sum(intlayer_score * int_score) * self.parameters_per_dim["intermediate"]
        return num_parameters

    def get_num_parameters_and_constraint_for_hidden(self): #! calculate the current parsity
        num_parameters = 0
       
        # 12 * 1 * 1
        # 12 * 12 * 1
        all_head_score, head_score = self.transform_scores_for_head()
        hidden_score = 1 - self.cdf_qz(0, self.hidden_loga) # 768

        if all_head_score is not None:
            head_score = (all_head_score * head_score).reshape(-1)
        else:
            head_score = head_score.reshape(-1)
        num_parameters += \
            torch.sum(torch.outer(hidden_score, head_score)) * self.parameters_per_dim["head"] / self.hidden_size

        intlayer_score = 1 - self.cdf_qz(0, self.intlayer_loga)  # 12
        int_score = 1 - self.cdf_qz(0, self.int_loga)  # 12 * 3072
        intlayer_score = intlayer_score.unsqueeze(-1)

        int_score = (intlayer_score * int_score).reshape(-1)
        num_parameters += torch.sum(torch.outer(hidden_score, int_score)) * 2
        return num_parameters





    def get_num_parameters_and_constraint(self):
        num_parameters = 0

        all_head_score, head_score = self.transform_scores_for_head()
        
        head_score = (1 - all_head_score * (1 - head_score))
        num_parameters += torch.sum(head_score) * self.parameters_per_dim["head"]

        all_int_score, int_score = self.transform_scores_for_mlp()

        int_score = (1 - all_int_score * (1 - int_score))
        num_parameters += torch.sum(int_score) * self.parameters_per_dim["intermediate"]
        return num_parameters


    def get_target_sparsity(self, pruned_steps: int):
        target_sparsity = self.target_sparsity
        if getattr(self, "lagrangian_warmup_steps", 0) > 0:
            target_sparsity = (target_sparsity - self.start_sparsity) * min(1, pruned_steps / self.lagrangian_warmup_steps) + self.start_sparsity
        return target_sparsity

    def get_num_parameters_and_constraint_for_final_MLP(self):
        """
        Calculate expected classifier parameters using z-scores ONLY.
        """
        inp_layer_score = 1 - self.cdf_qz(0, self.input_layer_mlp_loga)  # (8192,)
        hid_layer_score = 1 - self.cdf_qz(0, self.hidden_layer_mlp_loga)  # (1024,)

        # Linear 1: (inp, hid) with bias
        expected_params_1 = torch.sum(torch.outer(inp_layer_score, hid_layer_score))
        #expected_bias_1 = torch.sum(hid_layer_score)

        # Linear 2: (hid, 3) with bias
        expected_params_2 = torch.sum(torch.outer(hid_layer_score, torch.ones(3, device=hid_layer_score.device)))
        #expected_bias_2 = 3  # Always 3 (num classes)

        # BatchNorm1d: weight + bias for inp dimension
        #bn_params = 2 * torch.sum(inp_layer_score)

        num_parameters = (expected_params_1 + 
                         expected_params_2 )

        return num_parameters
    def lagrangian_regularization(self, pruned_steps):
        target_sparsity = self.target_sparsity
        if "hidden" in self.types:
            expected_size = self.get_num_parameters_and_constraint_for_hidden() + self.get_num_parameters_and_constraint_for_final_MLP() #! calculate \bar s
        else:
            expected_size = self.get_num_parameters_and_constraint() #! calculate \bar s
        expected_sparsity = 1 - (expected_size) / (self.prunable_model_size)
   
        del expected_size
        if self.lagrangian_warmup > 0:
            target_sparsity = self.get_target_sparsity(pruned_steps)
        lagrangian_loss = ( #! see appendix
                self.lambda_1 * (expected_sparsity - target_sparsity)
                + self.lambda_2 * (expected_sparsity - target_sparsity) ** 2 #! where is the lambda 1 and lambda 2 from
        )
        return lagrangian_loss, expected_sparsity, target_sparsity

    def get_eps(self, size):
        """Uniform random numbers for the concrete distribution"""
        eps = torch.FloatTensor(size).uniform_(epsilon, 1-epsilon)
        eps = Variable(eps)
        return eps

    # during training
    def _sample_z(self, loga):
        eps = self.get_eps(torch.FloatTensor(*loga.shape)).to(loga.device)
        z = self.quantile_concrete(eps, loga)
        z = F.hardtanh(z, min_val=0, max_val=1)
        return z

    # during inference
    def _deterministic_z(self, size, loga):
        # Following https://github.com/asappresearch/flop/blob/e80e47155de83abbe7d90190e00d30bfb85c18d5/flop/hardconcrete.py#L8 line 103
        
        expected_num_nonzeros = torch.sum(1 - self.cdf_qz(0, loga))
        
        expected_num_zeros = size - expected_num_nonzeros.item()
        try:
            num_zeros = round(expected_num_zeros)
        except:
            pdb.set_trace()
        if len(loga.shape) == 0: # layerz
            if num_zeros > 0:
                return torch.tensor(0).to(loga.device)
            else:
                return torch.tensor(1).to(loga.device)
            
        soft_mask = torch.sigmoid(loga / self.temperature * self.magical_number)
        if num_zeros > 0:
            if soft_mask.ndim == 0:
                soft_mask = torch.tensor(0).to(loga.device)
            else:
                _, indices = torch.topk(soft_mask, k=num_zeros, largest=False)
                soft_mask[indices] = 0.
        return soft_mask

    def get_z_from_zs(self, zs):
        numpified_zs = {} 
        for type in self.all_types:
            name = type[:-2]
            z = zs.get(type, np.ones(self.shapes[name]))
            if torch.is_tensor(z): 
                z = z.squeeze().detach().cpu().numpy() > 0
            numpified_zs[name] = z
        return numpified_zs

    def calculate_model_size(self, zs):
        numpified_zs = self.get_z_from_zs(zs)
        hidden_z = numpified_zs["hidden"]
        intermediate_z = numpified_zs["intermediate"]
        head_z = numpified_zs["head"]
        head_layer_z = numpified_zs["head_layer"].reshape(-1)
        mlp_z = numpified_zs["mlp"].reshape(-1)
        print("SHPAS ", 'head ', head_z.shape, 'head layer ', head_layer_z.shape  )

        remaining_hidden_dims = hidden_z.sum().item()
        remaining_intermediate_nums = intermediate_z.reshape(self.num_hidden_layers, self.intermediate_size).sum(-1)
        remaining_intermediate_nums = (self.intermediate_size - (self.intermediate_size - remaining_intermediate_nums) * mlp_z).tolist()

        remaining_head_nums = head_z.reshape(self.num_hidden_layers, self.num_key_value_heads).sum(-1)
        remaining_head_nums = (self.num_key_value_heads - (self.num_key_value_heads - remaining_head_nums) * head_layer_z).tolist()

        head_nums = np.outer((1 - head_layer_z.reshape(-1, 1) * (1 - head_z)).reshape(-1), hidden_z).sum().item() # 
        intermediate_nums = np.outer((1 - mlp_z.reshape(-1, 1) * (1 - intermediate_z)).reshape(-1), hidden_z).sum().item()
        q_per_kv = self.num_attention_heads // self.num_key_value_heads

        attn_per_hidden_per_kv = (
            2 * q_per_kv * self.dim_per_head  # Q + O
            + 2 * self.dim_per_head           # K + V
        )

        remaining_model_size = head_nums * attn_per_hidden_per_kv + intermediate_nums * 3

            
            
        #MLP
        mlp_final_hidden = numpified_zs.get("final_mlp_hidden", np.ones(1024))
        mlp_final_input = np.concatenate((hidden_z,hidden_z,hidden_z,hidden_z))
        remaining_mlp_inp = mlp_final_input.sum().item()
        remaining_mlp_hidden = mlp_final_hidden.sum().item()
        final = (remaining_mlp_inp * remaining_mlp_hidden)+ (remaining_mlp_hidden*3)
        remaining_model_size += final
        pruned_model_size = self.prunable_model_size - remaining_model_size

        
        results = {}
        # Not multiplied with each other
        if "layer" in self.pruning_type:
            results["head_layer"] = head_layer_z.reshape(-1).astype(int).tolist()
        if "mlp_layer" in self.pruning_type:
            results["mlp_layers"] = mlp_z.reshape(-1).astype(int).tolist()
            results["head_layers"] = head_layer_z.reshape(-1).astype(int).tolist()
        results["hidden_dims"] = remaining_hidden_dims
        results["intermediate_dims"] = remaining_intermediate_nums
        results["head_nums"] = remaining_head_nums
        results["final mlp "] = remaining_mlp_hidden
        results["pruned_params"] = pruned_model_size
        results["remaining_params"] = remaining_model_size
        results["pruned_model_sparsity"] = (pruned_model_size) / (self.prunable_model_size)
        
        if "layer" in self.pruning_type:
            logger.info(f"remaining_layers: {head_layer_z}")
        if "mlp_layer" in self.pruning_type:
            logger.info(f"remaining_mlp_layers: {mlp_z}")
            logger.info(f"remaining_head_layers: {head_layer_z}")
        logger.info(f"remaining_hidden_dims: {remaining_hidden_dims}")
        logger.info(f"remaining_intermediate_nums: {remaining_intermediate_nums}")
        logger.info(f"remaining_head_nums: {remaining_head_nums}")
        logger.info(f"pruned_model_size: {pruned_model_size}")
        logger.info(f"remaining_model_size: {remaining_model_size}")

        return results

    def forward(self, training=True,):
        zs = {f"{type}_z": [] for type in self.types}

        if training:
            for i, type in enumerate(self.types):
                loga = self.z_logas[type]
                z = self._sample_z(loga)
                zs[f"{type}_z"] = z.reshape(self.shapes[type])
                assert torch.isnan(zs[f"{type}_z"]).sum().item() == 0, f'Line 624, zs for {type} is nan'
                
        else:
            for i, types in enumerate(self.types):
                
                if types != "hidden" and types != 'final_mlp_hidden' and types != 'final_mlp_inp':
                    loga_all_layers = self.z_logas[types]
                    for layer in range(len(loga_all_layers)):
                        loga = loga_all_layers[layer]
                        size = self.sizes[types]
                   
                        z = self._deterministic_z(size, loga).float()
                        
                        zs[f"{types}_z"].append(z.reshape(self.shapes[types][1:]))
                else:
                    if types == 'hidden':
                        z = self._deterministic_z(self.sizes[types], self.hidden_loga)
                    elif types== 'final_mlp_hidden':
                        z = self._deterministic_z(self.sizes[types], self.hidden_layer_mlp_loga)
                        
                    zs[f"{types}_z"] = z
            for types in zs:
                assert types != "layer_z"
                if types != "hidden_z" and types != 'final_mlp_hidden_z' and types != 'final_mlp_inp_z': #used to be if type != hidden_z
                    zs[types] = torch.stack(zs[types])
            print(zs['mlp_z'])
        if "layer_z" in zs:
            zs.pop("layer_z")
        
        return zs 

if __name__ == "__main__":
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained("bert-base-uncased")
    l0_module = L0Module(config, lagrangian_warmup=200, target_sparsity=0.5)