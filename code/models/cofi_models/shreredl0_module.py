#old
# 
import math
import sys

import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F

from argparse import Namespace as NS
from typing import Any, List

import numpy as np
from transformers.utils import logging

logger = logging.get_logger(__name__)
limit_a, limit_b, epsilon = -.1, 1.1, 1e-6
'''
head:                                                                                                              | 34/3747 [00:14<25:03,  2.47it/s]
orig shape: torch.Size([22, 4])
output shape: [22, 1, 4, 1]
result shape torch.Size([22, 4])
For head, Score from cdf shape is torch.Size([22, 4])

intermediate
orig shape: torch.Size([22, 5632])
output shape: [22, 1, 1, 5632]
result shape torch.Size([22, 5632])
For intermediate, Score from cdf shape is torch.Size([22, 5632])
hidden
orig shape: torch.Size([2048])
output shape: [2048]
result shape torch.Size([2048])
For hidden, Score from cdf shape is torch.Size([2048])
mlp
orig shape: torch.Size([22])
output shape: [22]
result shape torch.Size([22])
For mlp, Score from cdf shape is torch.Size([22])
head_layer
orig shape: torch.Size([22])
output shape: [22]
result shape torch.Size([22])
For head_layer, Score from cdf shape is torch.Size([22])
final_mlp_hidden
orig shape: torch.Size([1024])
output shape: [1024]
result shape torch.Size([1024])
For final_mlp_hidden, Score from cdf shape is torch.Size([1024])
mlp_score SHAPE: torch.Size([22, 1]) intermediate_score SHAPE: torch.Size([22, 5632])

'''
class Mask(nn.Module):
    def __init__(self, 
                 name: str,
                 mask_shape: List, 
                 num_params_per_mask: int, 
                 mask_output_shape: List, 
                 
                 device: str,
                 target_mask_size: int) -> None:
        super().__init__()
        self.name = name
        self.num_params_per_mask = num_params_per_mask
        self.mask_output_shape = mask_output_shape
       

        self.droprate_init = 0.5
        self.temperature = 2./3.
        self.magical_number = 0.8
        self.device = device
        
        self.z_loga = self.initialize_mask(mask_shape) 
        self.mask_size = self.z_loga.shape[-1] # the full size of each unit
        self.target_mask_size = target_mask_size
        
        
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
        
        result = torch.sigmoid(logits * self.temperature - z_loga).clamp(min=epsilon, max=1 - epsilon)
        #print(f"{self.name}\norig shape: {z_loga.shape}\noutput shape: {self.mask_output_shape}\nresult shape {result.shape}")
        return result
    
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
        if self.target_mask_size is not None:
            expected_score = 1 - self.cdf_qz(z_loga)
            expected_num_nonzeros = expected_score.sum()
            expected_num_zeros = self.target_mask_size - expected_num_nonzeros.item()
        else:
            assert False, "target mask size not defined"
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
        #print(f"For {self.name}, Score from cdf shape is {score.shape}")
        sparsity = 1 - score.sum(-1) / self.target_mask_size
        return score, sparsity
    
 
class L0Module_LLAMA(nn.Module):
    def __init__(self,
                 config=None, 
                 model_name=None,
                 droprate_init=0.5,
                 temperature=1./3.,
                 lagrangian_warmup=0,
                 start_sparsity=0.0,
                 target_sparsity=0.0,
                 args='out',
                 full_model_size=None,
                 pruning_type="structured_heads+structured_mlp+hidden+layer+final_mlp_hidden",
                 magical_number=0.8, # from Wang et al. 2020
                 device='cuda'
                 ):
        super(L0Module_LLAMA, self).__init__()
        

        self.final_mlp_hidden = 1024
        self.out_params = 3
        
        # base and target model info
        n_matrix_mlp = 3
        #l0_module_cfg = cfg.l0_module
   
        #l0 module config attrs
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
        
        self.params_finalmlp_layer = (self.hidden_size * 4 * self.final_mlp_hidden) + (self.final_mlp_hidden* self.out_params)  
        self.params_per_hidden_dim_final_mlp = self.params_finalmlp_layer // self.final_mlp_hidden
        
        # we ignore the parameters in normalization layers (it takes a very small amount)
        self.full_model_size = (self.params_per_head_layer + self.params_per_mlp_layer) * self.num_hidden_layers + self.params_finalmlp_layer
        self.prunable_model_size = 0 
    
        
        # l0 config
        self.pruning_modules = pruning_type.split("+")   
        self.start_sparsity = start_sparsity
        self.lagrangian_warmup_steps = lagrangian_warmup
        self.device = device
        
        # l0 params
   
        self.lambda_1 = torch.nn.Parameter(torch.tensor(0.0))
        self.lambda_2 = torch.nn.Parameter(torch.tensor(0.0))
        self.masks = {}
        for pruning_module in self.pruning_modules:
            self.initialize_one_module(pruning_module)
        self.masks = torch.nn.ModuleDict(self.masks)
        
        # config after initialization
        self.prunable_model_size = self.calculate_prunable_model_size()
        #can comment this out bc else is equiv to setting the arget sparsity raw
        '''if target_model_cfg is not None:
            self.prunable_target_model_size = self.calculate_prunable_model_size()
            self.target_sparsity = 1 - self.prunable_target_model_size / self.prunable_model_size
        else:
            self.target_sparsity = l0_module_cfg.target_sparsity'''
        self.target_sparsity = target_sparsity
        print("********** Initializing L0 Module **********") 
        for pruning_module in self.masks:
            print(f"***** {pruning_module} *****")
            print(f"z.shape", self.masks[pruning_module].z_loga.shape)
            print(f"size", self.masks[pruning_module].mask_size)
        print(f"prunable model size: {self.prunable_model_size}")
        
    
    '''def set_model_info(self, cfg, n_matrix_mlp=3):
        ns = NS() 
        ns.hidden_size = cfg.d_model
        ns.intermediate_size = cfg.intermediate_size
        ns.num_attention_heads = cfg.n_heads
        ns.mlp_num_per_layer = 1
        ns.dim_per_head = ns.hidden_size // ns.num_attention_heads 
        ns.num_layers = cfg.n_layers
        ns.vocab_size = cfg.vocab_size

        ns.params_per_head_layer = ns.hidden_size * ns.hidden_size * 4
        ns.params_per_head =  ns.params_per_head_layer // ns.num_attention_heads
        ns.params_per_mlp_layer = ns.hidden_size * ns.intermediate_size * n_matrix_mlp
        ns.params_per_intermediate_dim = ns.params_per_mlp_layer // ns.intermediate_size

        ns.full_model_size = (ns.params_per_head_layer + ns.params_per_mlp_layer) * ns.num_layers
        return ns'''
        
    def set_lagrangian_warmup_steps(self, lagrangian_warmup):
        self.lagrangian_warmup_steps = lagrangian_warmup
        
    def calculate_prunable_model_size(self):
        prunable_mlp_size = self.params_per_mlp_layer * self.num_hidden_layers
        prunable_head_layer_size = self.params_per_head_layer * self.num_hidden_layers
        prunable_model_size = 0
       
        if "structured_heads" in self.pruning_modules:
            prunable_model_size += prunable_head_layer_size
        if "structured_mlp" in self.pruning_modules:
            prunable_model_size += prunable_mlp_size
        if 'final_mlp_hidden' in self.pruning_modules :
            prunable_model_size += self.params_finalmlp_layer
        return prunable_model_size
        
    def initialize_one_module(self, module_name: str):
        if module_name == "structured_mlp":
            self.initialize_structured_mlp()
        elif module_name == "structured_heads":
            
            self.initialize_structured_head()
        elif module_name == "hidden":
            self.initialize_hidden()
        elif module_name == "layer":
            self.initialize_whole_mlp()
            self.initialized_layer_structured_heads()
        elif module_name == 'final_mlp_hidden':
            print("final ml")
            self.initialize_final_hidden_layer_mlp()
            
    def initialize_hidden(self):
        mask_shape = [self.hidden_size]
        num_params_per_mask= (
                                self.hidden_size      # q input
                                + 256                 # k input
                                + 256                 # v input
                                + self.hidden_size    # o output
                                + 3 * self.intermediate_size
                            )
        
        target_mask_size = self.hidden_size
           
        
        hidden_mask = Mask(name="hidden",
                           mask_shape=mask_shape,
                           num_params_per_mask=num_params_per_mask,
                           mask_output_shape=[self.hidden_size],
                           
                           target_mask_size=target_mask_size,
                           device=self.device)
        self.masks["hidden"] = hidden_mask

    def initialize_structured_head(self):
        mask_shape = [self.num_hidden_layers, self.config.num_key_value_heads]
        num_params_per_mask = self.params_per_head
        mask_output_shape = [self.num_hidden_layers, 1, self.config.num_key_value_heads, 1] #was 22,1,4,1,1 in cofi code
        
    
        target_mask_size = self.config.num_key_value_heads
            
        head_mask = Mask(name="head",
                         mask_shape=mask_shape,
                         num_params_per_mask=num_params_per_mask,
                         mask_output_shape=mask_output_shape,
                         target_mask_size=target_mask_size,
                           device=self.device,)
        self.masks["head"] = head_mask 

        
        
        
    def initialize_final_hidden_layer_mlp(self): #also add final_layer_hid_mlp to self.types
      
        mask_shape = [self.final_mlp_hidden]
        num_params_per_mask=self.params_per_hidden_dim_final_mlp
        mask_output_shape = [self.final_mlp_hidden] 
        
        final_mlp_mask = Mask(name="final_mlp_hidden",
                              mask_shape=mask_shape,
                               num_params_per_mask=num_params_per_mask,
                               mask_output_shape=mask_output_shape,
                              target_mask_size=self.final_mlp_hidden,
                           device=self.device)
        
        self.masks["final_mlp_hidden"] = final_mlp_mask
       
    def initialized_layer_structured_heads(self):
        mask_shape = [self.num_hidden_layers]
        num_params_per_mask=self.params_per_head *  self.config.num_key_value_heads
        mask_output_shape = [self.num_hidden_layers] 
        target_mask_size = self.num_hidden_layers
        
        head_layer_mask = Mask(name="head_layer",
                              mask_shape=mask_shape,
                               num_params_per_mask=num_params_per_mask,
                               mask_output_shape=mask_output_shape,
                               target_mask_size=target_mask_size,
                           device=self.device)
        self.masks["head_layer"] = head_layer_mask
        
    def initialize_structured_mlp(self):
        mask_shape = [self.num_hidden_layers, self.intermediate_size]
        num_params_per_mask=self.params_per_intermediate_dim
        mask_output_shape = [self.num_hidden_layers, 1, 1, self.intermediate_size] 
        
        target_mask_size = self.intermediate_size
        
        int_mask = Mask(name="intermediate",
                        mask_shape=mask_shape,
                        num_params_per_mask=num_params_per_mask,
                        mask_output_shape=mask_output_shape,
                        target_mask_size=target_mask_size,
                           device=self.device)
        self.masks["intermediate"] = int_mask
       

    def initialize_whole_mlp(self):
        mask_shape = [self.num_hidden_layers]
        num_params_per_mask=self.params_per_mlp_layer
        mask_output_shape = [self.num_hidden_layers] 
        
        target_mask_size = self.num_hidden_layers
        mlp_mask = Mask(name="mlp",
                        mask_shape=mask_shape,
                        num_params_per_mask=num_params_per_mask,
                        mask_output_shape=mask_output_shape,
                        target_mask_size=target_mask_size,
                           device=self.device)
        self.masks["mlp"] = mlp_mask 

    
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
        head_score = expected_scores.get("head", torch.ones(22,4)) # 12 * 12

        head_layer_score = None
        if "head_layer" in expected_scores:
            head_layer_score = expected_scores.get("head_layer",  torch.ones(22,))
        if head_layer_score is not None:
            head_layer_score = head_layer_score.view(-1, 1) # 12 * 1
            assert head_layer_score.shape[0]==22 and head_layer_score.shape[1]==1, f"headlayer shpe hsould be 22,1 but is {head_layer_score.shape}"
        #print(f"HEAD SHAPE: {head_score.shape} HEAD LAYER SHAPE: {head_layer_score.shape}")
        #HEAD SHAPE: torch.Size([22, 4]) HEAD LAYER SHAPE: torch.Size([22, 1])   
        return head_layer_score, head_score.to(head_layer_score.device)

    def transform_scores_for_mlp(self, expected_scores: dict):
        mlp_score = None
        if "mlp" in expected_scores:
            mlp_score = expected_scores.get("mlp", torch.ones(22,)) # 12
        if mlp_score is not None:
            mlp_score = mlp_score.unsqueeze(-1)
        
        intermediate_score = expected_scores["intermediate"] # 12 * 3072
        #print(f"mlp_score SHAPE: {mlp_score.shape} intermediate_score SHAPE: {intermediate_score.shape}")
        return mlp_score, intermediate_score


    '''FOR. MHA NOT GQA def get_expected_num_params(self, expected_scores: dict): #! calculate the current parsity
        num_parameters = 0
       
        # 12 * 1 
        # 12 * 12
        head_layer_score, head_score = self.transform_scores_for_head(expected_scores)
        mlp_score, int_score = self.transform_scores_for_mlp(expected_scores)
        
        head_score = (head_layer_score * head_score) # 12 * 12
        int_score = (mlp_score * int_score) # 12 * 3072

                
        if "hidden" in expected_scores:
            hidden_score = expected_scores["hidden"] # 768 
            
            num_parameters += torch.sum(torch.matmul(hidden_score.reshape(1, -1, 1), qk_score.unsqueeze(1))) * 2 # 12 * 768 * 768
            num_parameters += torch.sum(torch.matmul(hidden_score.reshape(1, -1, 1), vo_score.unsqueeze(1))) * 2 # 12 * 768 * 768
            num_parameters += torch.sum(torch.matmul(hidden_score.reshape(1, -1, 1), int_score.unsqueeze(1))) * 3 # 12 * 768 * 3072
        else:
            num_parameters += torch.sum(head_score) * self.masks.head.num_params_per_mask
            num_parameters += torch.sum(int_score) * self.masks.intermediate.num_params_per_mask
        return num_parameters'''
    
    def get_expected_num_params(self, expected_scores: dict):
        """
        Expected remaining params for TinyLlama / LLaMA GQA.

        hidden_score:       (H,)
        head_score:         (L, KV)
        intermediate_score: (L, I)
        """

        num_parameters = 0.0

        head_layer_score, head_score = self.transform_scores_for_head(expected_scores)
        mlp_score, int_score = self.transform_scores_for_mlp(expected_scores)

        if head_layer_score is not None:
            head_score = head_layer_score * head_score  # (L, KV)

        if mlp_score is not None:
            int_score = mlp_score * int_score  # (L, I)

        if "hidden" in expected_scores:
            hidden_score = expected_scores["hidden"]  # (H,)

            q_per_kv = self.num_attention_heads // self.num_key_value_heads


            attn_per_hidden_per_kv = (
                2 * q_per_kv * self.dim_per_head
                + 2 * self.dim_per_head
            )


            hidden_head_pairs = torch.sum(
                torch.outer(hidden_score, head_score.reshape(-1))
            )

            num_parameters += hidden_head_pairs * attn_per_hidden_per_kv

            # =========================
            # LLaMA MLP params
            # gate, up, down => 3 matrices
            # =========================
            hidden_int_pairs = torch.sum(
                torch.outer(hidden_score, int_score.reshape(-1))
            )

            num_parameters += hidden_int_pairs * 3

            # =========================
            # final MLP classifier params
            # =========================
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

                num_parameters += torch.sum(final_hidden_score) * self.out_params

        else:
            # no hidden pruning: use normal per-mask parameter counts
            num_parameters += torch.sum(head_score) * self.masks["head"].num_params_per_mask
            num_parameters += torch.sum(int_score) * self.masks["intermediate"].num_params_per_mask

            if "final_mlp_hidden" in expected_scores:
                num_parameters += (
                    torch.sum(expected_scores["final_mlp_hidden"])
                    * self.masks["final_mlp_hidden"].num_params_per_mask
                )

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
        
        lagrangian_loss = _lag_loss(expected_sparsity, target_sparsity, self.lambda_1, self.lambda_2)
        
        return lagrangian_loss, expected_sparsity, target_sparsity
 


    def get_z_from_zs(self, zs):
        numpified_zs = {} 
        for type in self.masks:
            print(type, self.masks[type].get_size())
            
            z = zs.get(f"{type}_z", np.ones(self.masks[type].get_size()))
            if torch.is_tensor(z): 
                z = z.squeeze().detach().cpu().numpy() > 0
            numpified_zs[type] = z
        return numpified_zs
    
    def calculate_model_size(self, zs):
        numpified_zs = self.get_z_from_zs(zs)
        hidden_z = numpified_zs.get("hidden", np.ones(2048))
        intermediate_z = numpified_zs.get("intermediate", np.ones((22,5632)))
        head_z = numpified_zs.get("head", np.ones((22,4)))
        head_layer_z = numpified_zs.get("head_layer",np.ones(22,)).reshape(-1)
        mlp_z = numpified_zs.get("mlp", np.ones(22,)).reshape(-1)
            

        remaining_hidden_dims = hidden_z.sum().item()
        head_z = head_z.reshape(self.num_hidden_layers, self.num_key_value_heads)
        intermediate_z = intermediate_z.reshape(self.num_hidden_layers, self.intermediate_size)
        head_layer_z = head_layer_z.reshape(self.num_hidden_layers, 1)
        mlp_z = mlp_z.reshape(self.num_hidden_layers, 1)

        head_mask = head_z * head_layer_z
        intermediate_mask = intermediate_z * mlp_z

        remaining_head_nums = head_mask.sum(-1).tolist()
        remaining_intermediate_nums = intermediate_mask.sum(-1).tolist()

        head_nums = np.outer(head_mask.reshape(-1), hidden_z).sum().item()
        intermediate_nums = np.outer(intermediate_mask.reshape(-1), hidden_z).sum().item()
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
        if "head_layer" in self.masks:
            results["head_layer"] = head_layer_z.reshape(-1).astype(int).tolist()
        if "mlp" in self.masks:
            results["mlp"] = mlp_z.reshape(-1).astype(int).tolist()
            results["head_layer"] = head_layer_z.reshape(-1).astype(int).tolist()
        results["hidden_dims"] = remaining_hidden_dims
        results["intermediate_dims"] = remaining_intermediate_nums
        results["head_nums"] = remaining_head_nums
        results["final mlp "] = remaining_mlp_hidden
        results["pruned_params"] = pruned_model_size
        results["remaining_params"] = remaining_model_size
        results["pruned_model_sparsity"] = (pruned_model_size) / (self.prunable_model_size)
        
        if head_layer_z is not None:
            logger.info(f"remaining_layers: {head_layer_z}")
        if mlp_z is not None:
            logger.info(f"remaining_mlp_layers: {mlp_z}")
            logger.info(f"remaining_head_layers: {head_layer_z}")
        logger.info(f"remaining_hidden_dims: {remaining_hidden_dims}")
        logger.info(f"remaining_intermediate_nums: {remaining_intermediate_nums}")
        logger.info(f"remaining_head_nums: {remaining_head_nums}")
        logger.info(f"pruned_model_size: {pruned_model_size}")
        logger.info(f"remaining_model_size: {remaining_model_size}")

        return results
    def forward(self, training=True):

        
        zs = {f"{pruning_module}_z": [] for pruning_module in self.masks}
        
        
        if training:
            #dict_keys(['head', 'intermediate', 'hidden', 'mlp', 'head_layer', 'final_mlp_hidden']) 
            for pruning_module in self.masks:
                mask = self.masks[pruning_module]
                z = mask.sample_z()
                zs[f"{pruning_module}_z"] = z
        else: # removed layerwise!]
            with torch.no_grad():
                for pruning_module in self.masks:
                    
                    mask = self.masks[pruning_module]
                    z = mask.deterministic_z()
                    zs[f"{pruning_module}_z"] = z
        return zs 
    


def test_l0_module():
    from omegaconf import OmegaConf as om 
    cfg = om.load("/scratch/gpfs/mengzhou/space2/examples/examples/llm/yamls/llama/7b.yaml")
    cfg = om.load("/scratch/gpfs/mengzhou/space2/examples/examples/llm/yamls/pythia/410m.yaml")
    cfg.model.l0_module.pruning_modules = ["layer", "head", "intermediate", "hidden"]
    
    l0_module = L0Module(cfg.model, "cpu")
    
    # test run_through
    print("\n***************************** \n run forward pass during training")
    l0_module.train()
    zs = l0_module.forward(calculate_lagrangian=False)
    for key in zs:
        print(key, zs[key].shape)

    print("\n***************************** \n run forward pass during eval")
    l0_module.eval()
    zs = l0_module.forward(calculate_lagrangian=False)
    for key in zs:
        print(key, zs[key].shape)

    print("\n***************************** \n run forward pass during lagrangian")
    l0_module.train()
    loss, v = l0_module(calculate_lagrangian=True, pruned_steps=320)
    print("loss", loss.item())
    for key in v:
        if torch.is_tensor(v[key]): vv = v[key].item()
        else: vv = v[key]
        print(key, vv)
    
    print("\n***************************** \n Test target sparsity") 
    # test target_sparsity
    target_sparsity = l0_module.get_target_sparsity(50, l0_module.target_sparsity)
    print("target sparsity at step 50: ", target_sparsity)
     
    target_sparsity = l0_module.get_target_sparsity(100, l0_module.target_sparsity)
    print("target sparsity at step 100: ", target_sparsity)
    
    target_sparsity = l0_module.get_target_sparsity(200, l0_module.target_sparsity)
    print("target sparsity at step 200: ", target_sparsity)
    import pdb; pdb.set_trace()


  
if __name__ == "__main__":
    test_l0_module()
    