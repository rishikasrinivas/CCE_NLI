import math
import sys

import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F

from argparse import Namespace as NS
from typing import Any, List

from composer.core.time import Time

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
        # THIS PART MODIFIED FOR VANILLA COFI
        expected_score = 1 - self.cdf_qz(z_loga)
        expected_num_nonzeros = expected_score.sum()
        expected_num_zeros = self.mask_size - expected_num_nonzeros.item()
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
 
class L0Module_LLAMA(nn.Module):
    def __init__(self, self,
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
                 device='cuda'):
        super(L0Module, self).__init__()

        # base and target model info
        n_matrix_mlp =  3
        #self.base_model_info = self.set_model_info(cfg, n_matrix_mlp=n_matrix_mlp) 
        
        # Model Config Vars
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
        self.prunable_model_size = self.calculate_prunable_model_size(self) 
        
        
        # l0 config
        self.pruning_modules = pruning_type.split("+")         
        self.start_sparsity = 0.0
        self.lagrangian_warmup_steps = lagrangian_warmup
        self.device = device
    
        
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
        self.prunable_model_size = self.calculate_prunable_model_size(self.base_model_info)
        
        self.target_sparsity =  target_sparsity

        print("********** Initializing L0 Module **********") 
        for pruning_module in self.pruning_modules:
            print(f"***** {pruning_module} *****")
            print(f"z.shape", self.masks[pruning_module].z_loga.shape)
            print(f"size", self.masks[pruning_module].mask_size)
        print(f"prunable model size: {self.prunable_model_size}")
        
    
    
        
    def calculate_prunable_model_size(self):
        prunable_mlp_size = self.params_per_mlp_layer * self.num_layers
        prunable_head_layer_size = self.params_per_head_layer * self.num_layers
        prunable_model_size = 0
        if "hidden" in self.pruning_modules:
            return prunable_mlp_size + prunable_head_layer_size
        if "head_layer" in self.pruning_modules or "head" in self.pruning_modules:
            prunable_model_size += prunable_head_layer_size
        if "mlp" in self.pruning_modules or "intermediate" in self.pruning_modules:
            prunable_model_size += prunable_mlp_size
        if "final_mlp_hidden" in self.pruning_modules:
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
        mask_shape = [self.base_model_info.hidden_size]
        #num_params_per_mask=self.base_model_info.hidden_size * 4 + self.base_model_info.hidden_size * 4 * 2
        num_params_per_mask= (
            self.hidden_size      # q input
            + 256                 # k input
            + 256                 # v input
            + self.hidden_size    # o output
            + 3 * self.intermediate_size
        )
        
       
        target_mask_size = self.hidden_size
        pd = {"lambda_1_hidden": torch.nn.Parameter(torch.tensor(0.0, device=self.device)),
              "lambda_2_hidden": torch.nn.Parameter(torch.tensor(0.0, device=self.device))}
        self.lambdas.update(pd)
        
        hidden_mask = Mask(name="hidden",
                           mask_shape=mask_shape,
                           num_params_per_mask=num_params_per_mask,
                           mask_output_shape=[self.hidden_size],
                           target_mask_size=target_mask_size,
                           device=self.device,)
        self.masks["hidden"] = hidden_mask

    def initialize_head(self):
        mask_shape = [self.num_layers, self.num_key_value_heads]
        num_params_per_mask = self.params_per_head
        mask_output_shape = [self.num_layers, 1, self.num_key_value_heads, 1] 
        
        target_mask_size = self.num_key_value_heads
        pd = {"lambda_1_head": torch.nn.Parameter(torch.tensor(0.0, device=self.device)),
              "lambda_2_head": torch.nn.Parameter(torch.tensor(0.0, device=self.device))}
        self.lambdas.update(pd)
        head_mask = Mask(name="head",
                         mask_shape=mask_shape,
                         num_params_per_mask=num_params_per_mask,
                         mask_output_shape=mask_output_shape,
                         target_mask_size=target_mask_size,
                           device=self.device,)
        self.masks["head"] = head_mask 

        
    def initialize_head_layer(self):
        mask_shape = [self.num_layers]
        num_params_per_mask=self.params_per_head *  self.config.num_key_value_heads
        mask_output_shape = [self.num_layers] 
        

        target_mask_size = self.target_model_info.num_layers
        pd = {"lambda_1_head_layer": torch.nn.Parameter(torch.tensor(0.0, device=self.device)),
              "lambda_2_head_layer": torch.nn.Parameter(torch.tensor(0.0, device=self.device))}
        self.lambdas.update(pd)
        
        head_layer_mask = Mask(name="head_layer",
                               mask_shape=mask_shape,
                               num_params_per_mask=num_params_per_mask,
                               mask_output_shape=mask_output_shape,
                               target_mask_size=target_mask_size,
                           device=self.device)
        self.masks["head_layer"] = head_layer_mask
        
    def initialize_intermediate(self):
        mask_shape = [self.num_layers, self.intermediate_size]
        num_params_per_mask=self.params_per_intermediate_dim
        mask_output_shape = [self.num_layers, 1, 1, self.intermediate_size] 
        
       
        target_mask_size = self.intermediate_size
        pd = {"lambda_1_intermediate": torch.nn.Parameter(torch.tensor(0.0, device=self.device)),
              "lambda_2_intermediate": torch.nn.Parameter(torch.tensor(0.0, device=self.device))}
        self.lambdas.update(pd)
        
        int_mask = Mask(name="intermediate",
                        mask_shape=mask_shape,
                        num_params_per_mask=num_params_per_mask,
                        mask_output_shape=mask_output_shape,
                        target_mask_size=target_mask_size,
                           device=self.device)
        self.masks["intermediate"] = int_mask
       

    def initialize_mlp(self):
        mask_shape = [self.num_layers]
        num_params_per_mask=self.params_per_mlp_layer
        mask_output_shape = [self.num_layers] 
        
        
        target_mask_size = self.num_layers
        pd = {"lambda_1_mlp": torch.nn.Parameter(torch.tensor(0.0, device=self.device)),
              "lambda_2_mlp": torch.nn.Parameter(torch.tensor(0.0, device=self.device))}
        self.lambdas.update(pd)
        
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
        head_score = expected_scores["head"] # 12 * 12

        head_layer_score = None
        if "head_layer" in expected_scores:
            head_layer_score = expected_scores["head_layer"]
        
        if head_layer_score is not None:
            head_layer_score = head_layer_score.view(-1, 1) # 12 * 1
       
        return head_layer_score, head_score

    def transform_scores_for_mlp(self, expected_scores: dict):
        mlp_score = None
        if "mlp" in expected_scores:
            mlp_score = expected_scores["mlp"] # 12
        
        if mlp_score is not None:
            mlp_score = mlp_score.unsqueeze(-1)
        
        intermediate_score = expected_scores["intermediate"] # 12 * 3072
        return mlp_score, intermediate_score


    def get_expected_num_params(self, expected_scores: dict): #! calculate the current parsity
        num_parameters = 0
       
        # 12 * 1 
        # 12 * 12
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
            head_score = torch.repeat_interleave(head_score, self.base_model_info.dim_per_head, dim=1) # 12 * 768

            qk_score = head_score * qk_head_dim_score # 12 * 768
            vo_score = head_score * vo_head_dim_score # 12 * 768
                
        if "hidden" in expected_scores:
            hidden_score = expected_scores["hidden"] # 768 
            
            if qk_score is None:
         
                hidden_score = expected_scores["hidden"]  # (H,)

                q_per_kv = self.num_attention_heads // self.num_key_value_heads


                attn_per_hidden_per_kv = (
                    2 * q_per_kv * self.dim_per_head # Q+O (pruning 1 kv head affects 8 querys hidden dims and 8 output hidden dims each of 64 neurons)
                    + 2 * self.dim_per_head # K+V each kv head affects 64 neurons in key and value layer
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
                num_parameters += torch.sum(torch.matmul(hidden_score.reshape(1, -1, 1), qk_score.unsqueeze(1))) * 2 # 12 * 768 * 768
                num_parameters += torch.sum(torch.matmul(hidden_score.reshape(1, -1, 1), vo_score.unsqueeze(1))) * 2 # 12 * 768 * 768
                num_parameters += torch.sum(torch.matmul(hidden_score.reshape(1, -1, 1), int_score.unsqueeze(1))) * 3 # 12 * 768 * 3072
        else:
            num_parameters += torch.sum(head_score) * self.masks.head.num_params_per_mask
            num_parameters += torch.sum(int_score) * self.masks.intermediate.num_params_per_mask
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
        return lagrangian_loss, return_v
 
    def forward(self, calculate_lagrangian: bool = False, pruned_steps: int = 0):
        self.constrain_parameters()
        if calculate_lagrangian:
            return self.lagrangian_regularization(pruned_steps)
        
        zs = {f"{pruning_module}_z": [] for pruning_module in self.pruning_modules}
        
        
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
        
        return zs 


