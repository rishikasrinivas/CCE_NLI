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
                 ):
        super(L0Module_LLAMA, self).__init__()
        
        self.model_name = model_name
            
        self.pruning_type = pruning_type
        
        #both bowman and llm
        self.final_mlp_hidden = 1024
        self.out_params = 3
        

        '''for llama ill need 
        
        headlayer_z: which layers are pruned 
        kvhead_z: which kv heads are pruned? (instead of head_z) 
            
        mlp_z: which mlp blocks are pruned 
            int_z: which neurons in the mlp blocks are pruned
                hidden_z: to prune the 2048 hidden dims (but i would not apply this to k,v) so just q inp and o output'''
                            
                            
        self.all_types = ["hidden_z", "intermediate_z", "mlp_z", "head_layer_z", "head_z", 'final_mlp_hidden_z'] #reove inp_z, #load zs_llm hidden_z and do nn.Param(hidden_z copied 4 times).req_grad=False
        
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size 
        print("Intermediate size: ", self.intermediate_size)
        self.num_attention_heads = config.num_attention_heads
        self.mlp_num_per_layer = 1
        self.dim_per_head = self.hidden_size // self.num_attention_heads
        self.num_hidden_layers = config.num_hidden_layers
        self.vocab_size = config.vocab_size

        self.params_per_head_layer = 2048*2048*2 + (2048*256*2) #self.hidden_size * self.hidden_size * 4 + self.hidden_size * 4
        self.params_per_head =  self.params_per_head_layer // self.num_attention_heads


        self.params_per_mlp_layer = (self.hidden_size * self.intermediate_size * 2) + (self.intermediate_size * self.hidden_size)
        self.params_per_intermediate_dim = self.params_per_mlp_layer // self.intermediate_size


        self.params_finalmlp_layer = (self.hidden_size * 4 * self.final_mlp_hidden) + (self.final_mlp_hidden* self.out_params) + self.final_mlp_hidden #weights &bias
        self.params_per_hidden_dim_final_mlp = self.params_finalmlp_layer // self.final_mlp_hidden

        self.hidden_loga = None

        

        

        # we ignore the parameters in normalization layers (it takes a very small amount)
        #self.full_model_size = (self.params_per_head_layer + self.params_per_mlp_layer) * self.num_hidden_layers
        
        self.temperature = temperature
        self.droprate_init = droprate_init if droprate_init != 0. else 0.5
        
        self.types = []
        self.z_logas = {}
        self.parameters_per_dim = {}
        self.sizes = {}
        self.shapes = {}

    
        
        #both bowman and llm
        self.hidden_layer_mlp_loga = None 
        
        self.prunable_model_size=full_model_size
        
        
        types = self.pruning_type.split("+") #in bomwan should be ['final_mlp_hidden']
        print(f" self.prunable_model_size: { self.prunable_model_size}")
        self.learned_zs =None
        
        #for 2048 layer (3072 in llm) doesn't apply in bowman pruning. in llm this is initiallized with hidden layer hwich isnt used in bowman so initialize it as default to bowmans
        self.input_layer_mlp_loga=Parameter(torch.Tensor(2048))
        self.input_layer_mlp_loga.requires_grad=False
        self.reset_loga(self.input_layer_mlp_loga, mean=10) #how to get 2048 fromb omwan thru code
        
    
       #both bowman and llm
        for type in types:
            if type != "layer":
                print(f"Initializing {type}")
                self.initialize_one_module(type)
        if "layer" in types:
            print("Init layer")
            self.initialize_one_module("layer")
            
        
        #everything from here down in constructor: both bowman and llm
        self.magical_number = magical_number

        self.lambda_1 = torch.nn.Parameter(torch.tensor(0.0))
        self.lambda_2 = torch.nn.Parameter(torch.tensor(0.0))

        self.lagrangian_warmup = lagrangian_warmup
        self.start_sparsity = start_sparsity
        self.target_sparsity = target_sparsity

        logger.info("********** Initializing L0 Module **********") 
        for type in self.types:
            logger.info(f"***** {type} *****")
            logger.info(f"z.shape", self.z_logas[type].shape)
            logger.info(f"size", self.sizes[type])
        logger.info(f"prunable model size: {self.prunable_model_size}")

    def set_lagrangian_warmup_steps(self, lagrangian_warmup):
        self.lagrangian_warmup = lagrangian_warmup

        
    def full_model_size(self):
        prunable_model_size = self.params_per_head * self.num_hidden_layers * self.num_attention_heads + (self.params_per_mlp_layer * self.num_hidden_layers) #this changes for bomwna
        self.llm_size = prunable_model_size
        prunable_model_size  += self.params_finalmlp_layer 
        print(f"Prunable model size: {prunable_model_size}")
        return prunable_model_size
    
    
    def initialize_one_module(self, module_name):
        #both bowman and llm
        if module_name == "structured_mlp":
            self.initialize_structured_mlp()
        elif module_name == "structured_heads":
            self.initialize_structured_head()
        elif module_name == "hidden":
            self.initialize_hidden()
        elif module_name == "layer":
            self.initialize_whole_mlp()
            self.initialized_layer_structured_heads()
        #added for zs to final mlp
        elif module_name == 'final_mlp_hidden':
            print("final ml")
            self.initialize_final_hidden_layer_mlp()
            
    def add_one_module(self, z_loga, type, parameter_per_dim, size, shape): #! init the z_logas
        
        self.types.append(type)
        self.z_logas[type] = z_loga
        self.parameters_per_dim[type] = parameter_per_dim
        self.sizes[type] = size
        self.shapes[type] = shape

    def initialize_parameters(self, size, num_layer=None):
        if num_layer is not None:
            return Parameter(torch.Tensor(num_layer, size))
        else:
            return Parameter(torch.Tensor(size))

    def initialize_hidden(self): 
        if self.learned_zs and 'hidden_z' in self.learned_zs:
            self.hidden_loga = Parameter(self.learned_zs['hidden_z'].reshape(self.hidden_size)).to(device)
            self.hidden_loga.requires_grad = False
        else:
            self.hidden_loga = self.initialize_parameters(self.hidden_size) #shared across all layers if in 768->3072->768 neuron 0 is pruned out that is the 0th in the first  768 and the last 768 
            self.reset_loga(self.hidden_loga, mean=10)
        self.input_layer_mlp_loga = Parameter(torch.cat([self.hidden_loga for i in range(4)]))
        self.input_layer_mlp_loga.requires_grad=False  
        self.add_one_module(self.hidden_loga, type="hidden", 
                            parameter_per_dim=self.hidden_size * 4 + self.hidden_size * 4 * 2,
                            size=self.hidden_size, shape=[self.hidden_size])
        
        logger.info(f"Initialized hidden loga! Prunable_model_size = {self.prunable_model_size}")

    def initialize_structured_head(self, add_prunable_model_size=True):
        if self.learned_zs and 'head_z' in self.learned_zs:
            self.head_loga = Parameter(self.learned_zs['head_z'].reshape(self.num_hidden_layers, 4))
            self.head_loga.requires_grad = False
        else:
            self.head_loga = self.initialize_parameters(4, self.num_hidden_layers)
            self.reset_loga(self.head_loga, mean=10)
            
        self.add_one_module(self.head_loga, type="head", 
                            parameter_per_dim=self.params_per_head, size=4,
                            shape=[self.num_hidden_layers, 1, 4, 1, 1])
        logger.info(f"Initialized structured heads! Prunable_model_size = {self.prunable_model_size}")

    def initialized_layer_structured_heads(self):
        n_layer = self.num_hidden_layers
        if self.learned_zs and 'head_layer_z' in self.learned_zs:
            self.headlayer_loga = Parameter(self.learned_zs['head_layer_z'].reshape(n_layer))
            self.headlayer_loga.requires_grad = False
        else:
            self.headlayer_loga = self.initialize_parameters(n_layer)
            self.reset_loga(self.headlayer_loga, mean=10)
        self.add_one_module(self.headlayer_loga, type="head_layer", 
                            parameter_per_dim=self.params_per_head * self.num_attention_heads, size=1,
                            shape=[n_layer])
        logger.info(f"Initialized layerwise structured heads! Prunable_model_size = {self.prunable_model_size}")

    def initialize_structured_mlp(self):
        if self.learned_zs and 'intermediate_z' in self.learned_zs:
            self.int_loga=Parameter(self.learned_zs['intermediate_z'].reshape(self.num_hidden_layers, self.intermediate_size))
            print("intermeidate shaep ",self.int_loga.shape )
            self.int_loga.requires_grad = False
        else:
            self.int_loga = self.initialize_parameters(self.intermediate_size, self.num_hidden_layers)
            self.reset_loga(self.int_loga)
            
        self.add_one_module(self.int_loga, type="intermediate", 
                            parameter_per_dim=self.params_per_intermediate_dim, size=self.intermediate_size,
                            shape=[self.num_hidden_layers, 1, 1, self.intermediate_size])
        
        logger.info(f"Initialized structured mlp! Prunable_model_size = {self.prunable_model_size}")
    
    def initialize_whole_mlp(self):
        n_layer = self.num_hidden_layers
        if self.learned_zs and 'mlp_z' in self.learned_zs:
            self.intlayer_loga = Parameter(self.learned_zs['mlp_z'].reshape(n_layer))
            self.intlayer_loga.requires_grad = False
        else:
            self.intlayer_loga = self.initialize_parameters(n_layer)
            self.reset_loga(self.intlayer_loga, mean=10)
        self.add_one_module(self.intlayer_loga, type="mlp", 
                            parameter_per_dim=self.params_per_mlp_layer, size=self.mlp_num_per_layer,
                            shape=[n_layer])
        
        logger.info(f"Initialized whole mlps! Prunable_model_size = {self.prunable_model_size}")

    def initialize_final_hidden_layer_mlp(self): #also add final_layer_hid_mlp to self.types
      
         #both bowman and llm
        if self.learned_zs and 'final_mlp_hidden_z' in self.learned_zs:
            self.hidden_layer_mlp_loga = Parameter(self.learned_zs['final_mlp_hidden_z'].reshape(n_layer))
            self.hidden_layer_mlp_loga.requires_grad=False
            
        else:
            self.hidden_layer_mlp_loga = self.initialize_parameters(self.final_mlp_hidden) #3072,1024 this will prune the 1024
            self.reset_loga(self.hidden_layer_mlp_loga, mean=10)
        self.add_one_module(self.hidden_layer_mlp_loga, type="final_mlp_hidden", 
                            parameter_per_dim=self.out_params, size=self.final_mlp_hidden,
                            shape=[self.final_mlp_hidden])
       
        logger.info(f"Initialized final layer hidden mlp! Prunable_model_size = {self.prunable_model_size}")
    

    #both bowman and llm
    def reset_loga(self, tensor, mean=None):
        if mean is None:
            mean = math.log(1 - self.droprate_init) - math.log(self.droprate_init)
        tensor.data.normal_(mean, 1e-2)
        
    #both bowman and llm
    def reset_qz_logas(self):
        for key in self.z_logas:
            if key in ["head_layer", "mlp", "head"]:
                self.reset_loga(self.z_logas[key], 10)
            else:
                self.reset_loga(self.z_logas[key])
                
     #both bowman and llm
    def constrain_parameters(self):
        def _constrain(tensor):
            tensor.data.clamp_(min=math.log(1e-2), max=math.log(1e2))
        for key in self.z_logas:
            _constrain(self.z_logas[key])
            
    #both bowman and llm
    def cdf_qz(self, x, loga):
        """Implements the CDF of the 'stretched' concrete distribution"""
        xn = (x - limit_a) / (limit_b - limit_a)
        logits = math.log(xn) - math.log(1 - xn)
        return torch.sigmoid(logits * self.temperature - loga).clamp(min=epsilon, max=1 - epsilon)
    
    #both bowman and llm
    def quantile_concrete(self, x, loga):
        y = torch.sigmoid((torch.log(x) - torch.log(1 - x) + loga) / self.temperature)
        return y * (limit_b - limit_a) + limit_a
    #both bowman and llm
    def get_num_parameters_for_one(self, loga, parameter_size):
        return torch.sum(1 - self.cdf_qz(0, loga)) * parameter_size

    def transform_scores_for_head(self):
        if "head_layer" in self.types:
            
            all_head_score = 1 - self.cdf_qz(0, self.headlayer_loga)
        else:
            all_head_score = None
        head_score = 1 - self.cdf_qz(0, self.head_loga) # 12 * 12
       
        if all_head_score is not None:
            all_head_score = all_head_score.view(-1, 1, 1) # 12 * 1 * 1
        head_score = head_score.unsqueeze(-1)   # 12 * 12 * 1
       
        return all_head_score, head_score

    #both bowman and llm
    def get_num_parameters_and_constraint_for_hidden(self):
        """
        Calculate expected parameters using z-scores ONLY.
        Must match the actual architecture exactly.
        """
        num_parameters = 0

        # ---------------------
        # Scores from z variables
        # ---------------------
        all_head_score, head_score = self.transform_scores_for_head()
        hidden_score = 1 - self.cdf_qz(0, self.hidden_loga)  # (H_pruned,) = (2044,)

        if all_head_score is not None:
            layer_score = all_head_score.reshape(-1)  # (L,)
            head_score = (all_head_score * head_score).reshape(-1)  # (L * n_kv,)
        else:
            layer_score = torch.ones(self.num_layers, device=hidden_score.device)
            head_score = head_score.reshape(-1)

        H_orig = self.hidden_size  # 2048 (original)
        D = self.dim_per_head  # 64

        # =====================
        # ATTENTION PARAMETERS
        # =====================
        # Architecture shows:
        # Q: (H_pruned, H_orig) = (2044, 2048)
        # K: (H_pruned, 256) = (2044, n_heads_pruned * D)
        # V: (H_pruned, 256) = (2044, n_heads_pruned * D)
        # O: (H_orig, H_pruned) = (2048, 2044)

        # Q projection: sum(hidden_score) × H_orig per layer
        # O projection: H_orig × sum(hidden_score) per layer
        # Combined Q + O: 2 × sum(hidden_score) × H_orig per layer
        # Q+O: hidden_dim  (same for all layers) * total_active kv heads * 512 
            #say 2 layers if hidden = 2044 and lauer1 has 1 kv head pruned and layer 2 has 2 ie 3 in first, 2 in second:
                #2 * 2044 * (2048 - (64x8(4-3))) + 2 * 2044 * (2048 - (64x8(4-2)))  ----> 2*2044*64x8x3 + 2*2044*64*8*2 --> 2*hidden*512*(total kv heads saved)
        num_parameters +=  2 * torch.sum(hidden_score) * torch.sum(head_score) * D*8

        # K + V projections: sum(hidden_score) × D per head × 2 (K and V)
        num_parameters += torch.sum(head_score) * torch.sum(hidden_score) * D * 2

        # =====================
        # MLP PARAMETERS
        # =====================
        intlayer_score = 1 - self.cdf_qz(0, self.intlayer_loga)  # (L,)
        int_score = 1 - self.cdf_qz(0, self.int_loga)  # (L, I)
        int_score = (intlayer_score.unsqueeze(-1) * int_score).reshape(-1)  # (L*I,)

        # gate_proj: (H_pruned, I_layer)
        # up_proj: (H_pruned, I_layer)
        # down_proj: (I_layer, H_pruned)
        # All 3 have same parameter count: H_pruned × I_layer
        num_parameters += torch.sum(torch.outer(hidden_score, int_score)) * 3

        # =====================
        # LAYER NORMS
        # =====================
        # Each layer has 2 RMSNorms, final layer has 1 RMSNorm
        # Each RMSNorm has H_pruned parameters (weight only, no bias)
        num_layernorms = 2 * 22 + 1  # 2 per layer + 1 final
        num_parameters += num_layernorms * torch.sum(hidden_score)

        return num_parameters


    def get_num_parameters_for_mlp(self):
        """
        Calculate expected classifier parameters using z-scores ONLY.
        """
        inp_layer_score = 1 - self.cdf_qz(0, self.input_layer_mlp_loga)  # (8192,)
        hid_layer_score = 1 - self.cdf_qz(0, self.hidden_layer_mlp_loga)  # (1024,)

        # Linear 1: (inp, hid) with bias
        expected_params_1 = torch.sum(torch.outer(inp_layer_score, hid_layer_score))
        expected_bias_1 = torch.sum(hid_layer_score)

        # Linear 2: (hid, 3) with bias
        expected_params_2 = torch.sum(torch.outer(hid_layer_score, torch.ones(3, device=hid_layer_score.device)))
        expected_bias_2 = 3  # Always 3 (num classes)

        # BatchNorm1d: weight + bias for inp dimension
        bn_params = 2 * torch.sum(inp_layer_score)

        num_parameters = (expected_params_1 + expected_bias_1 + 
                         expected_params_2 + expected_bias_2 + 
                         bn_params)

        return num_parameters


   
    def get_num_parameters_and_constraint(self):
        num_parameters = 0

        all_head_score, head_score = self.transform_scores_for_head()
        primt("l0m llama 358 all_head_score:", all_head_score.shape, "head_score: ",  head_score.shape)
        head_score = head_score * all_head_score
        num_parameters += torch.sum(head_score) * self.parameters_per_dim["head"]

        intlayer_score = 1 - self.cdf_qz(0, self.intlayer_loga)  # 12
        int_score = 1 - self.cdf_qz(0, self.int_loga)  # 12 * 3072
        intlayer_score = intlayer_score.unsqueeze(-1)

        int_score = int_score * intlayer_score
        num_parameters += torch.sum(int_score) * self.parameters_per_dim["intermediate"]
        
        
        return num_parameters


    def get_target_sparsity(self, pruned_steps):
   
        target_sparsity = (self.target_sparsity - self.start_sparsity) * min(1, pruned_steps / self.lagrangian_warmup) + self.start_sparsity
        return target_sparsity


    def lagrangian_regularization(self, pruned_steps):
        target_sparsity = self.target_sparsity
        #if bowman only get_for_mlp
        expected_size = self.get_num_parameters_for_mlp() + self.get_num_parameters_and_constraint_for_hidden() #! calculate \bar s
   
        expected_sparsity = 1 - (expected_size / self.prunable_model_size)
        
    
        if self.lagrangian_warmup > 0:
            target_sparsity = self.get_target_sparsity(pruned_steps)
        lagrangian_loss = (
            self.lambda_1 * (expected_sparsity - target_sparsity)
            + self.lambda_2 * (expected_sparsity - target_sparsity)**2
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
                new_z = z.squeeze().detach().cpu().numpy() > 0
            numpified_zs[name] = new_z
        return numpified_zs

    def calculate_model_size_LLM(self, zs):
        numpified_zs = self.get_z_from_zs(zs)
        hidden_z = numpified_zs.get("hidden", np.ones(self.hidden_size))
        intermediate_z = numpified_zs.get("intermediate", np.ones((self.num_hidden_layers, self.intermediate_size)))
        mlp_z = numpified_zs.get("mlp", np.ones(self.num_hidden_layers)).reshape(-1, 1)
        head_z = numpified_zs.get("head", np.ones((self.num_hidden_layers, 4)))
        head_layer_z = numpified_zs.get("head_layer", np.ones((self.num_hidden_layers,))).reshape(-1, 1)

        # NOTE: final_mlp is excluded (it's the "classifier")
        # NOTE: We don't need final_mlp_hidden or final_mlp_input for parameter counting

        print(f"hidden_z: {hidden_z.shape}\nintermediate_z: {intermediate_z.shape}\nmlp_z: {mlp_z.shape}\nhead_z: {head_z.shape}\nhead_layer_z: {head_layer_z.shape}")

        remaining_hidden_dims = hidden_z.sum().item()
        remaining_intermediate_nums = intermediate_z.sum(axis=-1).tolist()
        remaining_head_nums = head_z.sum(axis=-1).tolist()

        H_orig = self.hidden_size  # 2048
        D = self.dim_per_head  # 64
        L = self.num_hidden_layers  # 22

        # ===========================
        # WHAT calculate_parameters() COUNTS:
        # - Attention: Q, K, V, O
        # - MLP: gate, up, down
        # - LayerNorms: input_layernorm, post_attention_layernorm, final norm
        # 
        # WHAT IT EXCLUDES (keys):
        # - "embedding": embed_tokens
        # - "layer_transformation": the Linear(2044, 2048) layer
        # - "classifier": the final MLP (bn, mlp.0, mlp.1, mlp.2, mlp.3)
        # - "pooler": (not present in your model)
        # ===========================

        # 1. ATTENTION PARAMETERS
        # Q: (H_pruned, H_orig) = (2044, 2048)
        # K: (H_pruned, 256) = (2044, n_heads_pruned * D)
        # V: (H_pruned, 256) = (2044, n_heads_pruned * D)
        # O: (H_orig, H_pruned) = (2048, 2044)

        # K + V: H_pruned * D per active head * 2
        kv_nums = np.outer((head_z * head_layer_z).reshape(-1), hidden_z).sum().item()

        kv_params = kv_nums * D * 2          # each KV head → (D, hidden)
        q_o_params = kv_nums * 512 * 2   

        attn_params = q_o_params + kv_params

        # 2. MLP PARAMETERS
        # gate, up, down: 3 * H_pruned * I_pruned
        intermediate_nums = np.outer((intermediate_z * mlp_z).reshape(-1), hidden_z).sum().item()
        mlp_params = intermediate_nums * 3

        # 3. LAYER NORMS
        # 2 per layer + 1 final = (2 * L + 1) * H_pruned
        num_layernorms = 2 * L + 1
        layernorm_params = num_layernorms * remaining_hidden_dims

        # TOTAL (backbone only, no classifier, no embeddings, no layer_transformation)
        
        mlp_final_hidden = numpified_zs.get("final_mlp_hidden", np.ones(1024))
        mlp_final_input = numpified_zs.get("final_mlp_input", np.ones(8192))
        remaining_mlp_inp = mlp_final_input.sum().item()
        remaining_mlp_hidden = mlp_final_hidden.sum().item()
        final = (remaining_mlp_inp * remaining_mlp_hidden) + remaining_mlp_hidden + (remaining_mlp_hidden*3)+3
        remaining_model_size = attn_params + mlp_params + final + layernorm_params
        pruned_model_size = self.prunable_model_size - remaining_model_size

        print(f"\nPARAMETER BREAKDOWN (matching calculate_parameters):")
        print(f"  Q+O:        {q_o_params:,}")
        print(f"  K+V:        {kv_params:,}")
        print(f"  MLP:        {mlp_params:,}")
        print(f"  LayerNorm:  {layernorm_params:,}")
        print(f"  Final MLP:  {final:,}")
        print(f"  ──────────────────────────")
        print(f"  BACKBONE:   {remaining_model_size:,}")
        print(f"\n  Prunable:   {self.prunable_model_size:,}")
        print(f"  Pruned:     {pruned_model_size:,}")
        print(f"  Sparsity:   {pruned_model_size / self.prunable_model_size * 100:.2f}%")

        # For results, still track classifier dimensions (even though not counted in sparsity)
        
        #[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 'mlp_layers': [1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1], 'hidden_dims': 2044, 'intermediate_dims': [5552, 5527, 5532, 5549, 5550, 5551, 5549, 5558, 5570, 5566, 5570, 5576, 5579, 5586, 5589, 5590, 5596, 5604, 5604, 5600, 5599, 5595], 'head_nums': [4, 4, 4, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4], 'mlp_input_8192': 8192.0, 'mlp_input_1024': 1022,
        80, 105, 100, 83, 82, 81, 83, 74, 62, 66, 62, 56, 53, 46, 43, 42, 36, 28, 28, 32, 33, 37
        
        results = {}
        results["head_layers"] = head_layer_z.reshape(-1).astype(int).tolist()
        results["mlp_layers"] = mlp_z.reshape(-1).astype(int).tolist()
        results["hidden_dims"] = remaining_hidden_dims
        results["intermediate_dims"] = remaining_intermediate_nums
        results["head_nums"] = remaining_head_nums
        results["mlp_input_8192"] = remaining_mlp_inp
        results["mlp_input_1024"] = remaining_mlp_hidden
        results["pruned_params"] = pruned_model_size
        results["remaining_params"] = remaining_model_size
        results["pruned_model_sparsity"] = pruned_model_size / self.prunable_model_size

        logger.info(f"remaining_head_layers: {head_layer_z.reshape(-1).tolist()}")
        logger.info(f"remaining_mlp_layers: {mlp_z.reshape(-1).tolist()}")
        logger.info(f"remaining_hidden_dims: {remaining_hidden_dims}")
        logger.info(f"remaining_mlp_inp: {remaining_mlp_inp}")
        logger.info(f"remaining_mlp_hidden: {remaining_mlp_hidden}")
        logger.info(f"remaining_intermediate_nums: {remaining_intermediate_nums}")
        logger.info(f"remaining_head_nums: {remaining_head_nums}")
        logger.info(f"pruned_model_size: {pruned_model_size}")
        logger.info(f"remaining_model_size: {remaining_model_size}")

        return results

    def calculate_model_size(self,zs):
        return self.calculate_model_size_LLM(zs)
    
    
    def forward(self, training=True,debug=False):
        zs = {f"{type}_z": [] for type in self.types}

        if training:
            
            for i, type in enumerate(self.types):
                loga = self.z_logas[type]
                z = self._sample_z(loga)
                
                zs[f"{type}_z"] = z.reshape(self.shapes[type])
        else:
            for i, type in enumerate(self.types):
 
                if type != "hidden" and type != 'final_mlp_hidden' and type != 'final_mlp_inp':
                #self.hidden_layer_mlp_loga, self.input_layer_mlp_loga, self.hidden_loga': # hidden is not a per layer sample
                    loga_all_layers = self.z_logas[type]
                    for layer in range(len(loga_all_layers)):
                        loga = loga_all_layers[layer]
                        size = self.sizes[type]
                        z = self._deterministic_z(size, loga)
                        zs[f"{type}_z"].append(z.reshape(self.shapes[type][1:]))
                else:
                    if type == 'hidden':
                        if debug:
                            z=self.hidden_loga
                        else:
                            z = self._deterministic_z(self.sizes[type], self.hidden_loga)
                    elif type== 'final_mlp_hidden':
                        if debug:
                            z=self.hidden_layer_mlp_loga
                        else:
                            z = self._deterministic_z(self.sizes[type], self.hidden_layer_mlp_loga)
                    zs[f"{type}_z"] = z
            for type in zs:
                if type != "hidden_z" and type != 'final_mlp_hidden_z' and type != 'final_mlp_inp_z': #used to be if type != hidden_z
                    zs[type] = torch.stack(zs[type])
        return zs 

if __name__ == "__main__":
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained("bert-base-uncased")
    l0_module = L0Module(config, lagrangian_warmup=200, target_sparsity=0.5)
