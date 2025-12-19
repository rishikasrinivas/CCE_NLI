import torch
import os
from transformers.modeling_utils import prune_linear_layer
from transformers import AutoConfig, BertForSequenceClassification
import torch
from safetensors.torch import load_file

from cofi.utils.utils import calculate_parameters

def edit_config(config, additional_args):
    config.transform_embedding = additional_args.transform_embedding
    config.do_distill = additional_args.do_distill
    config.do_layer_distill = additional_args.do_layer_distill

def initialize_layer_transformation(model):
    model.layer_transformation.weight.data.copy_(
        torch.eye(len(model.layer_transformation.weight)))
    model.layer_transformation.bias.data.fill_(0)

def load_model_with_zs(model_path, model, zs=None, encoder=None, **kwargs):
    if 'BOWMAN' in model_path:
        config=None
    else:
        #config=AutoConfig.from_pretrained(os.path.join("/".join(model_path.split("/")[:-1]), "config.json"))
        config=AutoConfig.from_pretrained(os.path.join(model_path, "config.json"))
    model = model.from_pretrained(
        pretrained_model_name_or_path= os.path.join(model_path, 'model_best.pth'), # if llm part of student model is alr trained itll be here otherwise a default model will be loaded and finetuned
        from_tf=False,
        teacher=True,
        config=config,
        encoder=encoder,
        **kwargs,
    )
    if zs is None:
        return model
    '''if "model_best.pth" not in model_path:
        p =  os.path.join(model_path, "model_best.pth")
    else:
        p = model_path
    loaded_weights = torch.load(p)['state_dict']
    
    model.load_state_dict(loaded_weights)'''
    
    print(f"Load weights from {model_path}")

    print(f"Model Size before pruning: {calculate_parameters(model)}")

    if model.model_name == 'bowman':
        update_LSTM_params(model, zs)
        prune_hidden_mlp(zs,model)
    else:
        update_LLM_params(model, zs) #changes weights
        prune_model_with_z(zs, model) #changes strucutre
        
    print(f"Model Size after pruning: {calculate_parameters(model)}")
    return model

def load_model(model_path, model, zs=None, encoder=None, **kwargs):
    model = load_model_with_zs(model_path, model, zs, encoder, **kwargs)
    print(f"Model Size: {calculate_parameters(model)}")
    return model

# load the l0 module
def load_l0_module(model_path):
    l0_module_path = os.path.join(model_path, "l0_module.pt")
    if os.path.exists(l0_module_path):
        return torch.load(l0_module_path, map_location=torch.device('cpu'))
    else:
        return None
# z values could be in [0, 1), we update the parameters accordingly with z values
def update_LSTM_params(model, zs):
    bowman = model 

    if zs is not None:
        if 'final_mlp_hidden_z' in zs:
            final_mlp_hidden_z = zs['final_mlp_hidden_z'].cpu().clone()
            bowman.mlp[3].weight.data=bowman.mlp[3].weight.data.mul(final_mlp_hidden_z)
            
# z values could be in [0, 1), we update the parameters accordingly with z values
def update_LLM_params(model, zs):
    bert = model.bert if hasattr(model, "bert") else model.model

    config = model.config
    hidden_dims = config.hidden_size
    num_heads = config.num_attention_heads
    dims_per_head = hidden_dims // num_heads
    num_layers = config.num_hidden_layers

    if zs is not None:
        if "intermediate_z" in zs:
            for layer in range(num_layers):
                intermediate_z = zs["intermediate_z"][layer].cpu().squeeze().clone()
                bert.encoder.layer[layer].output.dense.weight.data = bert.encoder.layer[layer].output.dense.weight.data.mul(intermediate_z)
                if "mlp_z" in zs:
                    mlp_z = zs["mlp_z"][layer].cpu()
                    bert.encoder.layer[layer].output.dense.weight.data = bert.encoder.layer[layer].output.dense.weight.data.transpose(0, 1).mul(mlp_z).transpose(0, 1)
                    bert.encoder.layer[layer].output.dense.bias.data = bert.encoder.layer[layer].output.dense.bias.data.mul(mlp_z)

        if "head_z" in zs:
            for layer in range(num_layers):
                head_z = zs["head_z"][layer].cpu().squeeze().clone()
                head_z = torch.repeat_interleave(head_z, dims_per_head)
                bert.encoder.layer[layer].attention.self.value.weight.data = bert.encoder.layer[layer].attention.self.value.weight.transpose(0, 1).data.mul(head_z).transpose(0, 1)
                bert.encoder.layer[layer].attention.self.value.bias.data = bert.encoder.layer[layer].attention.self.value.bias.data.mul(head_z)
                if "head_layer_z" in zs:
                    head_layer_z = zs["head_layer_z"][layer].cpu()
                    bert.encoder.layer[layer].attention.output.dense.weight.data = bert.encoder.layer[
                        layer].attention.output.dense.weight.transpose(0, 1).data.mul(head_layer_z).transpose(0, 1)
                    bert.encoder.layer[layer].attention.output.dense.bias.data = bert.encoder.layer[
                        layer].attention.output.dense.bias.data.mul(head_layer_z)

        if "hidden_z" in zs:
            hidden_z = zs["hidden_z"].cpu().squeeze().clone()
            bert.embeddings.word_embeddings.weight.data =\
                bert.embeddings.word_embeddings.weight.data.mul(hidden_z)
            bert.embeddings.position_embeddings.weight.data = \
                bert.embeddings.position_embeddings.weight.data.mul(hidden_z)
            bert.embeddings.token_type_embeddings.weight.data = \
                bert.embeddings.token_type_embeddings.weight.data.mul(hidden_z)
            for layer in range(num_layers):
                bert.encoder.layer[layer].attention.self.key.weight.data = bert.encoder.layer[layer].attention.self.key.weight.data.mul(hidden_z)
                bert.encoder.layer[layer].attention.self.query.weight.data = bert.encoder.layer[layer].attention.self.query.weight.data.mul(hidden_z)
                bert.encoder.layer[layer].attention.self.value.weight.data = bert.encoder.layer[layer].attention.self.value.weight.data.mul(hidden_z)
                bert.encoder.layer[layer].attention.output.dense.weight.data = bert.encoder.layer[layer].attention.output.dense.weight.data.transpose(0, 1).mul(hidden_z).transpose(0, 1)
                bert.encoder.layer[layer].attention.output.dense.bias.data = bert.encoder.layer[layer].attention.output.dense.bias.data.mul(hidden_z)
                bert.encoder.layer[layer].intermediate.dense.weight.data = bert.encoder.layer[layer].intermediate.dense.weight.data.mul(hidden_z)
                bert.encoder.layer[layer].output.dense.weight.data = bert.encoder.layer[layer].output.dense.weight.data.transpose(0, 1).mul(hidden_z).transpose(0, 1)
            if hasattr(bert.pooler, "dense"):
                bert.pooler.dense.weight.data = bert.pooler.dense.weight.data.mul(hidden_z)
            if hasattr(model, "qa_outputs"):
                model.qa_outputs.weight.data = model.qa_outputs.weight.data.mul(hidden_z)
            
            model.mlp[0].weight.data=model.mlp[0].weight.data.mul(torch.cat([hidden_z for _ in range(4)]))

        if 'final_mlp_hidden_z' in zs:
            final_mlp_hidden_z = zs['final_mlp_hidden_z'].cpu().clone()
            model.mlp[3].weight.data=model.mlp[3].weight.data.mul(final_mlp_hidden_z)
            

def prune_model_with_z(zs, model):
    
    if zs is None:
        return None, None
    concat_index=None
    bert = model.bert if hasattr(model, "bert") else None #this corresponds to cofiBert ir cofiLLama (calling both bert here for jow but 2nd bert is actually llama)
    llama = model.model if hasattr(model, "model") else None
    
    assert (hasattr(model, "model")  and llama is not None) or  (hasattr(model, "bert") and bert is not None)
    if "head_z" in zs:
        head_z = zs.get("head_z", None)
        
    
        head_layer_z = zs.get("head_layer_z", None)

        prune_heads = {}
        for layer in range(len(head_z)):
            head_z_layer = head_z[layer].cpu().squeeze().clone()
            if head_layer_z is not None:
                head_z_layer *= head_layer_z[layer]
            index = torch.where(head_z_layer == 0)[0].tolist()
            prune_heads[layer] = index

            print(f"Layer {layer}, heads {' '.join([str(i) for i in index])} pruned.")
        model.prune_heads(prune_heads)

        
    kept_intermediate_dims = None
    if "intermediate_z" in zs:
        kept_intermediate_dims = {}
        intermediate_zs = zs["intermediate_z"]
        mlp_z = zs.get("mlp_z", None)
        for layer in range(len(intermediate_zs)):
            intermediate_z_layer = intermediate_zs[layer].squeeze()
            intermediate_z_layer = intermediate_z_layer.cpu().clone()
            if mlp_z is not None:
                intermediate_z_layer *= mlp_z[layer]
            kept_intermediate_dims[layer] = intermediate_z_layer.nonzero().reshape(-1).tolist()

    def prune_layer_norm(layernorm, index):
        layernorm.weight = torch.nn.parameter.Parameter(
            layernorm.weight.index_select(0, index))
        layernorm.bias = torch.nn.parameter.Parameter(
            layernorm.bias.index_select(0, index))
        layernorm.normalized_shape = (len(index),)
    
    def prune_layer_norm_llama(layer_idx, index):
        #input
        llama.layers[layer_idx].input_layernorm.weight = torch.nn.parameter.Parameter(
            llama.layers[layer_idx].input_layernorm.weight.index_select(0, index))
        llama.layers[layer_idx].input_layernorm.normalized_shape = (len(index),)
        
        #post attention
        llama.layers[layer_idx].post_attention_layernorm.weight = torch.nn.parameter.Parameter(
            llama.layers[layer_idx].post_attention_layernorm.weight.index_select(0, index))
        llama.layers[layer_idx].post_attention_layernorm.normalized_shape = (len(index),)

    def prune_layer(layer, index, dim):
        layer = prune_linear_layer(layer, index, dim=dim)
        return layer
    
    print(model, type(model), hasattr(model, "bert"))
    if hasattr(model, "bert"):

        if "hidden_z" in zs:
            hidden_zs = zs["hidden_z"]
            index = torch.LongTensor(hidden_zs.squeeze().nonzero().squeeze().tolist())
            index = index.to(model.device)

            bert.embeddings.word_embeddings.weight = torch.nn.parameter.Parameter(
                bert.embeddings.word_embeddings.weight.index_select(1, index).clone().detach())
            bert.embeddings.word_embeddings.embedding_dim = index.shape[0]
            bert.embeddings.position_embeddings.weight = torch.nn.parameter.Parameter(
                bert.embeddings.position_embeddings.weight.index_select(1, index).clone().detach())
            bert.embeddings.position_embeddings.embedding_dim = index.shape[0]
            bert.embeddings.token_type_embeddings.weight = torch.nn.parameter.Parameter(
                bert.embeddings.token_type_embeddings.weight.index_select(1, index).clone().detach())
            bert.embeddings.token_type_embeddings.embedding_dim = index.shape[0]

            prune_layer_norm(bert.embeddings.LayerNorm, index)

            for layer in range(0, 12):
                if bert.encoder.layer[layer].attention.self.query is not None:
                    bert.encoder.layer[layer].attention.self.query = \
                        prune_layer(bert.encoder.layer[layer].attention.self.query , index, dim=1)
                    bert.encoder.layer[layer].attention.self.key = \
                        prune_layer(bert.encoder.layer[layer].attention.self.key , index, dim=1)
                if bert.encoder.layer[layer].attention.self.value is not None:
                    bert.encoder.layer[layer].attention.self.value = \
                        prune_layer(bert.encoder.layer[layer].attention.self.value , index, dim=1)
                    bert.encoder.layer[layer].attention.output.dense = \
                        prune_layer(bert.encoder.layer[layer].attention.output.dense , index, dim=0)
                    prune_layer_norm(bert.encoder.layer[layer].attention.output.LayerNorm, index)
                if bert.encoder.layer[layer].intermediate.dense is not None:
                    bert.encoder.layer[layer].intermediate.dense = \
                        prune_layer( bert.encoder.layer[layer].intermediate.dense, index, dim=1)
                    bert.encoder.layer[layer].output.dense = \
                        prune_layer( bert.encoder.layer[layer].output.dense, index, dim=0)
                    prune_layer_norm(bert.encoder.layer[layer].output.LayerNorm, index)
        if 'final_mlp_hidden_z' in zs:
          
            concat_index = torch.cat([index + (i * hidden_zs.shape[0]) for i in range(4)])  # 766 * 4 = 3064
            model.bn.weight = torch.nn.Parameter(model.bn.weight[concat_index].clone())
            model.bn.bias = torch.nn.Parameter(model.bn.bias[concat_index].clone())
            model.bn.num_features_tracked = concat_index.shape[0]
            model.bn.running_mean = model.bn.running_mean[concat_index].clone()
            model.bn.running_var = model.bn.running_var[concat_index].clone()

        
    elif hasattr(model, "model"):
        if "hidden_z" in zs:
            hidden_zs = zs["hidden_z"]
            print(hidden_zs.shape)
            index = torch.LongTensor(hidden_zs.squeeze().nonzero().squeeze().tolist())
            index = index.to(model.device)
            
            llama.embed_tokens.weight = torch.nn.parameter.Parameter(llama.embed_tokens.weight.index_select(1,index).clone().detach())
            for layer in range(0,21):
                prune_layer_norm_llama(layer, index)
                
                if llama.layers[layer].self_attn.q_proj is not None:
                    llama.layers[layer].self_attn.q_proj = \
                        prune_layer(llama.layers[layer].self_attn.q_proj , index, dim=1)
                    
                    llama.layers[layer].self_attn.k_proj = \
                        prune_layer(llama.layers[layer].self_attn.k_proj , index, dim=1)
                    
                if llama.layers[layer].self_attn.v_proj is not None:
                    llama.layers[layer].self_attn.v_proj = \
                        prune_layer(llama.layers[layer].self_attn.v_proj , index, dim=1)
                    
                    llama.layers[layer].self_attn.o_proj = \
                        prune_layer(llama.layers[layer].self_attn.o_proj , index, dim=1)
                    
                    
                    
                #encoder.layers.0.mlp.gate_proj.weight', 'encoder.layers.0.mlp.up_proj.weight', 'encoder.layers.0.mlp.down_proj.weight'
                if llama.layers[layer].mlp.gate_proj is not None:
                    llama.layers[layer].mlp.gate_proj = \
                        prune_layer(llama.layers[layer].mlp.gate_proj, index, dim=1)
                    
                    llama.layers[layer].mlp.up_proj= \
                        prune_layer(llama.layers[layer].mlp.up_proj, index, dim=1)
                    
                    llama.layers[layer].mlp.down_proj= \
                        prune_layer(llama.layers[layer].mlp.down_proj, index, dim=0)
        if 'final_mlp_hidden_z' in zs:
            concat_index = torch.cat([index + (i * hidden_dims) for i in range(4)])  # 766 * 4 = 3064
            model.bn.weight = torch.nn.Parameter(model.bn.weight[concat_index].clone())
            model.bn.bias = torch.nn.Parameter(model.bn.bias[concat_index].clone())
            model.bn.num_features_tracked = concat_index.shape[0]
            model.bn.running_mean = model.bn.running_mean[concat_index].clone()
            model.bn.running_var = model.bn.running_var[concat_index].clone()
        
                    
                    

    # accommodate for different models
    if hasattr(model, "classifier"):
        if hasattr(model.classifier, "dense"):
            model.classifier.dense = prune_linear_layer(model.classifier.dense, index, dim=1)
    if hasattr(model, "cls"):
        if hasattr(model.cls, "dense"):
            model.cls.dense = prune_linear_layer(model.classifier.dense, index, dim=1)
            
    if bert is not None:
        if hasattr(bert.pooler, "dense"):
            bert.pooler.dense = prune_linear_layer(bert.pooler.dense, index, dim=1)
    if hasattr(model, "qa_outputs"):
        model.qa_outputs = prune_linear_layer(model.qa_outputs, index, dim=1)
    if getattr(model, "layer_transformation", None) is not None:
        model.layer_transformation = prune_linear_layer(model.layer_transformation, index, dim=1)
        print("layer transformation", model.layer_transformation.weight.shape)
    if getattr(model, "mha_layer_transformation", None) is not None:
        model.mha_layer_transformation = prune_linear_layer(model.mha_layer_transformation, index, dim=1)
        print("layer mha_layer_transformation", model.mha_layer_transformation.weight.shape)
    if hasattr(model, 'mlp'):
        
        if 'final_mlp_hidden_z' in zs:
            model = prune_hidden_mlp(zs, model, concat_index)
        
            print("success")
                

    if kept_intermediate_dims is not None:
        print(model, type(model), hasattr(model, "bert"))
        #print("Want to keep the intermediate dims: ", kept_intermediate_dims)
        prune_intermediate_layers(model, kept_intermediate_dims)
    
    
    for layer in range(0, model.config.num_hidden_layers):
        if hasattr(model, 'bert'):
            print("Layer:", layer)
            if bert.encoder.layer[layer].attention.self.query is not None:
                print("query:", bert.encoder.layer[layer].attention.self.query.weight.shape)
                print("key:", bert.encoder.layer[layer].attention.self.key.weight.shape)
            else:
                print("query:", None)
                print("key:", None)
            if bert.encoder.layer[layer].attention.self.value is not None:
                print("value:", bert.encoder.layer[layer].attention.self.value.weight.shape)
                print("output:", bert.encoder.layer[layer].attention.output.dense.weight.shape)
            else:
                print("value:", None)
                print("output:", None)
            if bert.encoder.layer[layer].intermediate.dense is not None:
                print("up:", bert.encoder.layer[layer].intermediate.dense.weight.shape)
                print("down:", bert.encoder.layer[layer].output.dense.weight.shape)
            else:
                print("up", None)
                print("down", None)
        else:
            if llama.layers[layer].self_attn.q_proj is not None:
                print("query:", llama.layers[layer].self_attn.q_proj.weight.shape)
                print("key:", llama.layers[layer].self_attn.k_proj.weight.shape)
            else:
                print("query:", None)
                print("key:", None)
            if llama.layers[layer].self_attn.v_proj is not None:
                print("value:", llama.layers[layer].self_attn.v_proj.weight.shape)
                print("output:", llama.layers[layer].self_attn.o_proj.weight.shape)
            else:
                print("value:", None)
                print("output:", None)
            if llama.layers[layer].mlp.gate_proj is not None:
                print("gate:",llama.layers[layer].mlp.gate_proj.weight.shape)
                print("up:", llama.layers[layer].mlp.up_proj.weight.shape)
                print("down:", llama.layers[layer].mlp.down_proj.weight.shape)
            else:
                print("gate:",None)
                print("up", None)
                print("down", None)
    print("3072:",model.mlp[0].weight.shape)
    print("1024:", model.mlp[3].weight.shape)
            

def prune_hidden_mlp(zs, model, concat_index=None):
    fin_index = torch.LongTensor(zs['final_mlp_hidden_z'].squeeze().nonzero().squeeze().tolist())

    try:
        if concat_index is not None:
            model.mlp[0] = prune_linear_layer(model.mlp[0], concat_index, dim=1) #3072
        print(f"Pruning mlp now first layer is: {model.mlp[0]}, {len(fin_index)}")
        
        '''this lineis where we pruned the 1024 in 2048->1024'''
        model.mlp[0] = prune_linear_layer(model.mlp[0], fin_index, dim=0) #1024 in 3072x1024
        print(f"Pruning mlp now first layer is: {model.mlp[0]}, {len(fin_index)}")
        model.mlp[3] = prune_linear_layer(model.mlp[3], fin_index, dim=1) #1024 in 1024x3 ebcause we pruned the 1024 from the 3072x1024 and now dims need to match between 3072x1024 and 1024x3 ot pass the ou[ut of former into latter
        print(f"Pruning mlp now first layer is: {model.mlp[3]}, {len(fin_index)}")
    except:
        return
    return model

def prune_intermediate_layers(model, keep_dims):
    print("MOL:", model)
    bert = model.bert if hasattr(model, "bert") else None
    llama = model.model if hasattr(model, "model") else None
    
    assert (hasattr(model, "model")  and llama is not None) or  (hasattr(model, "bert") and bert is not None),  (hasattr(model, "bert"))
    
    device = model.device
    for layer in keep_dims:
        if bert is not None:
            if len(keep_dims[layer]) == 0:
                bert.encoder.layer[layer].intermediate.dense = None
                bert.encoder.layer[layer].output.dense = None
            else:
                bert.encoder.layer[layer].intermediate.dense = prune_linear_layer(bert.encoder.layer[layer].intermediate.dense, index=torch.LongTensor(keep_dims[layer]).to(device), dim=0)
                bert.encoder.layer[layer].output.dense = prune_linear_layer(bert.encoder.layer[layer].output.dense, index=torch.LongTensor(keep_dims[layer]).to(device), dim=1)

        elif llama is not None:
            if len(keep_dims[layer]) == 0:
                llama.layers[layer].mlp.gate_proj = None
                llama.layers[layer].mlp.up_proj=None
                llama.layers[layer].mlp.down_proj = None
            else:
                llama.layers[layer].mlp.gate_proj = prune_linear_layer(llama.layers[layer].mlp.gate_proj, index=torch.LongTensor(keep_dims[layer]).to(device), dim=0)
                llama.layers[layer].mlp.up_proj = prune_linear_layer(llama.layers[layer].mlp.up_proj, index=torch.LongTensor(keep_dims[layer]).to(device), dim=0)
                llama.layers[layer].mlp.down_proj = prune_linear_layer(llama.layers[layer].mlp.down_proj, index=torch.LongTensor(keep_dims[layer]).to(device), dim=1)
            


def load_zs(model_path):
    if model_path.endswith("zs_llm.pt") or model_path.endswith("zs_mlp.pt"):
        zs_path = model_path
    else:
        raise Error("Must enter path to llm or mlp weights")

    if os.path.exists(zs_path):
        zs = torch.load(zs_path, map_location="cpu")
        if zs is None:
            model_path = os.path.dirname(model_path)
            l0_module = torch.load(os.path.join(model_path, "l0_module.pt"), map_location="cpu")
            zs = l0_module.forward(training=False)
        return zs
    else:
        return None

def load_pruned_model(model, weights):
    if model.model_name == 'bowman':
        model.load_state_dict(weights, strict=False)
        print(model.mlp[0])
        return model
    config = model.config
    dim_per_head = config.hidden_size // config.num_attention_heads
    zs = {}
    architecture = config.architectures[0].lower()
    

    hidden_z = torch.zeros(config.hidden_size)

    hidden_z[:config.hidden_size] = 1
        
    zs["hidden_z"] = hidden_z

    head_z = torch.zeros(config.num_hidden_layers, config.num_attention_heads)
    head_layer_z = torch.zeros(config.num_hidden_layers)
    for i in range(config.num_hidden_layers):
        remaining_heads = config.hidden_size // dim_per_head
        head_z[i, :remaining_heads] = 1
        head_layer_z[i] = 1
    zs["head_z"] = head_z
    zs["head_layer_z"] = head_layer_z

    int_z = torch.zeros(config.num_hidden_layers, config.intermediate_size)
    mlp_z = torch.zeros(config.num_hidden_layers)
    for i in range(config.num_hidden_layers):

        int_z[i, :config.intermediate_size] = 1
        mlp_z[i] = 1
    zs["intermediate_z"] = int_z
    zs["mlp_z"] = mlp_z
 
    #prune_model_with_z(zs, model)
    print(model)
    model.load_state_dict(weights, strict=False)
    

    #print("Missing ", missing, "\nUn ", unexpected)
    return model

def get_full_model_size(model_class, model_name):
    model = model_class.from_pretrained(model_name)
    model_size = calculate_parameters(model)
    return model_size


