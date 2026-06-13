import time 
import heapq 
import torch 
import torch.nn as nn 
from layerwrapper import WrappedGPT
import util, train_utils, prune_utils
import math
import settings
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

NUM_SAMPLES=100

def find_layers(model, module, layers=[nn.Linear, nn.LSTM], name=''):
    
    """
    Recursively find the layers of a certain type in a module.

    Args:
        module (nn.Module): PyTorch module.
        layers (list): List of layer types to find.
        name (str): Name of the module.

    Returns:
        dict: Dictionary of layers of the given type(s) within the module.
    """
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(
            model, child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res



def check_sparsity(model, args):
    if args.model_type == 'bert':
        use_cache = model.config.use_cache 
        model.config.use_cache = False 
    if args.seg == 'bert':
        layers = model.encoder.encoder.layer
    elif args.seg == 'mlp':
        layers = model.mlp
    count = 0 
    total_params = 0
    for i, layer in enumerate(layers):
        subset = find_layers(model, layer)

        sub_count = 0
        sub_params = 0
        for name in subset:
            W = subset[name].weight.data
            count += (W==0).sum().item()
            total_params += W.numel()

            sub_count += (W==0).sum().item()
            sub_params += W.numel()
        try:
            print(f"layer {i} sparsity {float(sub_count)/sub_params:.6f}")
        except:
            pass

    if args.model_type == 'bert': 
        model.config.use_cache = use_cache 
    return float(count)/total_params 

#Using this to get bert and llama inputs
def get_model_inputs(model, args, dataloader, dev, dev2, seqlen, layers):
    dtype = next(iter(model.parameters())).dtype
    
    encoder = None
    if hasattr(model, 'bert'):
        encoder = model.bert
    elif hasattr(model, "model"):
        encoder = model.model
    elif hasattr(model, 'encoder'):
        encoder = model.encoder
    inps = torch.zeros(
        (NUM_SAMPLES, seqlen, encoder.config.hidden_size), dtype=dtype, device=dev2
    )
    
    cache = {'i': 0, 'attention_mask': [], 'position_embeddings': []}

    class Catcher_BERT(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, input_ids, attention_mask, head_mask=None,
            encoder_hidden_states=None, encoder_attention_mask=None,
            past_key_value=None, output_attentions=False):
            if cache['i'] < NUM_SAMPLES:
                
                inps[cache['i']][:input_ids[0].shape[0],:] = input_ids[0].to(dev2)
                if attention_mask is None:
                    default_mask = torch.ones(
                        1,  # batch_size
                        input_ids.shape[1],  # seq_len
                        input_ids.shape[1],  # seq_len
                        device=input_ids.device
                    )
                    print("default mask shape ",  default_mask.shape)
                    cache['attention_mask'].append(default_mask)
                else:
                    cache['attention_mask'].append(attention_mask[0])
                
                cache['i'] += 1
           
                raise ValueError
            else:
                # Already collected enough samples - proceed normally
                return self.module(input_ids, **kwargs)

    class Catcher_LLAMA(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            
            if cache['i'] < NUM_SAMPLES:
                inps[cache['i']][:inp[0].shape[0], :] = inp[0].to(dev2) #because inps sotres up till the highest possible seq len but this inp might be smaller
                
                cache['attention_mask'].append(kwargs['attention_mask'][0])
               
                cache['position_embeddings'].append(kwargs['position_embeddings'])
                cache['i'] += 1
                raise ValueError
            else:
                # Already collected enough samples - proceed normally
                return self.module(inp, **kwargs)
    
    if args.model_type == 'bert':      
        layers[0] = Catcher_BERT(layers[0])
    elif args.model_type == 'llama':
        layers[0] = Catcher_LLAMA(layers[0])
        
        
    for batch in dataloader:
        if cache['i'] >= NUM_SAMPLES: break
        
        s1,s2, _= batch
        s1=s1.to(dev)
        s2=s2.to(dev)
        #print(s1['input_ids'])
        
        try:
            model.to(dev)
            model(s1,s2)
        except ValueError:
            if cache['i'] >= NUM_SAMPLES:
                break
            continue
    
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    layers[0]= layers[0].module
        
    return inps, outs, layers, cache['attention_mask'], cache['position_embeddings']


#Demo for pack padded: https://gist.github.com/HarshTrivedi/f4e7293e941b17d19058f6fb90ab0fec 
def get_inputs_bowman(model,embedder, dataloader, dtype, device):
    inps = torch.zeros((NUM_SAMPLES, NUM_SAMPLES, 300), dtype=dtype, device=device)
    lengths = torch.zeros((NUM_SAMPLES, NUM_SAMPLES), dtype=dtype, device=device)
    inps.requires_grad = False
    i=0
    for batch in dataloader:
        if i >= NUM_SAMPLES: break
        try:
            s1, s2, target = batch #s1 is longest sent x 100 since batch size is 100

            s1 = s1.to(device)
            s2 = s2.to(device)
            
            for sentence1, sentence2 in zip(s1, s2):
                if i >= NUM_SAMPLES: break
                #saving encoding s1
                sentence1 = sentence1.unsqueeze(0)
                s1_enc= model.encoder.emb(sentence1)
                s1_enc.transpose(0, 1) 
                s1length = torch.tensor([s1_enc.shape[1]]).cpu()
                
                spk_s1 = pack_padded_sequence(s1_enc, s1length, enforce_sorted=False) #removes all padding making it total-num-tokens-in-batch x 300
               
                inps[i][:spk_s1.data.shape[0],:]= spk_s1.data
                i+=1
                
                
                    
                #saving encoding s2
                sentence2 = sentence2.unsqueeze(0)
                s2_enc= model.encoder.emb(sentence2)
                s2_enc.transpose(0, 1) 
                s2length = torch.tensor([s2_enc.shape[1]]).cpu()
                spk_s2 = pack_padded_sequence(s2_enc,  s2length, enforce_sorted=False) #removes all padding making it total-num-tokens-in-batch x 300
                inps[i][:spk_s2.data.shape[0],:]= spk_s2.data
                i+=1
                
                
        except ValueError:
            print("Caught ValueError")  
        outs = torch.zeros((NUM_SAMPLES, 100, 512), dtype=dtype, device=device)
    return inps, outs, lengths

def get_bert_encodings(model, embedder, s1, s2, device):
    if hasattr(model, 'bert'):
        model = model.bert
    elif hasattr(model, 'model'):
        model = model.model
    s1enc= model(**s1.to(device))
    s1enc = s1enc.last_hidden_state[:, 0, :]
    s2enc= model(**s2.to(device))
    s2enc = s2enc.last_hidden_state[:, 0, :]
                        
    return s1enc, s2enc
def encode_sentence(model, tokens):
    outputs = model(**tokens)
    hidden = outputs.last_hidden_state #float32


    mask = tokens["attention_mask"].unsqueeze(-1).float()  # (B, T, 1)
    reps = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)


    return reps
def get_llama_encodings(model, embedder, s1, s2, device):
    if hasattr(model, 'bert'):
        model = model.bert
    elif hasattr(model, 'model'):
        model = model.model
    s1enc= encode_sentence(model, s1.to(device))
    s2enc= encode_sentence(model, s2.to(device))
   
                        
    return s1enc, s2enc

def get_inputs_mlp(model, embedder, args, dataloader, dtype, device, seqlen):
    in_feats = model.mlp[0].in_features
    out_feats = model.mlp[0].out_features
    inps = torch.zeros((NUM_SAMPLES, in_feats), dtype=dtype, device=device)
    inps.requires_grad = False
    for i, batch in enumerate(dataloader):
        if i == NUM_SAMPLES:
            break
        try:
            

            if args.model_type in ['bert', 'llama']:
                s1, s2,  target = batch #s1 is longest sent x 100 since batch size is 100
                s1 = s1.to(device)
                s2 = s2.to(device)
                if args.model_type == 'bert' :
                    s1enc, s2enc = get_bert_encodings(model, embedder, s1,s2, device)
                else:
                    s1enc, s2enc = get_llama_encodings(model, embedder, s1,s2, device)
            elif args.model_type == 'bowman':
                s1, s1l, s2, s2l, _ = batch
                s1 = s1.to(device)
                s2 = s2.to(device)
                s1enc = model.encoder(s1, s1l)
                s2enc = model.encoder(s2, s2l)

            diffs = s1enc - s2enc
            prods = s1enc * s2enc

            mlp_input = torch.cat([s1enc, s2enc, diffs, prods], 1)
            mlp_input = model.bn(mlp_input)
            #print(mlp_input.shape)
            inps[i][:mlp_input.shape[1]]= mlp_input[0]

        except ValueError:
            print("Caught ValueError")  # Debug print '''
    outs = torch.zeros(NUM_SAMPLES,out_feats)
    attention_mask = None
    position_embeddings = None
    return inps, outs, attention_mask, position_embeddings
                             
def prepare_calibration_input(model, args, seg, dataloader, embedder, layers, device):
    if args.model_type == 'bert':
        use_cache = model.bert.config.use_cache
        model.bert.config.use_cache = False
        hidden_size=model.bert.config.hidden_size
    elif args.model_type == 'llama':
        use_cache = model.model.config.use_cache
        model.model.config.use_cache = False
        hidden_size=model.model.config.hidden_size
        
    

    dtype = next(iter(model.parameters())).dtype

    model = model.to(device)
    lengths, attention_mask, position_embeddings = None, None, None
    
    # Add more debug
    print("Starting data loop")
    if seg == 'enc':
        if args.model_type in ['bert', 'llama']:
            inps, outs, layers, attention_mask, position_embeddings = get_model_inputs(model,args, dataloader, device, device, hidden_size, layers)
        elif args.model_type == 'bowman':
            inps, outs, lengths = get_inputs_bowman(model, None, dataloader, dtype, device) 
    else: #layer == mlp
        inps, outs, attention_mask, position_embeddings = get_inputs_mlp(model, embedder, args, dataloader, dtype, device, hidden_size)
    

    if args.model_type == 'bert':
        model.bert.config.use_cache = use_cache
    elif args.model_type == 'llama':
        model.model.config.use_cache = use_cache
        

    return inps, outs, layers, lengths, attention_mask, position_embeddings 


        
def get_embedder(model):
    if hasattr(model, 'bert'):
        model = model.bert
    elif hasattr(model, "model"):
        model = model.model
    elif hasattr(model, 'encoder'):
        model = model.encoder
    for name, module in model.named_modules():
        if name=='':
            continue
        return module

def pruneLSTM(W_metric1, W_metric2, subset, name, sparsity_ratio, prune_n=0, wanda_var=False):
    W_mask1 = (torch.zeros_like(W_metric1) == 1)  ## initialize a mask to be all False
    W_mask2 = (torch.zeros_like(W_metric2) == 1)  ## initialize a mask to be all False
    
    if prune_n != 0:
        # structured n:m sparsity
        for ii in range(W_metric.shape[1]):
            if ii % prune_m == 0:
                tmp = W_metric[:,ii:(ii+prune_m)].float()
                W_mask.scatter_(1,ii+torch.topk(tmp, prune_n,dim=1, largest=False)[1], True)
    else:
        sort_res_1 = torch.sort(W_metric1, dim=-1, stable=True)
        sort_res_2 = torch.sort(W_metric2, dim=-1, stable=True)

        if wanda_var:
            # wanda variant 
            tmp_metric = torch.cumsum(sort_res[0], dim=1)
            sum_before = W_metric.sum(dim=1)

            alpha = 0.4
            alpha_hist = [0., 0.8]
            W_mask, cur_sparsity = return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before)
            while (torch.abs(cur_sparsity - sparsity_ratio)>0.001) and (alpha_hist[1]-alpha_hist[0]>=0.001):
                if cur_sparsity > sparsity_ratio:
                    alpha_new = (alpha + alpha_hist[0]) / 2.0
                    alpha_hist[1] = alpha
                else:
                    alpha_new = (alpha + alpha_hist[1]) / 2.0
                    alpha_hist[0] = alpha

                alpha = alpha_new 
                W_mask, cur_sparsity = return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before)
            print(f"alpha found {alpha} sparsity {cur_sparsity:.6f}")
        else:
            # unstructured pruning
            indices = sort_res_1[1][:,:int(W_metric1.shape[1]*sparsity_ratio)]
            W_mask1.scatter_(1, indices, True)
            
            indices = sort_res_2[1][:,:int(W_metric2.shape[1]*sparsity_ratio)]
            W_mask2.scatter_(1, indices, True)
    
    return W_mask1, W_mask2

def pruneLayer(W_metric, subset, name, sparsity_ratio, prune_n=0, wanda_var=False):
    W_mask = (torch.zeros_like(W_metric) == 1).cpu()  ## initialize a mask to be all False
    
    if prune_n != 0:
        # structured n:m sparsity
        for ii in range(W_metric.shape[1]):
            if ii % prune_m == 0:
                tmp = W_metric[:,ii:(ii+prune_m)].float()
                W_mask.scatter_(1,ii+torch.topk(tmp, prune_n,dim=1, largest=False)[1], True)
    else:
        sort_res = torch.sort(W_metric, dim=-1, stable=True)
        
        indices = sort_res[1][:,:int(W_metric.shape[1]*sparsity_ratio)] #indicdeos to prune
        print("Pruning ", len(indices))
        W_mask.scatter_(1, indices, True).cpu()
            
    
    return W_mask
    
def prune_wanda(args, model, seg, dataloader, sparsity_ratio, device=torch.device("cuda:0"), prune_n = 0, prune_m = 0, wanda_var=False):
    if args.model_type == 'bert':
        use_cache = model.bert.config.use_cache 
        model.bert.config.use_cache = False
    elif args.model_type == 'llama':
        use_cache = model.model.config.use_cache 
        model.model.config.use_cache = False 
        
    dataloaders=dataloader['train']
    embedder = get_embedder(model)
    #if args.model_type == 'bowman':
        #layers = [model.encoder.rnn]  
    if args.model_type == 'bert':
        layers = model.bert.encoder.layer
    elif args.model_type == 'llama':
        layers = model.model.layers
    
    if seg == 'mlp':
        layers = model.mlp

        
    with torch.no_grad():
        inps, outs, layers_updated, lengths, attention_mask, position_embeddings = prepare_calibration_input(model, args, seg, dataloaders, embedder, layers, device)
        
        if layers_updated:
            layers = layers_updated
    
    
    nsamples = len(inps)
    
    
    for i,layer in enumerate(layers): 
        torch.cuda.empty_cache()
        #get all the layers
        subset=find_layers(model, layer)
        
        if not subset:
            continue
        
        inps, outs  = inps.to(device), outs.to(device)
        print(f"inps stats: min={inps.min():.4f}, max={inps.max():.4f}, std={inps.std():.4f}")
        
        wrapped_layers = {}
        for name in subset:
            if args.model_type == 'bowman' and seg=='enc':
                #layer_name = 'lstm'
                raise Exception("Cannot prune bowman LSTM")
            else:
                layer_name = 'linear'
            wrapped_layers[subset[name]] = WrappedGPT(subset[name], layer_name = layer_name)

        def add_batch(name):
            def tmp(_, inp, out):
                wrapped_layers[name].add_batch(inp[0].data, out.data)
            return tmp

        handles = []
        for name in wrapped_layers:
            handles.append(name.register_forward_hook(add_batch(name)))
            
        for j in range(NUM_SAMPLES):
            with torch.no_grad():
                
                input_tmp = inps[j].unsqueeze(0)
                if args.model_type == 'llama' and seg=='enc':

                    num_tokens = position_embeddings[j][0].shape[1]
                    inputs = inps[j] 
                    attn_masks = attention_mask[j]
                    print("attn mask shape ", attn_masks.unsqueeze(0).shape) #1, __ ,2048
                    print("input shaoe ", inputs[:num_tokens,:].unsqueeze(0).shape) #1, __ ,2048
                    print("position_embeddings[j][0] shaoe ", position_embeddings[j][0].shape) #1, __ ,2048
                    
                    
                    outs[j][:num_tokens,:] = layer(inputs[:num_tokens,:].unsqueeze(0), attention_mask=attn_masks.unsqueeze(0) ,position_embeddings=position_embeddings[j])[0]
                    
                    
                elif args.model_type=='bert' and seg == 'enc':
                    #print(j, attention_mask[j].shape) #1,21,21
                    num_tokens = attention_mask[j].shape[1]
                    
                    outs[j][:num_tokens,:] = layer(inps[j][:num_tokens, :].unsqueeze(0) ,attention_mask=attention_mask[j])[0]
                else:
                    outs[j] = layer(inps[j].unsqueeze(0))[0]
                    
                    
                    
        for h in handles:
            h.remove()
               
        for name in subset:
            print(f"pruning layer {name}: {subset[name]}")
            subset_value = subset[name]
            print(f"scaler_row stats: min={wrapped_layers[subset_value].scaler_row.min():.4f}, max={wrapped_layers[subset_value].scaler_row.max():.4f}, std={wrapped_layers[subset_value].scaler_row.std():.4f}")
            W_metric = torch.abs(subset_value.weight.data).cpu() * torch.sqrt(wrapped_layers[subset_value].scaler_row.reshape((1,-1))).cpu()
       
            W_mask = pruneLayer(W_metric, subset, name, sparsity_ratio=sparsity_ratio)
        
            print(f"W_metric mean kept: {W_metric[~W_mask].mean():.4f}")
            print(f"W_metric mean pruned: {W_metric[W_mask].mean():.4f}")
            subset[name].weight.data[W_mask] = 0
            
            
            
            
        #passes the inps thru the lauer to get the inputs to thenext layer
       
        for j in range(NUM_SAMPLES):
            with torch.no_grad():
                
                input_tmp = inps[j].unsqueeze(0)
                if args.model_type == 'llama' and seg=='enc':

                    num_tokens = position_embeddings[j][0].shape[1]
                    inputs = inps[j] 
                    attn_masks = attention_mask[j]
                    
                    
                    outs[j][:num_tokens,:] = layer(inputs[:num_tokens,:].unsqueeze(0), attention_mask=attn_masks.unsqueeze(0)  ,position_embeddings=position_embeddings[j])[0]
                    
                    
                elif args.model_type=='bert' and seg == 'enc':
                    #print(j, attention_mask[j].shape) #1,21,21
                    num_tokens = attention_mask[j].shape[1]
                    
                    outs[j][:num_tokens,:] = layer(inps[j][:num_tokens, :].unsqueeze(0) ,attention_mask=attention_mask[j])[0]
                else:
                    outs[j] = layer(inps[j].unsqueeze(0))[0]
                    
              
        inps, outs = outs, inps
        if seg == 'mlp':
            break
    
    if args.model_type == 'bert':
        model.bert.config.use_cache = use_cache
    elif args.model_type=='llama':
        model.model.config.use_cache = use_cache
        
        
        
    
    torch.cuda.empty_cache()
 
