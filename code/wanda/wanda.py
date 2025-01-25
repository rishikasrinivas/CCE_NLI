import time 
import heapq 
import torch 
import torch.nn as nn 
from layerwrapper import WrappedGPT
import util, train_utils, prune_utils


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

    if args.model == 'bert': 
        model.config.use_cache = use_cache 
    return float(count)/total_params 


def get_inputs_bert(model, embedder, dataloader, dtype, device):
    #100 batches each eith 100 samples
    #so to store s1 and s2 from each batch of 100 we're stroing 2 samples
    inps = torch.zeros((NUM_SAMPLES, 100, 768), dtype=dtype, device=device)
    attention_mask = torch.zeros((NUM_SAMPLES, 100), dtype=dtype, device=device)
    inps.requires_grad = False
    i=0
    for batch in dataloader:
        if i >= NUM_SAMPLES-1: break
        try:
            s1, s1len, s2, s2len, target = batch #s1 is longest sent x 100 since batch size is 100
            s1= s1.transpose(1,0)
            s1 = s1.to(device)
            s2=s2.transpose(1,0)
            s2 = s2.to(device)
            for sentence1, sentence2 in zip(s1,s2):
                sentence1=sentence1.unsqueeze(0)
                s1_tokens = model.indices_to_bert_tokens(sentence1)
                s1_tokens = {k: v.to(device) for k, v in s1_tokens.items()}
                s1_tokens_embed = embedder(s1_tokens['input_ids'])
                inps[i][:s1_tokens_embed.shape[1], :] = s1_tokens_embed.squeeze(0)
                attention_mask[i][:s1_tokens['attention_mask'].shape[1]] = s1_tokens['attention_mask']
    
                i+=1
                
                if i >= NUM_SAMPLES-1: break
                sentence2=sentence2.unsqueeze(0)
                s2_tokens = model.indices_to_bert_tokens(sentence2)
                s2_tokens = {k: v.to(device) for k, v in s2_tokens.items()}
                s2_tokens_embed = embedder(s2_tokens['input_ids'])
                inps[i][:s2_tokens_embed.shape[1], :] = s2_tokens_embed.squeeze(0)
                attention_mask[i][:s2_tokens['attention_mask'].shape[1]] = s2_tokens['attention_mask']
    
                i+=1
                
                
        except ValueError:
            print("Caught ValueError")
    outs = torch.zeros_like(inps)
        
    return inps, outs, attention_mask, None #position_ids

#Demo for pack padded: https://gist.github.com/HarshTrivedi/f4e7293e941b17d19058f6fb90ab0fec 
def get_inputs_bowman(model,embedder, dataloader, dtype, device):
    inps = torch.zeros((NUM_SAMPLES, 100, 300), dtype=dtype, device=device)
    lengths = torch.zeros((NUM_SAMPLES, 100), dtype=dtype, device=device)
    inps.requires_grad = False
    i=0
    for batch in dataloader:
        try:
            s1, s1len, s2, s2len, target = batch #s1 is longest sent x 100 since batch size is 100

            s1= s1.transpose(1,0)
            s2= s2.transpose(1,0)
            s1 = s1.to(device)
            s2 = s2.to(device)
            
            for sentence1, sentence2 in zip(s1, s2):
                sentence1 = sentence1.unsqueeze(0)
                s1_enc= model.encoder.emb(sentence1)
                s1_enc.transpose(0, 1) 
                s1length = torch.tensor([s1_enc.shape[1]]).cpu()
                
                spk_s1 = pack_padded_sequence(s1_enc, s1length, enforce_sorted=False) #removes all padding making it total-num-tokens-in-batch x 300
                
                inps[i][:spk_s1.data.shape[0],:]= spk_s1.data
                
                i+=1
                
                if i >= NUM_SAMPLES-1: break
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
        
        #NEED TO USE LENGTHS AS NEEDED 
    return inps, outs, lengths

def get_bert_encodings(model, embedder, s1, s2, device):
    s1_tokens = model.indices_to_bert_tokens(s1.transpose(1,0))
    s1_tokens = {k: v.to(device) for k, v in s1_tokens.items()}
    s1_tokens_embed = embedder(s1_tokens['input_ids'])

    s2_tokens = model.indices_to_bert_tokens(s2.transpose(1,0))
    s2_tokens = {k: v.to(device) for k, v in s2_tokens.items()}
    s2_tokens_embed = embedder(s2_tokens['input_ids'])

    s1enc= model.encoder(**s1_tokens)
    s1enc = s1enc.last_hidden_state[:, 0, :]
    s2enc= model.encoder(**s2_tokens)
    s2enc = s2enc.last_hidden_state[:, 0, :]
                        
    return s1enc, s2enc

def get_inputs_mlp(model, embedder, args, dataloader, dtype, device):
    i=0
    in_feats = model.mlp[0].in_features
    out_feats = model.mlp[0].out_features
    num_inps = NUM_SAMPLES/100 #100 is batch size
    inps = torch.zeros((num_inps, 100, in_feats), dtype=dtype, device=device)
    inps.requires_grad = False
    for batch in dataloader:
        if i == num_inps:
            break
        try:
            s1, s1len, s2, s2len, target = batch #s1 is longest sent x 100 since batch size is 100
            s1 = s1.to(device)
            s2 = s2.to(device)

            if args.model_type == 'bert':
                s1enc, s2enc = get_bert_encodings(model, embedder, s1,s2, device)
            elif args.model_type == 'bowman':
                s1enc = model.encoder(s1, s1len)
                s2enc = model.encoder(s2, s2len)

            diffs = s1enc - s2enc
            prods = s1enc * s2enc

            mlp_input = torch.cat([s1enc, s2enc, diffs, prods], 1)

            inps[i][:mlp_input.shape[0],:]= mlp_input
            i+=1

        except ValueError:
            print("Caught ValueError")  # Debug print '''
    outs = torch.zeros(num_inps,100,out_feats)
    attention_mask = None
    position_ids = None
    return inps, outs, attention_mask, position_ids
                        
def prepare_calibration_input(model, args, seg, dataloader, embedder, device):
    if args.model_type == 'bert':
        use_cache = model.encoder.config.use_cache
        model.encoder.config.use_cache = False
    

    dtype = next(iter(model.parameters())).dtype

    model = model.to(device)
    lengths, attention_mask, position_ids = None, None, None
    
    # Add more debug
    print("Starting data loop")
    if seg == 'enc':
        if args.model_type == 'bert':
            inps, outs, attention_mask, position_ids = get_inputs_bert(model, embedder, dataloader, dtype, device)
        elif args.model_type == 'bowman':
            inps, outs, lengths = get_inputs_bowman(model, None, dataloader, dtype, device) 
    else: #layer==mlp
        inps, outs, attention_mask, position_ids = get_inputs_mlp(model, embedder, args, dataloader, dtype, device)
    

    if args.model_type == 'bert':
        model.encoder.config.use_cache = use_cache

    return inps, outs, lengths, attention_mask, position_ids 


        
def get_embedder(model):
    for name, module in model.encoder.named_modules():
        if name=='':
            continue
        return module


def prune_wanda(args, model, seg, dataloader, sparsity_ratio, device=torch.device("cuda:0"), prune_n = 0, prune_m = 0):
    if args.model_type == 'bert':
        use_cache = model.encoder.config.use_cache 
        model.encoder.config.use_cache = False 
        
    dataloaders=dataloader['val']
    print("loading calibdation data")
    
    print("dataset loading complete")
    embedder = get_embedder(model)
    with torch.no_grad():
        inps, outs, lengths, attention_mask, position_ids = prepare_calibration_input(model, args, seg, dataloaders, embedder, device)
    
    
    nsamples = len(inps)
    
    if args.model_type == 'bowman':
        layers = [model.encoder.rnn]  
    elif args.model_type == 'bert':
        layers = model.encoder.encoder.layer
    
    if seg == 'mlp':
        layers = model.mlp

    for layer in layers: 
        #get all the layers
        subset=find_layers(model, layer)
        
        if not subset:
            print("Continuing")
            continue
        print("SUBSET ", subset)
        
        
        #inps, outs, attention_mask, position_ids = inps.to(device), outs.to(device), attention_mask.to(device), position_ids.to(device)
        inps, outs  = inps.to(device), outs.to(device)
        
        wrapped_layers = {}
        for name in subset:
            wrapped_layers[subset[name]] = WrappedGPT(subset[name], layer_name = 'lstm' if args.model_type == 'bowman' and seg=='enc' else 'linear')

        def add_batch(name):
            def tmp(_, inp, out):
                if args.model_type == 'bowman' and seg == 'enc':
                    wrapped_layers[name].add_batch(inp[0].data, out[0].data)
                else:
                    wrapped_layers[name].add_batch(inp[0].data, out.data)
            return tmp

        handles = []
        for name in wrapped_layers:
            handles.append(name.register_forward_hook(add_batch(name)))
            
        for j in range(nsamples):
            with torch.no_grad():
                input_tmp = inps[j].unsqueeze(0)
                if args.model_type == 'bert' and seg=='enc':
                    outs[j] = layer(input_tmp, attention_mask=attention_mask)[0]
                else:
                    outs[j] = layer(input_tmp)[0]
                    
        for h in handles:
            h.remove()
            
            
        for name in subset:
            print(f"pruning layer {type(layer)} name {name}, {subset[name]}")
            subset_value = subset[name]
            if args.model_type == 'bowman' and seg == 'enc':
                W_metric = torch.abs(subset_value.weight_ih_l0.data) * torch.sqrt(wrapped_layers[subset_value].scaler_row.reshape((1,-1)))
            else:
                W_metric = torch.abs(subset_value.weight.data) * torch.sqrt(wrapped_layers[subset_value].scaler_row.reshape((1,-1)))

            W_mask = (torch.zeros_like(W_metric) == 1)  ## initialize a mask to be all False
            if prune_n != 0:
                # structured n:m sparsity
                for ii in range(W_metric.shape[1]):
                    if ii % prune_m == 0:
                        tmp = W_metric[:,ii:(ii+prune_m)].float()
                        W_mask.scatter_(1,ii+torch.topk(tmp, prune_n,dim=1, largest=False)[1], True)
            else:
                sort_res = torch.sort(W_metric, dim=-1, stable=True)

                if args.use_variant:
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
                    indices = sort_res[1][:,:int(W_metric.shape[1]*sparsity_ratio)]
                    W_mask.scatter_(1, indices, True)
            if args.model_type == 'bowman' and seg == 'enc':
                subset[name].weight_ih_l0.data[W_mask] = 0  ## set weights to zero 
            else:
                subset[name].weight.data[W_mask] = 0  ## set weights to zero
            
            
            
        #passes the inps thru the lauer to get the inputs to thenext layer
        for j in range(nsamples):
            input_tmp = inps[j].unsqueeze(0)
            if seg=='enc' and args.model_type=='bert':
                outs[j]= layer(input_tmp,attention_mask=attention_mask)[0]
            else:
                outs[j]=layer(input_tmp)[0]
              
        inps, outs = outs, inps
        if seg == 'mlp':
            break
    
    if args.model_type == 'bert':
        model.encoder.config.use_cache = use_cache
        
        
    
    torch.cuda.empty_cache()
 
