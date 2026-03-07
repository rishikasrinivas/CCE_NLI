"""
Train a bowman et al-style SNLI model
"""


import os
import torch
import torch.optim as optim
import torch.nn as nn
import pickle
from torch.utils.data import DataLoader
from data.snli import SNLI, pad_collate
from contextlib import nullcontext
from tqdm import tqdm
import numpy as np
from collections import defaultdict
import spacy
import pandas as pd
from transformers import AutoConfig, AutoTokenizer
import settings
import models
import util
import train_utils
import data.snli
from cofi.utils.utils import calculate_parameters
from activation_utils import compute_activ_ranges, create_clusters, build_act_mask, active_neurons, build_masks
from data.DataLoading import load_masks
from cofi.utils.cofi_utils import load_model
def predict(model, premise, hypothesis, nlp, stoi, args):
    pre, prelen = tokenize(premise, nlp, stoi)
    hyp, hyplen = tokenize(hypothesis, nlp, stoi)

    # unbatch
    pre = pre.unsqueeze(1)
    prelen = torch.tensor([prelen])
    hyp = hyp.unsqueeze(1)
    hyplen = torch.tensor([hyplen])

    if args.cuda:
        pre = pre.cuda()
        prelen = prelen.cuda()
        hyp = hyp.cuda()
        hyplen = hyplen.cuda()

    with torch.no_grad():
        logits = model(pre, prelen, hyp, hyplen)
        #  reprs = model.get_final_reprs(pre, prelen, hyp, hyplen)
        #  print(reprs[0, 39])
    pred = logits.squeeze(0).argmax().item()
    predtxt = data.snli.LABEL_ITOS[pred]
    return predtxt


def tokenize(text, nlp, stoi):
    toks = [t.lower_ for t in nlp(text)]
    ns = [stoi.get(t, stoi["UNK"]) for t in toks]
    return torch.tensor(ns), len(ns)


def from_stdin():
    while True:
        pre_raw = input("Premise: ")
        hyp_raw = input("Hypothesis: ")
        yield pre_raw, hyp_raw


def from_file(fpath):
    with open(fpath, "r") as f:
        lines = list(f)

    lines = [l.strip() for l in lines]
    lines = [l for l in lines if l]
    lines = [l for l in lines if not l.startswith("#")]

    if len(lines) % 2 != 0:
        raise RuntimeError("uneven src/hyp")

    for i in range(0, len(lines), 2):
        pre_raw = lines[i]
        hyp_raw = lines[i + 1]
        yield pre_raw, hyp_raw

def get_percent_pruned(model):
    final_weights = model.mlp[0].weight.detach().cpu().numpy()
    return 1 - ((final_weights.shape[0]*2048) + (final_weights.shape[0] * 3)) /((1024*2048)+(1024*3))
    final_weights_pruned= np.round(100*torch.where(torch.tensor(final_weights) == 0,1,0).sum().item()/(model.mlp[0].weight.shape[0]*model.mlp[0].weight.shape[1]), 3)
    return final_weights_pruned


def main(args):
    print("using weights from ", args.ckpt)
    nlp = spacy.load("en_core_web_sm", disable=["parser", "tagger", "ner"])
    ckpt = torch.load(args.ckpt, map_location = 'cuda' if settings.CUDA else 'cpu')
    
    train,_,dataloaders=train_utils.create_dataloaders(max_data=None, model_type=args.model_type, pruning_method=args.pruning_method)
    # ==== BUILD MODEL ====
    model,_ = train_utils.build_model(vocab_size=len(train.stoi), model_type=args.model_type, vocab={'stoi': train.stoi, 'itos': train.itos}, embedding_dim=300, hidden_dim=512, is_cofi=args.pruning_method=='CoFi')
   
    val_loader = dataloaders['val']
    accs = {}
    
    def fill_inputs_with_zs(zs, inputs):
        for key in zs:
            inputs[key] = zs[key]
        return inputs
    path_to_experiment=os.path.join("/workspace/CCE_NLI",args.model_type.upper(), 'exp', args.pruning_method, args.filename )
    for folder in os.listdir(args.root_dir):
        if '.ipynb' in folder: continue
        if '.pt' in folder: continue
        if '.csv' in folder: continue
        final_layer_activations = []
        neuron_acc = defaultdict(dict)
        masks_dir = os.path.join(path_to_experiment, folder)
        try:
            if args.pruning_method == 'CoFi':
                if args.model_type in ['bert', 'llama']:
                    #tokenizer = AutoTokenizer.from_pretrained(os.path.join(args.root_dir, f"0_Pruning_Iter"), trust_remote_code=True)
                    tokenizer=None
                else:
                    tokenizer = model.encoder
                

                if folder == '0_Pruning_Iter':
                    zs=None
                else:
                    zs=torch.load(os.path.join(args.root_dir, folder,"zs.pt"))

                pruned_model = load_model(os.path.join(args.root_dir, folder), model, zs,tokenizer, train_data=train, ckpt=os.path.join(args.root_dir, folder, 'model_best.pth'))
                pruned_model.eval()

                if settings.CUDA:
                    pruned_model.cuda()
                if args.model_type in ['bert', 'llama']:
                    pruned_model_size = calculate_parameters(pruned_model)
                    final_weights_pruned = 1 - (pruned_model_size / 1042921475) 
                else:
                    final_weights_pruned = get_percent_pruned(pruned_model)
                print("sparsity=", final_weights_pruned)
      
                all_preds = []
                all_targets = []

                # CORRECTED: Added conditional logic for batch handling
                for batch in dataloaders['val']:

                    if torch.cuda.is_available():
                        #batch = fill_inputs_with_zs(zs, batch)
                        if settings.CUDA:
                            batch = {k: v.to('cuda') for k, v in batch.items()}
                        targets = batch['labels']


                    batch_size = targets.shape[0]

                    with torch.no_grad():
                        logits = pruned_model(**batch)
                        final_layer_activations.append(pruned_model.get_final_reprs(**batch)) #may be list of tensors that i can stack where each tesnor is batchx1024 ?
                        

                    preds = logits[1][2].argmax(1)
                    all_preds.append(preds.cpu().numpy())
                    all_targets.append(targets.cpu().numpy())

                all_preds = np.concatenate(all_preds, 0)
                all_targets = np.concatenate(all_targets, 0)
                
          
            else:
                torch.cuda.empty_cache()
                if '.ipy' in folder or not folder[0].isdigit(): continue
                model.load_state_dict(torch.load(os.path.join(args.root_dir, folder, 'model_best.pth'))['state_dict'])
                all_preds = []
                all_targets = []
                model.eval()

                if settings.CUDA:
                    model = model.cuda()
                if args.model_type=='bowman':
                    for (s1, s1len, s2, s2len, targets) in val_loader:
                        if settings.CUDA:
                            s1 = s1.cuda()
                            s1len = s1len.cuda()
                            s2 = s2.cuda()
                            s2len = s2len.cuda()

                        with torch.no_grad():
                            logits = model(s1, s1len, s2, s2len)
                            final_layer_activations.append(model.get_final_reprs(s1, s1len, s2, s2len)) #may be list of tensors that i can stack where each tesnor is batchx1024 ?


                        preds = logits.argmax(1)

                        all_preds.append(preds.cpu().numpy())
                        all_targets.append(targets.cpu().numpy())
                else:
                    for s1, s2, targets in val_loader:
                        s1={k:v.cuda() for k,v in s1.items()}
                        s2={k:v.cuda() for k,v in s2.items()}

                        with torch.no_grad():
                            logits = model(s1, s2)
                            final_layer_activations.append(model.get_final_reprs(s1, s2)) #may be list of tensors that i can stack where each tesnor is batchx1024 ?

                        preds = logits.argmax(1)

                        all_preds.append(preds.cpu().numpy())
                        all_targets.append(targets.cpu().numpy())


                all_preds = np.concatenate(all_preds, 0)
                all_targets = np.concatenate(all_targets, 0)

            acc = (all_preds == all_targets)
            cw_predicted=defaultdict(list)
            for i,j in enumerate(acc):
                if j:
                    cw_predicted['correct'].append(i)
                else:
                    cw_predicted['wrong'].append(i)
            correct_len = len(cw_predicted['correct'])
            wrongs_len = len(cw_predicted['wrong'])
            if correct_len < wrongs_len:
                cw_predicted['correct'].extend([-1]*(wrongs_len-correct_len))
            else:
                cw_predicted['wrong'].extend([-1]*(correct_len-wrongs_len))
            
            
            pd.DataFrame(cw_predicted).transpose().to_csv(f"{path_to_experiment}/Prediction_CW_{folder}.csv")

            
         
            '''#then here ill have all the activitons in shape 10000x1024 so i can


            final_layer_activations = torch.cat([i for i in final_layer_activations], dim=0)
            masks_saved=False
            if not masks_saved:
                activations= final_layer_activations.cpu().t() #to get it to 10000x1024 if needed
                activation_ranges, dead_neur = create_clusters(activations,settings.NUM_CLUSTERS)
                masks = build_masks(activations, activation_ranges, settings.NUM_CLUSTERS, path_to_experiment) #how many ones per mask
            else:
                acts=[]
                for cluster_num in range(1,4):
                    if f"Cluster{cluster_num}masks.pt" in os.listdir(masks_dir):
                        acts.append(torch.load(f"{masks_dir}/Cluster{cluster_num}masks.pt").bool().numpy())
                masks = torch.tensor(acts)
            #then activs should be a  [mask c1, mask c2, mask c3]
            #then based on shape might bea gain 1024x10000 or somth so for each nueron find wher mask ==1 and 
            print("Mask shape ", masks.shape)
            for cluster, mask in enumerate(masks):


                for num, neuron in enumerate(mask):
                    accuracy=0
                    activ_samples = np.where(neuron)[0]
                    #print(f"Activs {neuron} {activ_samples}")

                    correctly_pred = np.where(acc)[0]
                    
                    #print("pred ", correctly_pred)
                    for i in activ_samples:
                        if i in correctly_pred:
                            
                            accuracy += 1

                    neuron_acc[num][f'{cluster+1}'] = accuracy/(len(activ_samples))
            pd.DataFrame(neuron_acc).transpose().to_csv(f"{path_to_experiment}/Neuron_accs_{folder}.csv")'''

        except Exception as e:
            print(e)
    #pd.DataFrame({'folder':accs.keys(), 'accs':accs.values()}).to_csv(f"{args.root_dir}/accuracy.csv")


   


    # ==== INTERACTIVE ====



def parse_args():
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

    parser = ArgumentParser(
        description=__doc__, formatter_class=ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--data",
        default="test.txt",
        help="Data to eval interactively (pairs of sentences); use - for stdin",
    )
    
    
    parser.add_argument("--root_dir", default="/workspace/CCE_NLI/BERT/models/wanda/Run0.25_5/")
    parser.add_argument("--ckpt", default="BERT/models/lottery_ticket/Run0.25_5/0_Pruning_Iter/model_best.pth")
    parser.add_argument("--model_type", default="bert", choices=["bowman", "bert", "llama"])
    parser.add_argument("--filename", default='Run0.25_5')
    parser.add_argument("--pruning_method", default="wanda", choices=["lottery_ticket", "wanda", "CoFi"])
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--eval_data_path", default="data/snli_1.0/")
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()



if __name__ == "__main__":
    args = parse_args()
    main(args)
