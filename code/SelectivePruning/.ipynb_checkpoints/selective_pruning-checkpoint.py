import sys
sys.path.append("./code")
import models
import cofi.utils.cofi_utils as cofi_utils
import torch
import snli_eval
import train_utils
import pandas as pd
import csv
from selective_pruning_utils import *

def get_model(args,ckpt, train,zs=None):

    # ==== BUILD MODEL ====
    model,tok = train_utils.build_model(vocab_size=len(train.stoi), model_type=args.model_type, vocab={'stoi': train.stoi, 'itos': train.itos}, embedding_dim=300, hidden_dim=512, is_cofi=args.pruning_method=='CoFi')
    model, _=train_utils.load_model(model_type=args.model_type, train=train, ckpt=ckpt, use_pretrained_weights=True, pruning_method=args.pruning_method, device='cpu', i=0, zs=zs)
    return model,tok

def prune_neurons(model, ckpt, neurons_to_prune):
    try:
        ckpt = torch.load(ckpt, map_location='cpu')['state_dict']
    except:
        print(f" ckpt not a file")
    for neuron in neurons_to_prune:
        ckpt['mlp.0.weight'][neuron] = torch.zeros_like(ckpt['mlp.0.weight'][neuron])
        
        #print(f"Pruned neuron {neuron}")
    
    
    
    model.load_state_dict(ckpt)
    assert (torch.sum(ckpt['mlp.0.weight'][neuron])==0 for neuron in neurons_to_prune)
    return model


    

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

    parser.add_argument("--ckpt", default="BERT/models/wanda/Run0.25_5/5_Pruning_Iter/enc_76_mlp0.pth")
    parser.add_argument("--model_type", default="bert", choices=["bowman", "bert", "llama"])
    parser.add_argument("--pruning_method", default="wanda", choices=["lottery_ticket", "wanda", "CoFi"])
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    args = parse_args()
    train,_,dataloaders=train_utils.create_dataloaders(max_data=None, model_type=args.model_type, pruning_method=args.pruning_method)
    val_loader = dataloaders['val']
    
   
    
    
    root_dir='/workspace/CCE_NLI/BERT/exp/wanda/Run0.25_5/Expls/0.0%Pruned'
    #BERT
    bert_foundaational_concepts_removed_25= ['pre:tok:guy', 'pre:tag:wp', 'pre:tok:sit', 'hyp:tok:some', 'pre:tok:team', 'hyp:tok:child', 'pre:tok:some', 'hyp:tok:guy', 'pre:tok:or', 'pre:tok:fish', 'pre:tok:their', 'pre:tok:market', 'hyp:tok:player', 'pre:tok:performs', 'pre:tok:jeans', 'pre:tok:him', 'pre:tok:vendor', 'hyp:tok:other', 'hyp:tok:and', 'pre:tok:costume', 'pre:tok:middle', 'pre:tok:picture', 'hyp:tok:bench', 'pre:tok:trick', 'pre:tok:about']
    bert_foundaational_concepts_removed_43 = ['pre:tok:guy', 'pre:tok:down', 'pre:tag:wp', 'hyp:tok:stage', 'pre:tok:team', 'hyp:tok:holds', 'pre:tok:some', 'hyp:tok:while', 'pre:tok:through', 'hyp:tag:rp', 'pre:tok:or', 'hyp:tok:from', 'hyp:tok:player', 'pre:tok:jeans', 'pre:tok:all', 'hyp:tok:three', 'pre:tok:vendor', 'hyp:tok:other', 'hyp:tok:smiling', 'pre:tok:rock', 'pre:tok:costume', 'pre:tok:four', 'pre:tok:middle', 'pre:tok:holding', 'pre:tok:picture', 'pre:tok:-', 'pre:tok:helmet', 'hyp:tag:md', 'pre:tok:trick', 'pre:tok:colorful', 'pre:tok:about']
    bert_foundaational_concepts_removed_57 = ['pre:tok:guy', 'pre:tag:vbn', 'hyp:tok:stage', 'pre:tok:sit', 'hyp:tok:some', 'hyp:tok:skateboarder', 'hyp:tag:cc', 'pre:tok:front', 'hyp:tok:holds', 'pre:tok:some', 'hyp:tok:while', 'pre:tok:another', 'hyp:tok:kids', 'pre:tok:or', 'hyp:tag:vbn', 'hyp:tag:rp', 'hyp:tok:from', 'pre:tok:long', 'hyp:tok:not', 'pre:tok:jumps', 'pre:tok:market', 'pre:tok:boys', 'pre:tok:all', 'pre:tok:him', 'hyp:tok:three', 'pre:tok:suit', 'pre:tok:vendor', 'pre:tok:rock', 'hyp:tok:with', 'pre:tok:runs', 'pre:tok:during', 'pre:tok:by', 'pre:tok:costume', 'pre:tok:female', 'pre:tok:middle', 'pre:tok:picture', 'hyp:tok:bench', 'pre:tok:-', 'pre:tok:helmet', 'hyp:tag:md', 'hyp:tok:field', 'pre:tok:trick', 'pre:tok:colorful', 'pre:tok:about']
    bert_foundaational_concepts_removed_68 = ['pre:tok:guy', 'pre:tok:he', 'pre:tag:wp', 'hyp:tok:stage', 'pre:tok:talking', 'pre:tok:sit', 'pre:tag:rb', 'pre:tok:city', 'hyp:tok:asleep', 'hyp:tok:skateboarder', 'hyp:tok:party', 'pre:tok:sits', 'pre:tok:team', 'pre:tok:is', 'hyp:tok:holds', 'pre:tok:at', 'hyp:tok:on', 'pre:tok:some', 'hyp:tok:waiting', 'pre:tok:walk', 'pre:tok:air', 'hyp:tok:they', 'pre:tok:waiting', 'pre:tok:another', 'pre:tok:through', 'hyp:tok:near', 'pre:tok:fish', 'pre:tok:or', 'hyp:tag:vbn', 'hyp:tag:rp', 'hyp:tok:from', 'pre:tok:long', 'hyp:tok:not', 'pre:tok:their', 'pre:tok:jumps', 'pre:tok:ocean', 'pre:tok:horse', 'pre:tok:performs', 'pre:tok:market', 'hyp:tok:player', 'pre:tok:all', 'pre:tok:baby', 'pre:tok:group', 'hyp:tok:young', 'pre:tok:kitchen', 'pre:tok:vendor', 'hyp:tok:other', 'pre:tok:dance', 'hyp:tok:with', 'pre:tok:runs', 'pre:tok:large', 'pre:tok:during', 'pre:tok:jumping', 'hyp:tok:shirt', 'pre:tok:looking', 'pre:tok:by', 'pre:tok:for', 'pre:tok:outside', 'pre:tok:costume', 'pre:tok:four', 'pre:tok:three', 'pre:tok:middle', 'pre:tok:holding', 'pre:tok:picture', 'pre:tok:stands', 'hyp:tok:bench', 'pre:tok:tree', 'pre:tok:helmet', 'hyp:tag:md', 'hyp:tok:field', 'pre:tok:something', 'pre:tok:colorful', 'pre:tok:about']
    bert_foundaational_concepts_removed_76 = ['pre:tok:guy', 'hyp:tok:snowboarder', 'pre:tok:team', 'pre:tok:skateboard', 'hyp:tok:they', 'pre:tok:boat', 'hyp:tok:holding', 'hyp:tok:guy', 'hyp:tok:sits', 'hyp:tok:not', 'pre:tok:performs', 'hyp:tok:day', 'pre:tok:boys', 'hyp:tok:three', 'pre:tok:on', 'hyp:tok:smiling', 'pre:tok:snowboarder', 'pre:tok:for', 'pre:tok:restaurant', 'pre:tok:band', 'pre:tok:middle', 'pre:tok:walks', 'pre:tok:helmet', 'pre:tok:about', 'hyp:tok:has', 'hyp:tok:watching', 'pre:tok:car', 'hyp:tok:party', 'pre:tok:fish', 'pre:tok:horse', 'pre:tok:all', 'pre:tok:dancing', 'pre:tok:dance', 'pre:tok:jumping', 'pre:tok:by', 'pre:tok:riding', 'pre:tok:tree', 'hyp:tag:md', 'hyp:tok:field', 'pre:tok:colorful', 'hyp:tok:near', 'pre:tok:runs', 'hyp:tok:stage', 'pre:tok:from', 'hyp:tok:some', 'hyp:tok:asleep', 'pre:tok:lady', 'hyp:tok:there', 'pre:tok:walk', 'pre:tok:or', 'pre:tok:jumps', 'pre:tok:market', 'hyp:tag:prp', 'pre:tok:kitchen', 'pre:tok:vendor', 'hyp:tok:other', 'pre:tok:rock', 'hyp:tok:with', 'pre:tok:outside', 'pre:tok:microphone', 'pre:tok:four', 'pre:tok:picture', 'pre:tok:something', 'pre:tok:trick', 'hyp:tok:pool', 'hyp:tag:rb', 'hyp:tok:park', 'pre:tok:other', 'hyp:tag:rp', 'hyp:tok:from', 'pre:tok:long', 'pre:tok:snowy', 'pre:tok:their', 'hyp:tok:race', 'pre:tok:baby', 'pre:tok:group', 'pre:tok:during', 'hyp:tok:home', 'pre:tok:costume', 'hyp:tok:human', 'pre:tok:guitar', 'pre:tok:three', 'pre:tok:stands', 'pre:tok:he', 'hyp:tok:holds']

    bert_survived= ['pre:tag:nn', 'pre:tag:jj', 'pre:tag:in', 'pre:tag:.', 'pre:tag:,', 'pre:tag:cc', 'pre:tag:dt', 'oth:overlap:overlap25', 'hyp:tag:nn', 'hyp:tag:in', 'pre:tag:nns', 'pre:tok:wearing', 'pre:tok:and', 'pre:tok:while', 'hyp:tok:playing', 'pre:tok:man', 'pre:tok:woman', 'hyp:tok:woman', 'hyp:tag:.', 'hyp:tag:nns', 'pre:tok:dog', 'pre:tok:shirt', 'pre:tok:blue', 'pre:tag:vbz', 'hyp:tok:dog', 'oth:overlap:overlap50', 'hyp:tag:dt', 'hyp:tok:man', 'pre:tok:girl', 'hyp:tok:wearing', 'pre:tok:walking', 'hyp:tok:girl', 'pre:tok:as', 'hyp:tok:people', 'hyp:tag:jj', 'pre:tok:sitting', 'pre:tok:red', 'pre:tag:cd', 'hyp:tok:men', 'pre:tok:with', 'hyp:tag:vb', 'hyp:tok:walking', 'hyp:tag:vbg', 'pre:tok:black', 'hyp:tok:women', 'pre:tag:vbg', 'hyp:tag:vbp', 'pre:tag:vbp', 'hyp:tok:sitting', 'pre:tok:people', 'hyp:tag:prp$', 'hyp:tok:to', 'hyp:tok:dogs', 'pre:tok:ball', 'pre:tok:white', 'hyp:tok:swimming', 'pre:tok:boy', 'oth:overlap:overlap75', 'pre:tok:men', 'pre:tok:to', 'hyp:tok:are', 'pre:tag:prp$', 'pre:tok:in', 'pre:tok:brown', 'hyp:tok:person', 'pre:tok:swimming', 'hyp:tok:girls', 'hyp:tok:boy', 'hyp:tok:eating', 'pre:tok:girls', 'pre:tok:green', 'pre:tok:water', 'pre:tag:prp', 'hyp:tok:is', 'pre:tok:his', 'hyp:tok:sleeping', 'hyp:tag:cd', 'hyp:tok:his', 'pre:tok:blond', 'hyp:tok:bike', 'hyp:tok:game', 'pre:tok:yellow', 'pre:tok:her', 'hyp:tok:cooking', 'pre:tok:pink', 'hyp:tok:outside', 'pre:tok:dogs', 'pre:tok:soccer', 'pre:tok:young', 'pre:tok:playing', 'pre:tok:two', 'pre:tok:street', 'hyp:tok:for', 'pre:tok:orange', 'pre:tok:snow', 'pre:tok:park', 'pre:tok:table', 'pre:tok:room', 'pre:tok:beach', 'pre:tag:nnp', 'hyp:tok:her', 'pre:tok:women', 'hyp:tok:play', 'pre:tok:camera', 'pre:tok:race', 'pre:tok:sidewalk', 'pre:tok:bicycle', 'hyp:tok:nobody', 'hyp:tag:vbz', 'hyp:tok:black', 'hyp:tok:beach', 'pre:tok:performing', 'pre:tok:cooking', 'hyp:tok:children', 'pre:tok:dress', 'hyp:tok:dinner', 'hyp:tok:at', 'hyp:tok:standing', 'hyp:tok:soccer', 'pre:tag:vb', 'pre:tok:are', 'pre:tok:food', 'pre:tok:stage', 'hyp:tok:inside', 'hyp:tok:riding', 'hyp:tok:lady', 'hyp:tok:red', 'hyp:tok:water', 'pre:tok:pool', 'pre:tok:eating', 'pre:tok:person', 'hyp:tok:blue', 'pre:tok:child', 'pre:tok:little', 'pre:tok:running', 'hyp:tok:dancing', 'hyp:tok:talking', 'hyp:tok:running', 'pre:tok:sand', 'hyp:tok:old', 'pre:tok:football', 'hyp:tok:boat', 'hyp:tag:ex', 'pre:tok:crowd', 'pre:tok:singing', 'pre:tok:purple', 'pre:tok:standing', 'hyp:tag:nnp', 'pre:tok:reading', 'hyp:tok:walks', 'hyp:tok:outdoors', 'hyp:tok:car', 'pre:tok:one', 'hyp:tok:couple', 'hyp:tok:two', 'hyp:tok:in', 'hyp:tok:tall', 'pre:tok:baseball', 'pre:tok:game', 'pre:tok:working', 'pre:tok:children']
     #LLAMA   
    foundaational_concepts_removed_25 = ['pre:tok:night', 'pre:tok:down', 'hyp:tok:watching', 'pre:tok:car', 'pre:tag:vbn', 'pre:tok:next', 'pre:tok:yellow', 'pre:tok:there', 'hyp:tok:bus', 'pre:tok:front', 'pre:tok:off', 'pre:tok:shopping', 'pre:tok:body', 'pre:tok:asian', 'pre:tok:trees', 'pre:tok:crowd', 'pre:tok:gray', 'pre:tok:wall', 'pre:tok:old', 'pre:tok:shirts', 'pre:tok:light', 'pre:tok:bowling', 'pre:tok:their', 'pre:tok:jeans', 'pre:tok:into', 'pre:tok:group', 'pre:tok:orange', 'pre:tok:little', 'pre:tok:rock', 'pre:tok:pants', 'hyp:tok:with', 'hyp:tok:and', 'pre:tok:behind', 'pre:tok:by', 'hyp:tok:ocean', 'hyp:tok:crowd', 'hyp:tok:she', 'pre:tok:horses', 'pre:tok:helmet', 'hyp:tok:going', 'hyp:tok:by']
    
    foundaational_concepts_removed_43 = ['pre:tok:guy', 'pre:tok:night', 'pre:tok:down', 'pre:tag:vbn', 'pre:tok:next', 'pre:tok:yellow', 'pre:tok:there', 'pre:tok:city', 'pre:tok:front', 'pre:tok:off', 'pre:tok:shopping', 'pre:tok:is', 'pre:tok:body', 'hyp:tok:two', 'pre:tag:rp', 'hyp:tok:band', 'pre:tok:trees', 'pre:tok:crowd', 'pre:tok:green', 'pre:tok:gray', 'pre:tok:wall', 'hyp:tag:vbn', 'hyp:tok:laying', 'pre:tok:snowy', 'pre:tok:shirts', 'pre:tok:light', 'pre:tok:play', 'pre:tok:bowling', 'hyp:tag:vbd', 'hyp:tok:by', 'pre:tok:orange', 'pre:tok:competition', 'hyp:tok:drinking', 'hyp:tok:with', 'hyp:tok:and', 'pre:tok:as', 'hyp:tok:ocean', 'pre:tok:floor', 'hyp:tok:she', 'pre:tok:horses', 'hyp:tok:humans', 'hyp:tok:going', 'hyp:tok:music']
    foundaational_concepts_removed_76=['pre:tok:guy', 'hyp:tok:dancing', 'pre:tok:laying', 'pre:tok:couple', 'pre:tok:are', 'pre:tok:bench', 'pre:tag:rp', 'pre:tok:microscope', 'pre:tok:crowd', 'pre:tok:green', 'hyp:tok:holding', 'pre:tok:gray', 'hyp:tok:they', 'hyp:tok:laying', 'pre:tok:light', 'pre:tok:skateboarder', 'pre:tok:ocean', 'hyp:tok:young', 'pre:tok:looking', 'pre:tok:restaurant', 'hyp:tok:ocean', 'hyp:tok:crowd', 'pre:tok:floor', 'pre:tok:helmet', 'pre:tok:bicycle', 'pre:tok:food', 'pre:tok:older', 'hyp:tok:music', 'pre:tok:wave', 'hyp:tok:watching', 'pre:tok:car', 'pre:tok:hair', 'pre:tok:enjoying', 'hyp:tok:snow', 'pre:tok:asian', 'hyp:tok:while', 'pre:tok:trees', 'pre:tok:mountain', 'pre:tok:old', 'hyp:tok:jumping', 'hyp:tok:bed', 'pre:tok:play', 'pre:tok:horse', 'pre:tok:jeans', 'pre:tok:into', 'hyp:tok:red', 'pre:tok:orange', 'hyp:tok:and', 'hyp:tok:performing', 'pre:tok:by', 'pre:tok:reading', 'pre:tok:volleyball', 'pre:tok:night', 'pre:tok:next', 'pre:tok:from', 'pre:tok:there', 'pre:tok:off', 'pre:tok:eating', 'hyp:tok:work', 'hyp:tok:looking', 'pre:tok:wall', 'hyp:tag:vbn', 'pre:tok:bowling', 'pre:tag:prp', 'hyp:tag:vbd', 'pre:tok:rock', 'pre:tok:pants', 'pre:tok:behind', 'pre:tok:as', 'hyp:tok:someone', 'hyp:tok:beach', 'pre:tok:smiling', 'hyp:tok:she', 'hyp:tok:cooking', 'pre:tag:vbn', 'pre:tok:yellow', 'pre:tok:city', 'hyp:tok:child', 'pre:tok:shopping', 'hyp:tok:guys', 'pre:tok:snowy', 'pre:tok:one', 'pre:tok:singing', 'pre:tok:shirts', 'pre:tok:their', 'pre:tok:snow', 'pre:tok:tennis', 'pre:tok:group', 'pre:tok:little', 'pre:tok:competition', 'pre:tok:blond', 'pre:tok:large', 'pre:tok:horses', 'hyp:tok:humans', 'hyp:tok:going', 'hyp:tok:by']
    foundaational_concepts_removed_68=['pre:tok:are', 'pre:tok:skateboard', 'pre:tok:is', 'pre:tag:rp', 'hyp:tok:they', 'pre:tok:crowd', 'pre:tok:green', 'pre:tok:gray', 'hyp:tok:laying', 'pre:tok:light', 'pre:tok:ocean', 'pre:tok:sunglasses', 'pre:tok:on', 'hyp:tok:drinking', 'pre:tok:looking', 'hyp:tok:ocean', 'hyp:tok:crowd', 'hyp:tok:old', 'pre:tok:floor', 'pre:tok:helmet', 'pre:tok:bicycle', 'pre:tok:older', 'hyp:tok:music', 'pre:tok:down', 'hyp:tok:watching', 'pre:tok:car', 'pre:tok:hair', 'hyp:tok:bus', 'hyp:tok:snow', 'pre:tok:asian', 'hyp:tok:while', 'pre:tok:trees', 'pre:tok:old', 'hyp:tok:jumping', 'hyp:tok:bed', 'pre:tok:play', 'pre:tok:into', 'pre:tok:orange', 'hyp:tok:and', 'pre:tok:by', 'pre:tok:racing', 'pre:tok:night', 'pre:tok:next', 'pre:tok:from', 'pre:tok:there', 'hyp:tok:going', 'pre:tok:front', 'pre:tok:off', 'hyp:tok:looking', 'pre:tok:air', 'pre:tok:wall', 'hyp:tag:vbn', 'pre:tok:bowling', 'pre:tok:rock', 'hyp:tok:with', 'pre:tok:behind', 'pre:tok:as', 'hyp:tok:someone', 'pre:tag:vbn', 'pre:tok:sidewalk', 'pre:tok:city', 'pre:tok:shopping', 'pre:tok:body', 'hyp:tok:guys', 'hyp:tok:basketball', 'pre:tok:snowy', 'pre:tok:one', 'pre:tok:tennis', 'pre:tok:group', 'pre:tok:little', 'pre:tok:competition', 'pre:tok:blond', 'pre:tok:large', 'hyp:tok:sit', 'pre:tok:horses', 'hyp:tok:humans', 'pre:tag:prp', 'hyp:tok:by']
    foundaational_concepts_removed_57=['pre:tok:guy', 'pre:tok:night', 'hyp:tok:watching', 'pre:tok:car', 'pre:tok:next', 'pre:tag:rb', 'pre:tok:from', 'pre:tok:there', 'pre:tok:sidewalk', 'pre:tok:city', 'pre:tok:couple', 'pre:tok:front', 'pre:tok:off', 'pre:tok:shopping', 'pre:tok:is', 'pre:tok:body', 'hyp:tok:work', 'hyp:tok:guys', 'pre:tok:trees', 'hyp:tok:they', 'pre:tok:gray', 'pre:tok:wall', 'pre:tok:old', 'hyp:tag:vbn', 'hyp:tok:laying', 'pre:tok:snowy', 'pre:tok:one', 'pre:tok:light', 'pre:tok:play', 'hyp:tok:bed', 'pre:tok:bowling', 'pre:tag:prp', 'pre:tok:shirts', 'pre:tok:jeans', 'pre:tok:their', 'pre:tok:sunglasses', 'hyp:tag:vbd', 'pre:tok:orange', 'hyp:tok:young', 'pre:tok:little', 'pre:tok:competition', 'hyp:tok:drinking', 'pre:tok:pants', 'hyp:tok:with', 'pre:tok:as', 'hyp:tok:boat', 'pre:tok:by', 'hyp:tok:someone', 'pre:tok:restaurant', 'hyp:tok:ocean', 'hyp:tok:old', 'pre:tok:horses', 'pre:tok:helmet', 'hyp:tok:cooking', 'hyp:tok:going', 'hyp:tok:by']


    
    llama_surived = ['pre:tag:nn', 'pre:tag:jj', 'hyp:tag:nn', 'pre:tok:man', 'pre:tag:.', 'pre:tok:sitting', 'pre:tag:nns', 'hyp:tok:outside', 'pre:tag:in', 'hyp:tok:man', 'hyp:tag:in', 'hyp:tok:men', 'pre:tok:woman', 'pre:tok:men', 'hyp:tag:dt', 'hyp:tok:for', 'oth:overlap:overlap25', 'hyp:tok:to', 'pre:tok:boy', 'pre:tag:dt', 'pre:tok:walking', 'hyp:tok:sleeping', 'hyp:tag:vbg', 'hyp:tok:woman', 'hyp:tok:nobody', 'hyp:tok:people', 'hyp:tag:nns', 'pre:tok:girl', 'hyp:tok:wearing', 'hyp:tok:sitting', 'hyp:tok:boy', 'pre:tok:dog', 'hyp:tag:.', 'hyp:tok:walking', 'hyp:tok:person', 'pre:tok:people', 'pre:tag:vbg', 'pre:tok:sits', 'hyp:tag:prp$', 'hyp:tok:girl', 'hyp:tag:vbp', 'pre:tok:standing', 'hyp:tok:there', 'hyp:tag:ex', 'pre:tok:for', 'pre:tok:women', 'pre:tag:vbp', 'hyp:tok:women', 'pre:tok:race', 'pre:tok:sit', 'hyp:tag:jj', 'pre:tok:stage', 'hyp:tag:vb', 'hyp:tok:outdoors', 'pre:tok:running', 'pre:tok:girls', 'pre:tag:cd', 'pre:tag:vb', 'pre:tok:table', 'hyp:tok:riding', 'hyp:tok:playing', 'hyp:tok:race', 'oth:overlap:overlap50', 'pre:tok:in', 'pre:tok:person', 'pre:tok:ball', 'pre:tok:football', 'pre:tok:shirt', 'hyp:tok:his', 'hyp:tok:tall', 'pre:tok:to', 'hyp:tok:near', 'pre:tok:riding', 'hyp:tok:girls', 'pre:tok:player', 'pre:tok:boys', 'pre:tok:black', 'pre:tok:workers', 'pre:tok:and', 'hyp:tok:running', 'hyp:tok:asleep', 'hyp:tok:dogs', 'pre:tok:walk', 'pre:tok:walks', 'hyp:tok:something', 'pre:tok:wearing', 'pre:tok:male', 'hyp:tok:dog', 'hyp:tok:at', 'pre:tok:playing', 'hyp:tag:vbz', 'pre:tag:vbz', 'pre:tok:lady', 'pre:tok:beach', 'hyp:tok:on', 'pre:tok:bike', 'hyp:tok:her', 'hyp:tok:eating', 'pre:tok:at', 'pre:tok:street', 'pre:tok:red', 'hyp:tok:talking', 'pre:tok:his', 'hyp:tag:cd', 'hyp:tok:has', 'hyp:tag:rb', 'hyp:tok:standing', 'hyp:tok:game', 'hyp:tok:alone', 'pre:tag:cc', 'hyp:tok:boys', 'pre:tok:blue', 'pre:tok:pool', 'pre:tok:baseball', 'pre:tok:game', 'pre:tok:holding', 'pre:tok:runs', 'pre:tok:dogs', 'hyp:tok:is', 'pre:tok:dancing', 'pre:tok:jumping', 'pre:tok:room', 'pre:tok:her', 'hyp:tok:lady', 'hyp:tok:water', 'pre:tok:talking', 'pre:tag:,', 'pre:tok:water', 'pre:tok:boat', 'pre:tok:with', 'hyp:tok:no', 'pre:tok:working', 'pre:tag:prp$', 'pre:tok:jumps', 'pre:tok:soccer', 'pre:tok:white', 'pre:tok:outside', 'hyp:tok:shirt', 'pre:tok:young', 'hyp:tok:working', 'hyp:tok:soccer', 'hyp:tag:nnp', 'hyp:tok:walks', 'pre:tok:guitar', 'pre:tok:while', 'pre:tok:book', 'hyp:tok:next', 'hyp:tok:in', 'pre:tok:something', 'hyp:tok:sits', 'hyp:tok:white', 'pre:tok:basketball', 'hyp:tok:are', 'pre:tok:children', 'oth:overlap:overlap75', 'hyp:tok:room', 'pre:tok:performing', 'pre:tok:cooking', 'pre:tok:hat', 'pre:tok:microphone', 'hyp:tok:their', 'hyp:tok:after', 'pre:tok:grass', 'pre:tok:through', 'hyp:tok:swimming', 'pre:tok:snowboarder', 'pre:tok:band', 'pre:tok:swimming', 'hyp:tok:black', 'hyp:tag:cc', 'hyp:tag:prp', 'pre:tok:park', 'hyp:tok:bike', 'hyp:tok:smiling', 'hyp:tok:park', 'hyp:tok:inside', 'pre:tok:gentleman', 'pre:tok:cellphone', 'pre:tok:pink', 'hyp:tok:nap', 'pre:tok:kitchen', 'hyp:tok:human', 'pre:tok:child']
    
    
    baseline_removed_43=['pre:tok:guy', 'hyp:tok:run', 'hyp:tok:friend', 'hyp:tok:swing', 'pre:tok:bride', 'pre:tok:drinking', 'pre:tok:is', 'hyp:tok:two', 'hyp:tag:jjs', 'pre:tok:tracks', 'pre:tag:rp', 'hyp:tok:band', 'pre:tok:crowd', 'pre:tok:green', 'pre:tok:gray', 'hyp:tok:empty', 'hyp:tok:laying', 'pre:tok:light', 'hyp:tok:not', 'hyp:tok:day', 'hyp:tok:drinking', 'hyp:tok:ocean', 'pre:tok:drums', 'pre:tok:just', 'hyp:tok:concert', 'pre:tok:floor', 'hyp:tok:music', 'pre:tok:catch', 'pre:tok:down', 'pre:tok:forest', 'hyp:tok:police', 'hyp:tok:tv', 'hyp:tok:flying', 'hyp:tok:wife', 'pre:tok:trees', 'pre:tok:cap', 'pre:tok:cyclist', 'pre:tok:smoking', 'pre:tok:play', 'pre:tok:orange', 'pre:tok:woods', 'pre:tok:beer', 'hyp:tok:and', 'pre:tok:face', 'pre:tok:biker', 'pre:tok:lake', 'hyp:tok:tired', 'hyp:tok:friends', 'hyp:tag:md', 'pre:tok:shirtless', 'hyp:tok:lake', 'pre:tok:night', 'pre:tok:apron', 'pre:tok:next', 'pre:tok:there', 'pre:tok:front', 'pre:tok:off', 'hyp:tok:vacation', 'hyp:tok:bathroom', 'pre:tok:wall', 'hyp:tag:vbn', 'pre:tok:leaps', 'pre:tok:bowling', 'pre:tok:truck', 'hyp:tok:because', 'hyp:tag:vbd', 'hyp:tok:with', 'pre:tok:as', 'pre:tok:corner', 'pre:tok:himself', 'hyp:tok:she', 'hyp:tok:market', 'pre:tag:vbn', 'pre:tok:yellow', 'pre:tok:city', 'hyp:tok:moving', 'pre:tok:shopping', 'pre:tok:body', 'hyp:tok:sad', 'pre:tok:violin', 'pre:tok:dances', 'pre:tok:long', 'pre:tok:snowy', 'pre:tok:shirts', 'pre:tok:sink', 'pre:tok:competition', 'hyp:tok:light', 'pre:tok:it', 'pre:tok:horses', 'hyp:tok:humans', 'hyp:tok:going', 'hyp:tok:by', 'hyp:tok:chasing']
    baseline_removed_25=['hyp:tok:run', 'hyp:tok:friend', 'hyp:tok:swing', 'hyp:tag:jjs', 'pre:tok:tracks', 'pre:tok:crowd', 'pre:tok:gray', 'pre:tok:light', 'hyp:tok:not', 'hyp:tok:day', 'pre:tok:military', 'pre:tok:masks', 'hyp:tok:ocean', 'hyp:tok:win', 'hyp:tok:crowd', 'pre:tok:drums', 'pre:tok:helmet', 'pre:tok:bikes', 'hyp:tok:runs', 'pre:tok:catch', 'pre:tok:down', 'hyp:tok:watching', 'pre:tok:car', 'hyp:tok:police', 'hyp:tok:flying', 'hyp:tok:bus', 'hyp:tok:he', 'pre:tok:asian', 'pre:tok:trees', 'pre:tok:cap', 'pre:tok:cyclist', 'pre:tok:old', 'hyp:tok:fishing', 'pre:tok:jeans', 'pre:tok:into', 'pre:tok:track', 'pre:tok:orange', 'pre:tok:woods', 'pre:tok:beer', 'hyp:tok:competition', 'hyp:tok:and', 'pre:tok:face', 'pre:tok:biker', 'pre:tok:by', 'hyp:tok:eats', 'pre:tok:lake', 'hyp:tok:tired', 'hyp:tok:friends', 'hyp:tag:md', 'pre:tok:shirtless', 'hyp:tok:lake', 'pre:tok:night', 'pre:tok:apron', 'pre:tok:next', 'pre:tok:there', 'pre:tok:front', 'pre:tok:off', 'hyp:tok:vacation', 'hyp:tok:bathroom', 'hyp:tok:funny', 'pre:tok:wall', 'pre:tok:leaps', 'pre:tok:bowling', 'hyp:tok:chef', 'hyp:tok:because', 'hyp:tok:driving', 'pre:tok:rock', 'pre:tok:pants', 'hyp:tok:with', 'pre:tok:behind', 'pre:tok:corner', 'pre:tok:himself', 'hyp:tok:she', 'hyp:tok:market', 'pre:tag:vbn', 'pre:tok:yellow', 'hyp:tok:moving', 'pre:tok:shopping', 'pre:tok:body', 'pre:tok:violin', 'pre:tok:long', 'hyp:tok:male', 'pre:tok:shirts', 'pre:tok:their', 'pre:tok:sink', 'pre:tok:group', 'pre:tok:little', 'pre:tok:during', 'hyp:tok:just', 'pre:tok:it', 'pre:tok:horses', 'hyp:tok:going', 'pre:tok:son', 'hyp:tok:by']
    baseline_removed_76=['hyp:tok:cat', 'pre:tok:guy', 'hyp:tok:dancing', 'pre:tok:laying', 'hyp:tok:friend', 'hyp:tok:swing', 'pre:tok:lying', 'hyp:tok:run', 'pre:tok:couple', 'pre:tok:are', 'pre:tok:bride', 'pre:tok:bed', 'pre:tok:drinking', 'pre:tok:bench', 'hyp:tok:walk', 'hyp:tag:jjs', 'pre:tok:tracks', 'hyp:tok:family', 'pre:tag:rp', 'pre:tok:sweeping', 'pre:tok:microscope', 'pre:tok:crowd', 'pre:tok:green', 'hyp:tok:holding', 'pre:tok:gray', 'hyp:tok:they', 'pre:tok:party', 'hyp:tok:laying', 'pre:tok:light', 'pre:tok:skateboarder', 'hyp:tok:couch', 'hyp:tok:not', 'pre:tok:ocean', 'hyp:tok:horse', 'hyp:tok:day', 'hyp:tok:young', 'pre:tok:military', 'pre:tok:chef', 'pre:tok:masks', 'pre:tok:day', 'pre:tok:looking', 'pre:tok:restaurant', 'hyp:tok:ocean', 'hyp:tok:crowd', 'pre:tok:drums', 'pre:tok:just', 'hyp:tok:concert', 'pre:tok:desk', 'pre:tok:helmet', 'pre:tok:floor', 'pre:tok:bicycle', 'pre:tok:food', 'pre:tok:older', 'pre:tok:bikes', 'hyp:tok:music', 'hyp:tok:runs', 'pre:tok:wave', 'pre:tok:catch', 'pre:tok:forest', 'hyp:tok:watching', 'pre:tok:car', 'hyp:tok:police', 'pre:tok:hair', 'hyp:tok:tv', 'pre:tok:enjoying', 'hyp:tok:flying', 'hyp:tok:wife', 'hyp:tok:he', 'hyp:tok:party', 'hyp:tok:street', 'hyp:tok:snow', 'pre:tok:asian', 'hyp:tok:while', 'pre:tok:trees', 'pre:tok:cyclist', 'pre:tok:mountain', 'pre:tok:smoking', 'pre:tok:old', 'hyp:tok:jumping', 'hyp:tok:moon', 'hyp:tok:bed', 'pre:tok:play', 'pre:tok:horse', 'pre:tok:jeans', 'pre:tok:into', 'pre:tok:track', 'hyp:tok:red', 'pre:tok:orange', 'pre:tok:woods', 'pre:tok:dance', 'pre:tok:beer', 'hyp:tok:and', 'hyp:tok:performing', 'pre:tok:face', 'pre:tok:biker', 'pre:tok:frisbee', 'pre:tok:sand', 'pre:tok:by', 'pre:tok:sleeping', 'hyp:tok:eats', 'pre:tok:lake', 'pre:tok:climbing', 'pre:tok:fruit', 'pre:tok:reading', 'hyp:tok:tired', 'pre:tok:volleyball', 'hyp:tok:friends', 'hyp:tag:md', 'hyp:tok:lake', 'pre:tok:shirtless', 'hyp:tok:climbing', 'pre:tok:night', 'hyp:tok:stage', 'pre:tok:apron', 'pre:tok:next', 'pre:tok:from', 'pre:tok:there', 'hyp:tok:going', 'pre:tok:off', 'pre:tok:river', 'pre:tok:eating', 'hyp:tok:work', 'hyp:tok:vacation', 'hyp:tok:bathroom', 'hyp:tok:funny', 'hyp:tok:looking', 'pre:tok:rides', 'pre:tok:wall', 'hyp:tag:vbn', 'pre:tok:leaps', 'pre:tok:store', 'pre:tag:prp', 'pre:tok:truck', 'hyp:tok:chef', 'hyp:tok:because', 'pre:tok:bowling', 'hyp:tok:driving', 'hyp:tag:vbd', 'hyp:tok:naked', 'pre:tok:train', 'pre:tok:rock', 'pre:tok:pants', 'pre:tok:behind', 'pre:tok:as', 'hyp:tok:someone', 'pre:tok:corner', 'hyp:tok:beach', 'pre:tok:smiling', 'pre:tok:himself', 'hyp:tok:she', 'hyp:tok:cooking', 'hyp:tok:market', 'pre:tag:vbn', 'pre:tok:yellow', 'pre:tok:city', 'hyp:tok:least', 'hyp:tok:moving', 'hyp:tok:child', 'pre:tok:shopping', 'hyp:tok:sad', 'pre:tok:violin', 'pre:tok:dances', 'hyp:tok:guys', 'pre:tok:brown', 'pre:tok:long', 'hyp:tok:male', 'pre:tok:snowy', 'pre:tok:one', 'pre:tok:singing', 'pre:tok:shirts', 'pre:tok:their', 'pre:tok:snow', 'pre:tok:tennis', 'pre:tok:sink', 'pre:tok:group', 'pre:tok:little', 'pre:tok:competition', 'hyp:tok:light', 'pre:tok:blond', 'pre:tok:large', 'hyp:tok:about', 'pre:tok:during', 'hyp:tok:indoors', 'pre:tok:it', 'pre:tok:smiles', 'pre:tok:selling', 'pre:tok:horses', 'hyp:tok:humans', 'pre:tok:cap', 'pre:tok:son', 'hyp:tok:by', 'hyp:tok:chasing', 'hyp:tok:wet']
    baseline_removed_57 = ['pre:tok:guy', 'pre:tok:lying', 'hyp:tok:friend', 'hyp:tok:swing', 'hyp:tok:run', 'pre:tok:couple', 'pre:tok:is', 'hyp:tag:jjs', 'pre:tok:tracks', 'hyp:tok:they', 'pre:tok:gray', 'hyp:tok:laying', 'pre:tok:light', 'hyp:tok:not', 'pre:tok:sunglasses', 'hyp:tok:young', 'hyp:tok:drinking', 'pre:tok:masks', 'pre:tok:restaurant', 'hyp:tok:ocean', 'pre:tok:drums', 'hyp:tok:old', 'pre:tok:just', 'hyp:tok:concert', 'pre:tok:desk', 'pre:tok:helmet', 'pre:tok:catch', 'pre:tok:forest', 'hyp:tok:watching', 'pre:tok:car', 'hyp:tok:police', 'hyp:tok:tv', 'hyp:tok:flying', 'hyp:tok:wife', 'pre:tok:trees', 'pre:tok:cyclist', 'pre:tok:smoking', 'pre:tok:old', 'hyp:tok:fishing', 'hyp:tok:bed', 'pre:tok:play', 'pre:tok:jeans', 'pre:tok:track', 'pre:tok:orange', 'pre:tok:woods', 'pre:tok:beer', 'hyp:tok:competition', 'pre:tok:face', 'pre:tok:frisbee', 'pre:tok:by', 'hyp:tok:eats', 'pre:tok:lake', 'pre:tok:fruit', 'hyp:tag:md', 'pre:tok:shirtless', 'hyp:tok:lake', 'hyp:tok:climbing', 'pre:tok:night', 'hyp:tok:stage', 'pre:tok:apron', 'pre:tok:next', 'pre:tok:from', 'pre:tok:there', 'hyp:tok:going', 'pre:tok:front', 'pre:tok:off', 'pre:tok:field', 'hyp:tok:work', 'hyp:tok:bathroom', 'pre:tok:wall', 'hyp:tag:vbn', 'pre:tok:leaps', 'pre:tok:bowling', 'pre:tok:truck', 'hyp:tok:chef', 'hyp:tok:because', 'hyp:tag:vbd', 'pre:tok:pants', 'hyp:tok:with', 'pre:tok:as', 'hyp:tok:boat', 'hyp:tok:someone', 'pre:tok:corner', 'pre:tok:himself', 'hyp:tok:cooking', 'pre:tag:rb', 'pre:tok:sidewalk', 'pre:tok:city', 'hyp:tok:least', 'hyp:tok:moving', 'pre:tok:shopping', 'pre:tok:body', 'hyp:tok:sad', 'pre:tok:violin', 'pre:tok:dances', 'hyp:tok:guys', 'pre:tok:long', 'pre:tok:snowy', 'pre:tok:one', 'pre:tok:shirts', 'pre:tok:their', 'pre:tok:sink', 'pre:tok:little', 'pre:tok:competition', 'hyp:tok:light', 'hyp:tok:about', 'pre:tok:it', 'pre:tok:smiles', 'pre:tok:horses', 'pre:tag:prp', 'pre:tok:son', 'hyp:tok:by', 'hyp:tok:wet']
    baseline_removed_68 = ['pre:tok:lying', 'hyp:tok:friend', 'hyp:tok:swing', 'hyp:tok:run', 'pre:tok:are', 'pre:tok:bride', 'pre:tok:skateboard', 'pre:tok:drinking', 'pre:tok:is', 'hyp:tag:jjs', 'pre:tok:tracks', 'hyp:tok:family', 'pre:tag:rp', 'pre:tok:sweeping', 'hyp:tok:they', 'pre:tok:crowd', 'pre:tok:green', 'pre:tok:gray', 'hyp:tok:laying', 'pre:tok:light', 'hyp:tok:couch', 'pre:tok:ocean', 'pre:tok:sunglasses', 'pre:tok:military', 'pre:tok:on', 'hyp:tok:drinking', 'pre:tok:chef', 'pre:tok:masks', 'pre:tok:day', 'pre:tok:looking', 'hyp:tok:ocean', 'hyp:tok:crowd', 'pre:tok:drums', 'hyp:tok:old', 'pre:tok:just', 'hyp:tok:concert', 'pre:tok:desk', 'pre:tok:helmet', 'pre:tok:floor', 'pre:tok:bicycle', 'pre:tok:older', 'pre:tok:bikes', 'hyp:tok:music', 'hyp:tok:runs', 'pre:tok:catch', 'pre:tok:down', 'hyp:tok:watching', 'pre:tok:forest', 'hyp:tok:police', 'pre:tok:car', 'pre:tok:hair', 'hyp:tok:tv', 'hyp:tok:flying', 'hyp:tok:bus', 'hyp:tok:wife', 'hyp:tok:he', 'hyp:tok:street', 'hyp:tok:snow', 'pre:tok:asian', 'hyp:tok:while', 'pre:tok:trees', 'pre:tok:cap', 'pre:tok:cyclist', 'pre:tok:smoking', 'pre:tok:old', 'hyp:tok:fishing', 'hyp:tok:jumping', 'hyp:tok:bed', 'pre:tok:play', 'pre:tok:into', 'pre:tok:track', 'pre:tok:orange', 'pre:tok:beer', 'hyp:tok:and', 'pre:tok:face', 'pre:tok:biker', 'pre:tok:frisbee', 'pre:tok:sand', 'pre:tok:by', 'pre:tok:lake', 'pre:tok:fruit', 'pre:tok:racing', 'hyp:tok:tired', 'hyp:tok:friends', 'hyp:tag:md', 'hyp:tok:lake', 'pre:tok:shirtless', 'hyp:tok:climbing', 'hyp:tok:sleeps', 'pre:tok:night', 'hyp:tok:stage', 'pre:tok:apron', 'pre:tok:next', 'pre:tok:from', 'pre:tok:there', 'hyp:tok:going', 'pre:tok:front', 'pre:tok:off', 'hyp:tok:vacation', 'hyp:tok:wedding', 'hyp:tok:bathroom', 'hyp:tok:funny', 'hyp:tok:blue', 'hyp:tok:looking', 'pre:tok:air', 'pre:tok:rides', 'pre:tok:wall', 'hyp:tag:vbn', 'pre:tok:leaps', 'pre:tok:store', 'pre:tok:bowling', 'pre:tok:truck', 'hyp:tok:chef', 'hyp:tok:because', 'hyp:tok:driving', 'pre:tok:train', 'pre:tok:rock', 'hyp:tok:with', 'pre:tok:behind', 'pre:tok:as', 'hyp:tok:someone', 'pre:tok:corner', 'pre:tok:himself', 'hyp:tok:market', 'pre:tag:vbn', 'pre:tok:sidewalk', 'pre:tok:city', 'hyp:tok:least', 'hyp:tok:moving', 'pre:tok:shopping', 'pre:tok:body', 'hyp:tok:sad', 'pre:tok:violin', 'pre:tok:dances', 'hyp:tok:guys', 'hyp:tok:basketball', 'pre:tok:long', 'pre:tok:snowy', 'pre:tok:one', 'pre:tok:tennis', 'pre:tok:sink', 'pre:tok:group', 'pre:tok:little', 'pre:tok:competition', 'hyp:tok:light', 'pre:tok:blond', 'pre:tok:large', 'pre:tok:during', 'hyp:tok:home', 'hyp:tok:indoors', 'pre:tok:it', 'pre:tok:smiles', 'hyp:tok:sit', 'pre:tok:selling', 'pre:tok:horses', 'hyp:tok:humans', 'pre:tag:prp', 'pre:tok:son', 'hyp:tok:by', 'hyp:tok:wet']
    #test2 groups
    neuron_groups_formulas=get_all_cps_for_pi(root_dir)
    sparsity= {1:25.0, 2:43.75, 3:57.812, 4: 68.359, 5:76.27}
    cps_to_prune = {1:{'removed_found': bert_foundaational_concepts_removed_25, 'baseline_rem': baseline_removed_25},
                   2:{'removed_found': bert_foundaational_concepts_removed_43, 'baseline_rem': baseline_removed_43},
                   3:{'removed_found': bert_foundaational_concepts_removed_57, 'baseline_rem': baseline_removed_57},
                   4:{'removed_found': bert_foundaational_concepts_removed_68, 'baseline_rem': baseline_removed_68},
                   5:{'removed_found': bert_foundaational_concepts_removed_76, 'baseline_rem': baseline_removed_76},}
    print("Removing BASELINE Concepts lost to WANDA")
    for i in [1,2,3,4,5]:
        ckpt = f'LLAMA/models/wanda/Run0.25_pruneonlyenc/{i}_Pruning_Iter/model_best.pth'
        model,_ = get_model(args, ckpt, train)
        dense_val_acc = train_utils.run_eval(model, val_loader, args.model_type, args.pruning_method)
        print(f"Initial dense acc at {sparsity[i]} = ", dense_val_acc)
        #for unit in neurons_to_prune:
        
        print(f"Pruning out concepts that are lost to wanda from dense&pretrained {sparsity[i]}% ")
        neurons_to_prune = get_neurons_for_cps(cps_to_prune[i]['removed_found'], neuron_groups_formulas)
        print(f"Pruning {len(neurons_to_prune)} neurons that encompass all forgotten concepts")
        specifically_pruned_model = prune_neurons(model, ckpt, neurons_to_prune=neurons_to_prune)
        isPruned = [torch.sum(model.state_dict()['mlp.0.weight'][neuron])==0 for neuron in neurons_to_prune]
        for j in isPruned:
            assert j, 'some neuron not pruned'
        specifically_pruned_model.eval()
        specifically_pruned_model_val_acc = train_utils.run_eval(specifically_pruned_model, val_loader, args.model_type, args.pruning_method)
        print(f"Validation acc: after pruning {len(neurons_to_prune)} = {specifically_pruned_model_val_acc}")

                                    
        num_removed_found=len(cps_to_prune[i]['removed_found'])
        #concepts_to_prune = cps_to_prune[i]['removed_found'] + surived
        random_neurons_to_prune, concepts_pruned_out, _ = get_k_neurons(bert_survived, 0, neuron_groups_formulas, k=int(sparsity[i]*10.24))
        #number_survived_concepts = max(0, len(concepts_pruned_out) - num_removed_found)
        #num_foundational_removed = len(concepts_pruned_out) - number_survived_concepts
        #print(f"Pruning {len(random_neurons_to_prune)} neurons that encompass {number_survived_concepts} survived concepts and {num_foundational_removed} lost from pt&dense { concepts_pruned_out}")
        randomly_pruned_model = prune_neurons(model, ckpt, neurons_to_prune=random_neurons_to_prune)
        isPruned = [torch.sum(model.state_dict()['mlp.0.weight'][neuron])==0 for neuron in random_neurons_to_prune]
        for j in isPruned:
            assert j, 'some neuron not pruned'
        randomly_pruned_model.eval()
        randomly_pruned_model_val_acc = train_utils.run_eval(randomly_pruned_model, val_loader, args.model_type, args.pruning_method)
        print(f"Validation acc: after pruning {len(random_neurons_to_prune)} = {randomly_pruned_model_val_acc}")
        
        
    '''#test3 remov non foundationals
    foundationals={}
    for i in range(1,4):
        foundationals[i] = get_clusterwise_foundationals(root_dir, i)
    c=3
    for pi in sorted(os.listdir(root_dir)):
        if '0.0%P' in pi or 'ipynb' in pi: continue
        if '44' in pi or '4375': i=2
        if '25' in pi: i=1
        if '579' in pi or '578' in pi: i=3
        if '68' in pi:i=4
        if '76' in pi: i=5
        results={}
        
   
        folder = os.path.join(root_dir, pi)
        neuron_formulas = get_all_cps_for_pi_cluster(folder, c)
        
        mask = build_binary_mask(neuron_formulas, foundationals[c])
        
        concepts,f = get_topk_concepts(mask,len(foundationals[c]), foundationals[c])
        print(concepts,f)
        results = defaultdict(list)
        
        if args.pruning_method == 'CoFi':
            zs_path= os.path.join(args.model_type.upper(), "models", args.pruning_method, 'Run0.25_3', f'{i}_Pruning_Iter/zs.pt')
            zs = torch.load(zs_path)
            print("loded zs fro ", zs_path)
            model,tok = get_model(args, os.path.join(args.model_type.upper(), "models", args.pruning_method, 'Run0.25_3', f'{i}_Pruning_Iter'), train, zs=zs)
            def_model =cofi_utils.load_model(os.path.join(args.model_type.upper(), "models", args.pruning_method, 'Run0.25_3', f'{i}_Pruning_Iter'), model, zs=zs, encoder=tok)
            print("pruned model")
            
        pruned_model=  prune_neurons(def_model,def_model.state_dict(), neurons_to_prune=[])
        pruned_model.eval()
        
        pruned_model_val_acc = train_utils.run_eval(pruned_model, val_loader, args.model_type, args.pruning_method)
        print(f"Original Accuracy at {pi}% sparsity: (Pruning out no extra neurons) | Accuracy = {pruned_model_val_acc}")
        
        for start_concept in range(0, len(concepts)):
            #neurons_to_prune = get_neurons_for_cps(concepts, neuron_formulas)
            neurons_to_prune, concepts_used, last_cp_idx = get_k_neurons(concepts, start_concept, neuron_formulas, k=300)
            #test1: prune out indiv cps
            if args.pruning_method == 'CoFi':
                zs_path= os.path.join(args.model_type.upper(), "models", args.pruning_method, 'Run0.25_3', f'{i}_Pruning_Iter/zs.pt')
                zs = torch.load(zs_path)
                ckpt = f'/workspace/CCE_NLI/BOWMAN/models/CoFi/Run0.25_3/{i}_Pruning_Iter'
                model,tok=get_model(args,ckpt, train,zs=None)
                pruned_model = cofi_utils.load_model(ckpt, model, zs=zs, encoder=tok)
                pruned_model.eval()
        
                pruned_model_val_acc = train_utils.run_eval(pruned_model, val_loader, args.model_type, args.pruning_method)
                print(f"Pruned Accuracy at {pi}% sparsity: (Pruning out no extra neurons) | Accuracy = {pruned_model_val_acc}")
                print("pruned model")
            specifically_pruned_model = prune_neurons(pruned_model, pruned_model.state_dict(), neurons_to_prune=neurons_to_prune)
            specifically_pruned_model.eval()
            specifically_pruned_model_val_acc = train_utils.run_eval(specifically_pruned_model, val_loader, args.model_type, args.pruning_method)
            print(f"Pruning out {concepts_used} and {len(neurons_to_prune)} neurons: finall acc: {specifically_pruned_model_val_acc}")
            results['Order_in_Most_Frequent'].append(f"{start_concept+1} to {last_cp_idx}")
            results['Accuracy'].append(specifically_pruned_model_val_acc)
            results['Accuracy_Drop'].append(pruned_model_val_acc - specifically_pruned_model_val_acc)
            results['Number_Neurons_Pruned'].append(len(neurons_to_prune))
            results['Concepts_Pruned'].append(concepts_used)
            results['is_Foundational'].append(True)
        
        pd.DataFrame(results).to_csv(f"Results/pruning_foundational_concepts_{pi}.csv")
 
    

        non_foundationals=get_non_foundationals(foundationals, folder)
        neuron_formulas = get_all_cps_for_pi(folder)
        mask = build_binary_mask(neuron_formulas, non_foundationals)
        concepts, frequencys = get_topk_concepts(mask,len(non_foundationals), non_foundationals)
        neurons_to_prune = get_neurons_for_cps(concepts, neuron_formulas)


        non_found_model=  prune_neurons(pruned_model,f'/workspace/CCE_NLI/BERT/models/lottery_ticket/Run0.25_3/{i}_Pruning_Iter/model_best.pth', neurons_to_prune=neurons_to_prune)
        non_found_model.eval()

        non_found_model_valacc = train_utils.run_eval(non_found_model, val_loader, args.model_type, args.pruning_method)
        print(f"At {pi}% sparsity: Pruning out {len(concepts)} non foundational cps: {len(neurons_to_prune)} neurons: {non_found_model_valacc}")
        pd.DataFrame(results).to_csv(f"Results/pruning_non_foundational_concepts_{pi}.csv")'''
        
    
   
    
"""
Bowman: 377,595 are the only 2 neurons learning concepts (in theory as per our alg) in the 3rd cluster and they explain foundational cps so removing these neuron should drop the accuracy, and it does so slightly
        - that could mean that bowman doesnt depend so much on these high activations 
        
386,683 are the only 2 neurons that don't explain any foundational concepts (in C2 but do in C1) so removing these as per the hypothesis shouldn't change the accuracy, and it doesn't
    -if you remove only 683 u see an increase in performance, but only 386 decreased performance and both 386 & 683 no change in performance
    - 683 has 1 foundational cp in C1 and 386 has 1 in C1 so 
    - 683 fires 1376 samples in C1 386 fires for 3284 samples in C1 
            so removing 683 helping maybe means that it wasnt positively contrivuting 
    
If you remove 750, 26 which are 2 neurons explaining foundational concepts in the 2nd cluster, there is a slight performance drop
        - this is keeping all other neurons so these 2 neurons made a diff but the ones tht don't explain foundational doesn't 
        
Removing 0,12 which explain foundationals in Cluster 1 had no effect on performance, but 815,213 slightly does
        - 0,12 cover a lot of samples, but 815 and 213 don't yet that latter pair impacted accuracy. These both are at Cluster 1 (213 is activated at C1 for the most samples of the 4 neurons)
        - 750 and 377 cover 90% of the dataset and removing these causes slight drop in accuracy
so acc will drop if you remove the neurons that fire a lot (regardless of the cluster) irresective of

Removing neurons that fire a lot hurts acc (regardless of the cluster they fire at) but removing neurons that dont fire much doesnt (147 -low reach vs 172-high reach in C1)
also removing neurons that not expl foundational at a given cluster helps the acc r hurts dependng on the reach of that neuron (683 had low reach in C2 but 386 had high even tho the foundational wasnt there)


but if i remove (at 67%) the 3 non foundational cps neurons fro C1, acc dropps ([330, 492, 1010]) and thses 3 arent expld anywhere else in that iter so that means that 
    its mot that foundational concepts are the only contributors bc otherwise remving these wuld have no effect
    and removing all C3 neurons hurt the acc a lot (71.7 to 71.1) so means it does depend on these?
 
 
overall theres no significant inc/decresase in the performanve baed on pruning out neurons. meaning if you prune out all C3 neurons you'll see some change
but in terms of remoing neurons that don't define foundational I'm seeing that theres no fluctuation from if u were to remive indiv neurons at random
but if you remove non foundatonal neurons some of them result in performance improvement

like if u remove all neurons that are expainabe at a Cluster 1, the performacne drops. meaning that even tho these neurins arent explainable at the 3rd cluster they still are crtiical to the underlying knowledhe  
but if u remove C1 expl neurons that are not expl at C3, performance drop is not as bad as if you remove c1 that are expl at c3
if u remove localized expl nurons (neurons only expld in c1 or c2 or c3) perf drop not significant but if u remove cross clusteer neurons it is (c1,c2, or c1,c2,c3 or c2,c3)

so theres light arg that the model can be learning founational and thats what helps it retain acc (bc if u prune out some non foundational neurons you see a perf increase) but 
overall there isnt a significant difference.  like removing an arbitrary neuron that does expl foundational cps can cause a performance slight increase or decrease and pruning a non foundational can also result in a performane decrease
"""