from models import BowmanEntailmentClassifier, TextEncoder
from snli_eval import predict, tokenize, parse_args
from train_utils import build_model

import spacy

import torch

def main(args):
    nlp = spacy.load("en_core_web_sm", disable=["parser", "tagger", "ner"])
    ckpt = torch.load('models/snli/6.pth')
    stoi = ckpt["stoi"]
 
    
    model = build_model(len(ckpt["stoi"]), args.model_type)
    model.load_state_dict(ckpt["state_dict"])
    model.eval() 
    p = "The sun came out in the morning"
    h = "The sun got mad at the moon"
    
    p_, plen = tokenize(p, nlp, stoi)
    p_=p_.unsqueeze(1)
    h_, hlen = tokenize(h, nlp, stoi)
    h_=h_.unsqueeze(1)
    hlen = torch.tensor([hlen])
    plen = torch.tensor([plen])
    
    
    s1enc = model.encoder(p_, plen)
    s2enc = model.encoder(h_, hlen)


    diffs = s1enc - s2enc
    prods = s1enc * s2enc

    mlp_input = torch.cat([s1enc, s2enc, diffs, prods], 1) #1x2048

    mlp_input = model.bn(mlp_input)
    mlp_input = model.dropout(mlp_input)

    weight= model.mlp[0].weight.detach().cpu()
    mask=torch.ones(weight.shape)
    mask[0][0]=0
    print("Old ", weight)
    mlp_input = model.mlp[0].weight.detach().copy_(weight*mask)
    print("New :", mlp_input)
    
    
    
if __name__ == '__main__':
    args = parse_args()
    main(args)
    
    
    