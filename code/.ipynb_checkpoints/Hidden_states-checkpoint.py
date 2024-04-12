from models import BowmanEntailmentClassifier, TextEncoder
from snli_eval import predict, tokenize, parse_args,build_model

import spacy

import torch

def main(args):
    nlp = spacy.load("en_core_web_sm", disable=["parser", "tagger", "ner"])
    ckpt = torch.load(args.model)
    stoi = ckpt["stoi"]
    
    
    model = build_model(len(ckpt["stoi"]), args.model_type)
    model.load_state_dict(ckpt["state_dict"])
    
    p = "The sun came out in the morning"
    h = "The sun got mad at the moon"
    
    p_, plen = tokenize(p, nlp, stoi)
    p_=p_.unsqueeze(1)
    h_, hlen = tokenize(h, nlp, stoi)
    h_=h_.unsqueeze(1)
    hlen = torch.tensor([hlen])
    plen = torch.tensor([plen])
    
    out = model(p_,plen,h_,hlen)
    text_enc = model.get_encoder()

    cel = text_enc.get_last_cell_state(p_, plen)
    print(cel.shape)
    states = text_enc.get_states(p_, plen)
    hidden = text_enc.get_hidden(p_,plen)
    states=states.permute(1,0,2)
    hidden=hidden.permute(1,0,2)
    
    print(states, hidden)
    
    
    
if __name__ == '__main__':
    args = parse_args()
    main(args)
    
    
    