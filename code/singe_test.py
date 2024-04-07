from snli_eval import predict, tokenize, parse_args,build_model
from activation_utils import create_clusters, build_act_mask
import spacy
import torch

def test_one_sentence(final):
    records = search_feats(acts, states, (tok_feats, tok_feats_vocab), weights, dataset, cluster=cluster_num)
    
def main(args):
    nlp = spacy.load("en_core_web_sm", disable=["parser", "tagger", "ner"])
    ckpt = torch.load(args.model)
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
    
    out = predict(model, p, h, nlp, stoi, args)
    
    activ=model.get_final_reprs(p_, plen, h_, hlen)
    activ_ranges=create_clusters(activ,4)
    print(activ_ranges)
    act_mask=build_act_mask(activ, activ_ranges, 2)
    
    
    test_one_sentence(p_,h_)
    
    
if __name__ == "__main__":
    args = parse_args()
    main(args)
