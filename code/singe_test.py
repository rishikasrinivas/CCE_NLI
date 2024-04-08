from snli_eval import predict, tokenize, parse_args,build_model
from activation_utils import create_clusters, build_act_mask
import spacy
import torch


def load_tok_feats():
    path = "data/tok_feats.csv"
    arr_loaded = np.loadtxt(path, delimiter=',')
    return arr_loaded

def load_vocab():
    path = "data/tok_feats_vocab.txt"
    with open(path, "r") as f:
        d = f.read()
    vocab = {'itos': [], 'stoi': []}
    d= d[10:]
    
    i = 0
  
    while (d[i] != '}'):
        quote_num = 0
        key = ""
        val = ""
        while (d[i] != ':'):
            key += d[i]
            i+=1
        while (quote_num != 2):
            if d[i] == "'" or d[i] == '"':
                quote_num +=1
            val += d[i]
            i+=1
    
        print(key,val)
        i +=2
        pair = [int(key.strip()), val]
        vocab['itos'].append(pair[0])
        if i == 25:
            break
    print(vocab['itos'])
        
def test_one_sentence(acts,states,weights,tok_feats,tok_feats_vocab, cluster_num):
    records = search_feats(acts, states, (tok_feats, tok_feats_vocab), weights, None, cluster=cluster_num)
    
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
    
    weights = model.mlp[-1].weight.t().detach().cpu().numpy()
    
    #load_tok_feats()
    load_vocab()
    
    #test_one_sentence(act_mask,activ,weights,tok_feats,tok_feats_vocab, cluster_num=2)
    
    
if __name__ == "__main__":
    args = parse_args()
    main(args)
