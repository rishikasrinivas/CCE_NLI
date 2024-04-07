from snli_eval import predict, tokenize, parse_args,build_model
from activation_utils import create_clusters, build_act_mask
import spacy
import torch

def test_one_sentence():
    model, dataset = data.snli.load_for_analysis(
        settings.MODEL,
        settings.DATA,
        model_type=settings.MODEL_TYPE,
        cuda=settings.CUDA,
    )
  
    # Last model weight
    if settings.MODEL_TYPE == "minimal":
        weights = model.mlp.weight.t().detach().cpu().numpy()
    else:
        weights = model.mlp[-1].weight.t().detach().cpu().numpy()

    print("Extracting features")
    toks, states, feats, idxs = extract_features(
        model,
        dataset,
    )

    print("Extracting sentence token features")
    tok_feats, tok_feats_vocab = to_sentence(toks, feats, dataset)
    
    settings.NEURONS = [15, 19, 1023, 203]
    single_neuron(tok_feats, tok_feats_vocab,states,feats, weights, dataset)
    #default(tok_feats, tok_feats_vocab,states,feats, weights, dataset)
    
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
    
    
    test_one_sentence(p_,h_, model, args.data)
    
    
if __name__ == "__main__":
    args = parse_args()
    main(args)
