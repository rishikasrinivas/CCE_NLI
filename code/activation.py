import data.snli
import os
import settings
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import pickle
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence

def pad_collate(batch, sort=True):
    src, src_feats, src_multifeats, src_len, idx = zip(*batch)
    idx = torch.tensor(idx)
    src_len = torch.tensor(src_len)
    src_pad = pad_sequence(src, padding_value=data.analysis.PAD_IDX)
    # NOTE: part of speeches are padded with 0 - we don't actually care here
    src_feats_pad = pad_sequence(src_feats, padding_value=-1)
    src_multifeats_pad = pad_sequence(src_multifeats, padding_value=-1)
    if sort:
        src_len_srt, srt_idx = torch.sort(src_len, descending=True)
        src_pad_srt = src_pad[:, srt_idx]
        src_feats_pad_srt = src_feats_pad[:, srt_idx]
        src_multifeats_pad_srt = src_multifeats_pad[:, srt_idx]
        idx_srt = idx[srt_idx]
        return (
            src_pad_srt,
            src_feats_pad_srt,
            src_multifeats_pad_srt,
            src_len_srt,
            idx_srt,
        )
    return src_pad, src_feats_pad, src_multifeats_pad, src_len, idx


def pairs(x):
    """
    (max_len, batch_size, *feats)
    -> (max_len, batch_size / 2, 2, *feats)
    """
    if x.ndim == 1:
        return x.unsqueeze(1).view(-1, 2)
    else:
        return x.unsqueeze(2).view(x.shape[0], -1, 2, *x.shape[2:])
from transformers import AutoTokenizer
    
def save_features(
    model,
    model_type,
    loader,
    save_activs_dir,
    is_cofi,
    train,
):
    all_states = []
    os.makedirs(save_activs_dir, exist_ok=True)
    model.eval()
    device='cuda' if torch.cuda.is_available() else 'cpu'
    if model_type in ['bert', 'llama']:
        itos=train.itos

        model_name = "bert-base-uncased" if model_type == 'bert' else "knowledgator/Llama-encoder-1.0B"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token  = tokenizer.eos_token 
        converted_val_batches = []
        for src, src_feats, src_multifeats, src_lengths, idx in tqdm(loader):
            #  words = dataset.to_text(src)
            src = src.to(device)
            src_lengths = src_lengths.to(device)
            # Memory bank - hidden states for each step
            with torch.no_grad():
                # Combine q/h pairs
                src_one = src.squeeze(2)
                src_one_comb = pairs(src_one)
                src_lengths_comb = pairs(src_lengths)


                s1 = src_one_comb[:, :, 0]
                s1len = src_lengths_comb[:, 0]
                s2 = src_one_comb[:, :, 1]
                s2len = src_lengths_comb[:, 1]

                s1_indices, s2_indices = s1.cpu().numpy().T, s2.cpu().numpy().T
                s1_sentences = [" ".join([itos.get(idx, "") for idx in row if idx not in (0, 1)]) for row in s1_indices]
                s2_sentences = [" ".join([itos.get(idx, "") for idx in row if idx not in (0, 1)]) for row in s2_indices]
                s1_tokenized = tokenizer(s1_sentences, return_tensors="pt", padding=True, truncation=True)
                s2_tokenized = tokenizer(s2_sentences, return_tensors="pt", padding=True, truncation=True)
                
                s2_tokenized = {k: v.to(device) for k, v in s2_tokenized.items()}
                s1_tokenized = {k: v.to(device) for k, v in s1_tokenized.items()}
                
                if is_cofi:
                    result= {
                        "pre_input_ids": s1_tokenized["input_ids"].to(device),
                        "pre_attention_mask": s1_tokenized["attention_mask"].to(device),
                        "hyp_input_ids": s2_tokenized["input_ids"].to(device),
                        "hyp_attention_mask": s2_tokenized["attention_mask"].to(device),
                       
                    }
                    final_reprs = model.get_final_reprs(**result)
                else:
                    final_reprs = model.get_final_reprs(s1_tokenized, s2_tokenized)
                
            all_states.extend(list(final_reprs.cpu().numpy()))
    else:
        for src, src_feats, src_multifeats, src_lengths, idx in tqdm(loader):
      
            #  words = dataset.to_text(src)
            src = src.to(device)
            src_lengths = src_lengths.to(device)
            # Memory bank - hidden states for each step
            with torch.no_grad():
                # Combine q/h pairs
                src_one = src.squeeze(2)
                src_one_comb = pairs(src_one)
                src_lengths_comb = pairs(src_lengths)


                s1 = src_one_comb[:, :, 0]
                s1len = src_lengths_comb[:, 0]
                s2 = src_one_comb[:, :, 1]
                s2len = src_lengths_comb[:, 1]
    

                if is_cofi:
                    
                    results = {'s1': s1, 's1len': s1len, 's2':s2, 's2len': s2len, 'labels':0}
                    final_reprs = model.get_final_reprs(**results)
                else:
                    final_reprs = model.get_final_reprs(s1, s1len, s2, s2len)
            # Pack the sequence

            all_states.extend(list(final_reprs.cpu().numpy()))

    with open(f'{save_activs_dir}/final_layer_activations.pkl', 'wb') as file:
        print(f"Saved activations to {save_activs_dir}/final_layer_activations.pkl")
        pickle.dump(all_states, file)
        
    return all_states

def initiate_exp_run(args):
    os.makedirs(args.save_activs_dir, exist_ok=True)
  
    for ckpt_dir in os.listdir(args.prune_metrics_dir):
        if not ckpt_dir[0].isdigit(): continue
        ckpt = os.path.join(args.prune_metrics_dir, ckpt_dir, "model_best.pth")
        print(f"Loading model from {ckpt}")
        model, dataset = data.snli.load_for_analysis(
            ckpt,
            settings.DATA,
            model_type=args.model_type,
            cuda=args.cuda
        )
        
        model.cuda()

        save_features(
            model,
            dataset,
            f"{args.save_activs_dir}/{ckpt_dir}"
        )
    
def main():
    from data import analysis
    parser = ArgumentParser(
        description=__doc__, formatter_class=ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--model_type", default="bowman", choices=["bowman", "minimal", "bert"])   
    parser.add_argument("--save_activs_dir", default="BOWMAN/activations/wanda/Run1")    
    parser.add_argument("--prune_metrics_dir", default="BOWMAN/models/wanda/Run1")    
    parser.add_argument("--cuda", action="store_true")
    
    initiate_exp_run(parser.parse_args())
#main()

