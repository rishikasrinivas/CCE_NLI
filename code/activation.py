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

    
def extract_features(
    model,
    dataset,
    save_masks_dir,
):
    os.makedirs(save_masks_dir, exist_ok=True)
    model.eval()
    loader = DataLoader(
        dataset,
        shuffle=False,
        batch_size=32,
        collate_fn=lambda batch: pad_collate(batch, sort=False),
    )

    all_states = []
    for src, src_feats, src_multifeats, src_lengths, idx in tqdm(loader):
      
        #  words = dataset.to_text(src)
        if settings.CUDA:
            src = src.cuda()
            src_lengths = src_lengths.cuda()
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
            final_reprs = model.get_final_reprs(s1, s1len, s2, s2len)
        # Pack the sequence
        
        all_states.extend(list(final_reprs.cpu().numpy()))

    with open(f'{save_masks_dir}/final_layer_activations.pkl', 'wb') as file:
        print(f"Saved activations to {save_masks_dir}/final_layer_activations.pkl")
        pickle.dump(all_states, file)
        
    return all_states

def initiate_exp_run(args):
    os.makedirs(args.save_masks_dir, exist_ok=True)
  
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

        extract_features(
            model,
            dataset,
            f"{args.save_masks_dir}/{ckpt_dir}"
        )
    
def main():
    from data import analysis
    parser = ArgumentParser(
        description=__doc__, formatter_class=ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--model_type", default="bowman", choices=["bowman", "minimal", "bert"])   
    parser.add_argument("--save_masks_dir", default="activations/bowman/lottery_ticket/Run1")    
    parser.add_argument("--prune_metrics_dir", default="models/snli/prune_metrics/lottery_ticket/bowman/Run1")    
    parser.add_argument("--cuda", action="store_true")
    
    initiate_exp_run(parser.parse_args())
main()