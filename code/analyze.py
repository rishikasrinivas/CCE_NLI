
#!/usr/bin/env python
# -*- coding: utf-8 -*-
#todo: make clusters
from __future__ import unicode_literals

import multiprocessing as mp
import os
from collections import Counter, defaultdict

import numpy as np
#import onmt.opts as opts
import pandas as pd
import torch
#from onmt.utils.parse import ArgumentParser
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence
from torch.utils.data import DataLoader
from tqdm import tqdm
from scipy.spatial.distance import cdist
from sklearn.metrics import precision_score, recall_score

import formula as FM
from formula import BinaryNode
import settings
import util
from vis import report, pred_report
import data
import data.snli
import data.analysis
from activation_utils import compute_activ_ranges, create_clusters, build_act_mask, active_neurons, build_masks
from data.DataLoading import load_masks
import snli_eval
GLOBALS = {}

def get_feat_vec_for_concept(concept_num, feats, vocab):
    print("Feature name: ", vocab["itos"][concept_num])
    return feats[:,concept_num]
    
def save_with_acts(preds, acts, fname):
    preds_to_save = preds.copy()
    for i in range(acts.shape[1]):
        preds_to_save[str(i)] = acts[:, i] * 1
    preds_to_save.to_csv(fname, index=False)


def load_vecs(path):
    vecs = []
    vecs_stoi = {}
    vecs_itos = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            tok, *nums = line.split(" ")
            nums = np.array(list(map(float, nums)))

            assert tok not in vecs_stoi
            new_n = len(vecs_stoi)
            vecs_stoi[tok] = new_n
            vecs_itos[new_n] = tok
            vecs.append(nums)
    vecs = np.array(vecs)
    return vecs, vecs_stoi, vecs_itos


# Load vectors
VECS, VECS_STOI, VECS_ITOS = load_vecs(settings.VECPATH)


NEIGHBORS_CACHE = {}


def get_neighbors(lemma):
    """
    Get neighbors of lemma given glove vectors.
    """
    if lemma not in VECS_STOI:
        # No neighbors
        return []
    if lemma in NEIGHBORS_CACHE:
        return NEIGHBORS_CACHE[lemma]
    lemma_i = VECS_STOI[lemma]
    lvec = VECS[lemma_i][np.newaxis]
    dists = cdist(lvec, VECS, metric="cosine")[0]
    # first dist will always be the vector itself
    nearest_i = np.argsort(dists)[1 : settings.EMBEDDING_NEIGHBORHOOD_SIZE + 1]
    nearest = [VECS_ITOS[i] for i in nearest_i]
    NEIGHBORS_CACHE[lemma] = nearest
 
    return nearest


def get_mask(feats, f, dataset, feat_type):
    """
    Serializable/global version of get_mask for multiprocessing
    """
    # Mask has been cached
    '''if f.mask is not None:
        return f.mask
'''
    if isinstance(f, FM.And):
        masks_l = get_mask(feats, f.left, dataset, feat_type)
        masks_r = get_mask(feats, f.right, dataset, feat_type)
        return masks_l & masks_r
    elif isinstance(f, FM.Or):
        masks_l = get_mask(feats, f.left, dataset, feat_type)
        masks_r = get_mask(feats, f.right, dataset, feat_type)
        return masks_l | masks_r
    elif isinstance(f, FM.Not):
        masks_val = get_mask(feats, f.val, dataset, feat_type)
        return 1 - masks_val
    elif isinstance(f, FM.Neighbors):
        if feat_type == "word":
            # Neighbors can only be called on Lemma Leafs. Can they be called on
            # ORs of Lemmas? (NEIGHBORS(A or B))? Is this equivalent to
            # NEIGHBORS(A) or NEIGHBORS(B)?
            # (When doing search, you should do unary nodes that apply first,
            # before looping through binary nodes)
            # Can this only be done on an atomic leaf? What are NEIGHBORS(
            # (1) GET LEMMAS belonging to the lemma mentioned by f;
            # then search for other LEMMAS; return a mask that is 1 for all of
            # those lemmas.
            # We can even do NEIGHBORS(NEIGHBORS) by actually looking at where the
            # masks are 1s...but I wouldskip that for now
            # FOR NOW - just do N nearest neighbors?
            # TODO: Just pass in the entire dataset.
            # The feature category should be lemma
            # Must call neighbors on a leaf
            assert isinstance(f.val, FM.Leaf)
            ci = dataset.fis2cis[f.val.val]
            assert dataset.citos[ci] == "lemma"

            # The feature itself should be a lemma
            full_fname = dataset.fitos[f.val.val]
            assert full_fname.startswith("lemma:")
            # Get the actual lemma
            fname = full_fname[6:]
            # Get neighbors in vector space
            neighbors = get_neighbors(fname)
            if neighbors == []:
                return np.zeros(feats.shape[0], dtype=bool)
            
            # Turn neighbors into candidate feature names
            neighbor_fnames = set([f"lemma:{word}" for word in neighbors])
            # Add the original feature name
            neighbor_fnames.add(full_fname)
            # Convert to indices if they exist
            neighbors = [
                dataset.fstoi[fname]
                for fname in neighbor_fnames
                if fname in dataset.fstoi
            ]
            print("nigh mk", np.isin(feats["onehot"][:, ci].shape))
            return np.isin(feats["onehot"][:, ci], neighbors)
        else:
            assert isinstance(f.val, FM.Leaf)
            fval = f.val.val
            fname = dataset["itos"][fval]
            part, fword = fname.split(":", maxsplit=1)

            neighbors = get_neighbors(fword)
            if neighbors == []:
                return np.zeros(feats.shape[0], dtype=bool)
            part_neighbors = [f"{part}:{word}" for word in neighbors]
            neighbor_idx = [
                dataset["stoi"][word]
                for word in part_neighbors
                if word in dataset["stoi"]
            ]
            neighbor_idx.append(fval)
            neighbor_idx = np.array(list(set(neighbor_idx)))
           
            neighbors_mask = np.logical_or.reduce(feats[:, neighbor_idx], 1)
            print("nigh mk", neighbors_mask.shape)
            return neighbors_mask
    elif isinstance(f, FM.Leaf):
        if feat_type == "word":
            # Get category
            ci = dataset.fis2cis[f.val]
            cname = dataset.fis2cnames[f.val]
            if dataset.ctypes[cname] == "multi":
                # multi is in n-hot tensor shape, so we just return the column
                # corresponding to the correct feature
                midx = dataset.multi2idx[f.val]
                return feats["multi"][:, midx]
            else:
                return feats["onehot"][:, ci] == f.val
        else:
            return feats[:, f.val]
    else:
        raise ValueError("Most be passed formula")


    
def iou(a, b):
    intersection = (a & b).sum()
    union = (a | b).sum()
    return intersection / (union + np.finfo(np.float32).tiny)


def get_max_ofis(states, feats, dataset):
    """
    Get maximally activated open feats
    """
    max_order = np.argsort(states)[::-1]
    sel_ofeats = []
    for ocname in dataset.ocnames:
        ci = dataset.cstoi[ocname]
        ofeats = feats["onehot"][:, ci]
        max_ofeats = ofeats[max_order]
        max_ofeats = max_ofeats[max_ofeats != 0]
        # pd preserves order
        unique_ofeats = pd.unique(max_ofeats)
        sel_ofeats.extend(unique_ofeats[: settings.MAX_OPEN_FEATS])
    return sel_ofeats


# Category-specific composition operators
# are tuples of the shape (op, do_negate)
OPS = defaultdict(
    list,
    {
        "all": [(FM.Or, False), (FM.And, False), (FM.And, True)],
        "lemma": [(FM.Neighbors, False)],
        # WordNet synsets. For now just do hypernyms? Note: for hypernyms - how far
        # up to go? Go too far = activates for all synsets. Too low = ?
        #  'synset': [(FM.Hypernym, False)],
        # NOTE: Will beam search even work? Can I even do "compounds"? I.e. if I
        # have synset OR synset, will I ever explore synset OR hyponyms(synset)?
        # ALSO: don't forget glove vectors
    },
)
import re

def get_concept(formula, dataset):
    """
        gets the concept associated with the formula (assuming the formula is a single number rn
        )
    """
    inds = re.findall("[0-9]+",str(formula))
    c = ""
    if len(inds) >= 2:
        for i in inds:
            c += dataset['itos'][int(i)] + " "
        return c
    return []
    
import csv  
def write_to_file(unit, file, col_names, col_vals):
    if not os.path.isfile(file):
        with open(file, "w") as fp:
            wr = csv.writer(fp, dialect='excel')
            wr.writerow(col_names)
    else:
        with open(file, "a") as fp:
            wr = csv.writer(fp, dialect='excel')
            wr.writerow(col_vals)
            
def calculate_act_mask_align_index(unit, formula, cluster, acts, masks):
    #masks = masks.reshape(-1,1)
    samples_entailing_formula = np.where(masks == 1) #sample indices where concept in sample
  
    #print("For neuron ", unit, "\nConcept: ", concept, "\nAct mask is: ", acts)
    samples_where_neuron_activs = np.where(acts==1)
    sample_nums_commonTo_act_and_mask = np.intersect1d(samples_entailing_formula, samples_where_neuron_activs)
    

    write_to_file(unit, f"Cluster{cluster}SamplesFiringPerConcept.csv",["formula", "samples_entailing_formula", "num_samples_entailing_formula"], [concept, formula, samples_entailing_formula, len(samples_entailing_formula[0])] )
                 
    
   


def compute_iou(unit, cluster, formula, acts, feats, dataset, feat_type="word", sentence_num=None):
    masks = get_mask(feats, formula, dataset, feat_type) #10,000x1 saying if the formula is in the sample'
    # Cache mask

    
    
    formula.mask = masks
    
    
    
    concept=get_concept(formula,dataset)
  
    '''if (len(concept) >= 2):
        if sentence_num is None: #if not running this on 1 sentence at a time
            calculate_act_mask_align_index(unit,formula, cluster, run, concept, acts, masks)'''

    if settings.METRIC == "iou":
        #if running w only 1 sentence iou would be.1 or 0, acts will be 1,1 (this is act for each neuron so 1 sentence and.1 neuon) mask is also 1x1
        comp_iou = iou(masks, acts)
        if comp_iou == 1:
            calculate_act_mask_align_index(unit,formula, cluster, concept, acts, masks)
       # if (concept != -1):
           # write_to_file(unit, f"Run{run}Cluster{cluster}IOUs.csv", concept, formula, comp_iou, "")
    elif settings.METRIC == "precision":
        comp_iou = precision_score(masks, acts)
    elif settings.METRIC == "recall":
        comp_iou = recall_score(masks, acts)
    else:
        raise NotImplementedError(f"metric: {settings.METRIC}")
    comp_iou = (settings.COMPLEXITY_PENALTY ** (len(formula) - 1)) * comp_iou
    
    return comp_iou

#call this for each activ range from search_feats
def compute_best_sentence_iou(args):
    (unit,cluster) = args

    print("Processsing neuron ", unit)
    acts = GLOBALS["acts"][:,unit]
    #acts reprseent states in activ range
    #for each neuron identify the samples where acts in col# neuron#==1
    if acts.sum() == 0:
        return { #if the neuron is dead dont run expls on it
            "unit": unit,
            "best": (FM.Leaf(0),0.0),
            "best_noncomp": (FM.Leaf(0),0.0),
        }

    feats = GLOBALS["feats"]
    #print("FEATS. ", feats.shape) #10,000rows each row holds num concepts saying true if concept at the index is in the sample else false  
    dataset = GLOBALS["dataset"] #holds each concept ex: hyp:tok:dog


    feats_to_search = list(range(feats.shape[1]))
    formulas = {}
    for fval in feats_to_search:
        formula = FM.Leaf(fval)
        formulas[formula] = compute_iou(unit, cluster,
            formula, acts, feats, dataset, feat_type="sentence"
        )
        
        for op, negate in OPS["lemma"]:
            # FIXME: Don't evaluate on neighbors if they don't exist
            new_formula = formula
            if negate:
                new_formula = FM.Not(new_formula)
                
            #handling if neightbors dont exist --added
            new_formula = op(new_formula)
            
            new_iou = compute_iou(unit, cluster,
                new_formula, acts, feats, dataset, feat_type="sentence"
            )
            
            formulas[new_formula] = new_iou
            
    nonzero_iou = [k.val for k, v in formulas.items() if v > 0]
    formulas = dict(Counter(formulas).most_common(settings.BEAM_SIZE))
    best_noncomp = Counter(formulas).most_common(1)[0]
 
    
    #identifying the most common formula associated with a neuron then applying and/or/not on each neighbor until reaching
    # formula length of MAX Length
    for i in range(settings.MAX_FORMULA_LENGTH - 1):
        new_formulas = {}
        for formula in formulas:
            # Generic binary ops
            for feat in nonzero_iou:
                for op, negate in OPS["all"]:
                    if not isinstance(feat, FM.F):
                        new_formula = FM.Leaf(feat)
                    else:
                        new_formula = feat
                    if negate:
                        new_formula = FM.Not(new_formula)
                    new_formula = op(formula, new_formula)
                    new_iou = compute_iou(unit, cluster,
                        new_formula, acts, feats, dataset, feat_type="sentence"
                    )
                    new_formulas[new_formula] = new_iou

        formulas.update(new_formulas)
        # Trim the beam
        formulas = dict(Counter(formulas).most_common(settings.BEAM_SIZE))

    best = Counter(formulas).most_common(1)[0]

    return {
        "unit": unit,
        "best": best,
        "best_noncomp": best_noncomp,
    }

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
):
    loader = DataLoader(
        dataset,
        shuffle=False,
        batch_size=32,
        collate_fn=lambda batch: pad_collate(batch, sort=False),
    )

    all_srcs = []
    all_states = []
    all_feats = []
    all_multifeats = []
    all_idxs = []
    
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
        all_srcs.extend(list(np.transpose(src_one_comb.cpu().numpy(), (1, 2, 0))))
        
        all_feats.extend(
            list(np.transpose(pairs(src_feats).cpu().numpy(), (1, 2, 0, 3)))
        )
        all_multifeats.extend(
            list(np.transpose(pairs(src_multifeats).cpu().numpy(), (1, 2, 0, 3)))
        )
        all_states.extend(list(final_reprs.cpu().numpy()))
        all_idxs.extend(list(pairs(idx).cpu().numpy()))

    all_feats = {"onehot": all_feats, "multi": all_multifeats}
    return all_srcs, all_states, all_feats, all_idxs


def search_feats(acts, states, feats, weights, dataset, cluster,sentence_num =None, run=None, save_dir=None):
    if save_dir is None:
        return "Invalid save_dir"
    
    rfile = os.path.join(save_dir, "result.csv")
    #if os.path.exists(rfile):
        #print(f"Loading cached {rfile}")
        #return pd.read_csv(rfile).to_dict("records")

    # Set global vars
    if cluster is not None:
        print("cluster ",cluster)
    GLOBALS["acts"] = acts #should be 10,000x1024 
    assert acts.shape[0]==10000 and acts.shape[1]==1024
    GLOBALS["states"] = states  
     #feats: 10000 rows 40087 cols
        # each row is a sentence and each col says if concept at col is in sent
        #print(feats[0][sentence_num].reshape(1,-1).shape)
    GLOBALS["dataset"] = feats[1]
    feats_vocab = feats[1]
    if sentence_num != None:
        GLOBALS["feats"] = feats[0][sentence_num].reshape(1,-1)
    else:
        GLOBALS["feats"] = feats[0]
    
        
    
    
    def namer(i):
        return feats_vocab["itos"][i]

    def cat_namer(i):
        return feats_vocab["itos"][i].split(":")[0]

    def cat_namer_fine(i):
        return ":".join(feats_vocab["itos"][i].split(":")[:2])

    ioufunc = compute_best_sentence_iou

    records = []
    if settings.NEURONS is None:
        units = range(acts.shape[1])
    else:
        units = settings.NEURONS
    mp_args = [(u,cluster) for u in units]
    

    if settings.PARALLEL < 1:
        print("fake pool")
        pool_cls = util.FakePool
    else:
        print("multithread")
        pool_cls = mp.Pool

    n_done = 0
    with pool_cls(settings.PARALLEL) as pool, tqdm(
        total=len(units), desc="Units"
    ) as pbar:
        #do for each neuron and foor each range
        for res in pool.imap_unordered(ioufunc, mp_args):
            unit = res["unit"]
            
            best_lab, best_iou = res["best"]
       
            best_name = best_lab.to_str(namer, sort=True)

            best_cat = best_lab.to_str(cat_namer, sort=True) 
            best_cat_fine =  best_lab.to_str(cat_namer_fine, sort=True)
                    
            
            entail_weight = weights[unit, 0]
            neutral_weight = weights[unit, 1]
            contra_weight = weights[unit, 2]

            if best_iou > 0:
                tqdm.write(f"{unit:02d}\t{best_name}\t{best_iou:.3f}")
                write_to_file(unit, f"{save_dir}/Cluster{cluster}IOUS1024N.csv", ["unit", "best_name", "best_iou"], [unit, best_name, best_iou])
            
            r = {
                "cluster": cluster,
                "neuron": unit,
                "feature": best_name,
                "category": best_cat,
                "category_fine": best_cat_fine,
                "iou": best_iou,
                "feature_length": len(best_lab),
                "w_entail": entail_weight,
                "w_neutral": neutral_weight,
                "w_contra": contra_weight,
            }
            records.append(r)
            pbar.update()
            n_done += 1
            #if n_done % settings.SAVE_EVERY == 0:
            pd.DataFrame(records).to_csv(rfile, index=False)

        # Save progress
        if len(records) % 32 == 0:
            pd.DataFrame(records).to_csv(rfile, index=False)

    pd.DataFrame(records).to_csv(rfile, index=False)
    return records

 
def to_sentence(toks, feats, dataset, tok_feats_vocab=None):
    """
    Convert token-level feats to sentence feats
    """
    tokens = np.zeros(len(dataset.stoi), dtype=np.int64)
    encoder_uniques = []
    decoder_uniques = []
    #  both_uniques = []

    encoder_tag_uniques = []
    decoder_tag_uniques = []
    #  both_tag_uniques = []

    tag_i = dataset.cstoi["tag"]

    other_features = []
    oth_names = [
        ("overlap25", "overlap"),
        ("overlap50", "overlap"),
        ("overlap75", "overlap"),
    ]

    for pair, featpair in zip(toks, feats["onehot"]):
        pair_counts = np.bincount(pair.ravel())
        tokens[: len(pair_counts)] += pair_counts
        
        enct = np.unique(pair[0]) #get the unique concepts in premise and hyp
        dect = np.unique(pair[1])

        encu = np.setdiff1d(enct, dect)
        decu = np.setdiff1d(dect, enct)
        both = np.intersect1d(enct, dect)
        encoder_uniques.append(enct)
        decoder_uniques.append(dect)
        #  both_uniques.append(both)

        # PoS
        enctag = np.unique(featpair[0, :, tag_i]) #vector correspodnignt to tag value
        dectag = np.unique(featpair[1, :, tag_i])

        enctag = enctag[enctag != -1] #remove padding
        dectag = dectag[dectag != -1]

        #  enctagu = np.setdiff1d(enctag, dectag)
        #  dectagu = np.setdiff1d(dectag, enctag)
        #  bothtagu = np.intersect1d(enctag, dectag)

        encoder_tag_uniques.append(enctag)
        decoder_tag_uniques.append(dectag)
        #  both_tag_uniques.append(bothtagu)

        # Compute degree of overlap in tokens (gt 50%)
        overlap = len(both) / (len(encu) + len(decu) + 1e-5)
        # TODO: Do overlap at various degrees
        other_features.append(
            (
                overlap > 0.25,
                overlap > 0.5,
                overlap > 0.75,
            )
        )

    SKIP = {"a", "an", "the", "of", ".", ",", "UNK", "PAD", '"'}
 
    if tok_feats_vocab is None:
        for s in SKIP:
            if s in dataset.stoi:
                tokens[dataset.stoi[s]] = 0

        # Keep top tokens, use as features
        tokens_by_count = np.argsort(tokens)[::-1]
        tokens_by_count = tokens_by_count[: settings.N_SENTENCE_FEATS]
    
        # Create feature dict
        # Token features
        tokens_stoi = {}
        for prefix in ["pre", "hyp"]:
            for t in tokens_by_count:
                if t == 0:
                    continue
                ts = dataset.itos[t]
                t_prefixed = f"{prefix}:tok:{ts}"
                tokens_stoi[t_prefixed] = len(tokens_stoi)

            # PoS
            spec_tok=False
            for pos_i in dataset.cnames2fis["tag"]:
                pos = dataset.fitos[pos_i].lower()
                for s in SKIP:
                    if s in dataset.fitos[t]:
                        spec_tok = True
                        break
                if spec_tok:
                    continue
                assert pos.startswith("tag:")
                pos_prefixed = f"{prefix}:{pos}"
                tokens_stoi[pos_prefixed] = len(tokens_stoi)

        # Other features
        for oth, oth_type in oth_names:
            oth_prefixed = f"oth:{oth_type}:{oth}"
            tokens_stoi[oth_prefixed] = len(tokens_stoi)

        tokens_itos = {v: k for k, v in tokens_stoi.items()}

        tok_feats_vocab = {
            "itos": tokens_itos,
            "stoi": tokens_stoi,
        }
   
    # Binary mask - encoder/decoder
    token_masks = np.zeros((len(toks), len(tok_feats_vocab["stoi"])), dtype=np.bool)
    for i, (encu, decu, enctagu, dectagu, oth) in enumerate(
        zip(
            encoder_uniques,
            decoder_uniques,
            encoder_tag_uniques,
            decoder_tag_uniques,
            other_features,
        )
    ):
        # Tokens
        for prefix, toks in [("pre", encu), ("hyp", decu)]:
            for t in toks:
                ts = dataset.itos[t]
                t_prefixed = f"{prefix}:tok:{ts}"
                if t_prefixed in tok_feats_vocab["stoi"]:
                    ti = tok_feats_vocab["stoi"][t_prefixed]
                    token_masks[i, ti] = 1

        # PoS
        spec_tok=False
        for prefix, tags in [("pre", enctagu), ("hyp", dectagu)]:
            for t in tags:
                for s in SKIP:
                    if s in dataset.fitos[t]:
                        spec_tok = True
                        break
                if spec_tok:
                    continue
                ts = dataset.fitos[t].lower()
                t_prefixed = f"{prefix}:{ts}"
                assert t_prefixed in tok_feats_vocab["stoi"]
                ti = tok_feats_vocab["stoi"][t_prefixed]
                token_masks[i, ti] = 1

        # Other features
        assert len(oth) == len(oth_names)
        for (oth_name, oth_type), oth_u in zip(oth_names, oth):
            oth_prefixed = f"oth:{oth_type}:{oth_name}"
            oi = tok_feats_vocab["stoi"][oth_prefixed]
            token_masks[i, oi] = oth_u

    return token_masks, tok_feats_vocab

def quantile_features(feats):
    if settings.ALPHA is None:
        return np.stack(feats) > 0

    quantiles = get_quantiles(feats, settings.ALPHA)
    return feats > quantiles[np.newaxis]

#added
def default(tok_feats, tok_feats_vocab,states,feats, weights, dataset):
    acts = quantile_features(states)

    print("Mask search")
    records = search_feats(acts, states, (tok_feats, tok_feats_vocab), weights, dataset, cluster=None)

    print("Mask search")
    records = search_feats(acts, states, feats, weights, dataset, cluster=None)

#added
def load_sents(path):
    with open(path, 'r') as f:
        sents = f.readlines()
   
    return sents

import pickle

def searching_dead_neurons(states, threshold, model, weights, val_loader):
    activations= torch.from_numpy(np.array(states)).t()
    accs=[]
    acc=snli_eval.run_eval(model, val_loader)
    accs.append(acc)
    print(f"defult: {acc}")
    for thresh in threshold:
        activation_ranges, dead_neurons = create_clusters(activations,4)
        
        weights=model.mlp[0].weight.t().detach()
        weights[dead_neurons]= torch.zeros((1,1024)).cuda()
        model.mlp[:-1][0].weight.t().detach().copy_(weights)
        assert torch.equal(model.mlp[:-1][0].weight.t().detach()[dead_neurons],torch.zeros((len(dead_neurons),1024)).cuda())
        acc=snli_eval.run_eval(model, val_loader)
        accs.append(acc)
        print(f"{thresh}: {acc}")
    
    
def clustered_NLI(tok_feats, tok_feats_vocab,states,feats, weights, dataset, save_exp_dir, save_masks_dir, masks_saved):
    activations= torch.from_numpy(np.array(states)).t() #1024x10000
    #1st  time run this, after that dont
    print("activs shaoe ", activations.shape)
    if not masks_saved:
        print("creating masks storing in ",save_masks_dir )#check how many ones per row here
        activation_ranges, dead_neur = create_clusters(activations,4)
        pckl_file = open(f"{save_masks_dir}/Masks.pkl", "wb")
       
        pickle.dump(activation_ranges, pckl_file)
        pckl_file.close()
        
        pckl_file = open(f"{save_masks_dir}/DeadNeurons.pkl", "wb")
       
        pickle.dump(dead_neur, pckl_file)
        pckl_file.close()
        
        build_masks(activations, activation_ranges, 4, save_masks_dir) #how many ones per mask
    masks_saved = True
    for cluster_num in range(1,5): 
        if masks_saved:
            print(f"{cluster_num} found : {f'Cluster{cluster_num}masks.pt' in os.listdir(save_masks_dir)}")
            if f"Cluster{cluster_num}masks.pt" in os.listdir(save_masks_dir):
                acts = torch.load(f"{save_masks_dir}/Cluster{cluster_num}masks.pt")
                #if acts.dtype == torch.float32:
                print("converting")
                acts=acts.t()
                acts = acts.bool().numpy()

            else:
                raise Exception("cant find")
                return
        
        assert type(states)==list and len(states)==10000 and len(states[0]) == 1024 #should be list 100000 ittems ach of len 1024
     
        assert(acts.shape[0] == 10000 and acts.shape[1]==1024)
        records = search_feats(acts, states, (tok_feats, tok_feats_vocab), weights, dataset, cluster=cluster_num, run = 0, save_dir=save_exp_dir)
    return activations


#added         
def per_sent_single_neuron(tok_feats, tok_feats_vocab,states,feats, weights, dataset):
    import csv
    sents=load_sents("data/analysis/snli_1.0_dev.tok")
    i=0
    sent_num = 0
    
    with open("compExpCompareSent100.csv", "w") as fp:
        wr = csv.writer(fp, dialect='excel')
        wr.writerow(["sentences", 'cluster','neuron','feature','iou'])
        for state in states[:100]:
            sent_num += 1
            sentences=[sents[i], sents[i+1]]
            print(sentences)
            i += 2
            state = torch.tensor(np.array(state)).unsqueeze(1) 
            best_exp=[]
            activations= torch.from_numpy(state.numpy().reshape(state.shape[0]*state.shape[1], 1)) #flatten it
            activation_ranges = create_clusters(activations, 5)
            activations= torch.from_numpy(activations.numpy().reshape(1,1024)) #reform it
            for cluster_num in range(1,6):
                acts=build_act_mask(activations,activation_ranges, cluster_num)
                records = search_feats(acts, state, (tok_feats, tok_feats_vocab), weights, dataset, cluster=cluster_num,sentence_num = sent_num)
               
                for rec in records:
                    if rec['iou'] > 0.0:
                        print(rec['cluster'],rec['neuron'],rec['feature'],rec['iou'])
                        wr.writerow([sentences, rec['cluster'],rec['neuron'],rec['feature'],rec['iou']])
    return states
            

def initiate_exp_run(save_exp_dir, save_masks_dir,masks_saved, path=settings.MODEL, model_=None, dataset=None, q_ret=0):
    
    if model_==None and dataset==None:
        model, dataset = data.snli.load_for_analysis(
            path,
            settings.DATA,
            model_type=settings.MODEL_TYPE,
            cuda=settings.CUDA
        )
    else:
        model= model_
        dataset =dataset
        
        if settings.CUDA:
            model=model.to('cuda')
            

 
    # Last model weight
    if settings.MODEL_TYPE == "minimal":
        weights = model.mlp.weight.t().detach().cpu().numpy()
    else:
        weights = model.mlp[-1].weight.t().detach().cpu().numpy()
        final_weights = model.mlp[0].weight.t().detach().cpu().numpy()

    print("Extracting features")
    
    toks, states, feats, idxs = extract_features(
        model,
        dataset
    )
    
    if q_ret==1:
        return states, final_weights
   
    print("Extracting sentence token features")
    
    tok_feats, tok_feats_vocab = to_sentence(toks, feats, dataset)
    
    acts = clustered_NLI(tok_feats, tok_feats_vocab,states,feats, weights, dataset, save_exp_dir, save_masks_dir, masks_saved=masks_saved)
    return acts, final_weights
from data.snli import SNLI
def main():
    os.makedirs(settings.RESULT, exist_ok=True)
    model, dataset = data.snli.load_for_analysis(
            settings.MODEL,
            settings.DATA,
            model_type=settings.MODEL_TYPE,
            cuda=settings.CUDA
        )
    
    weights= model.mlp[0].weight.t().detach().cpu().numpy()
    
    states, weights = initiate_exp_run(save_exp_dir='code/', save_masks_dir='code/',masks_saved=False, q_ret=1)
    
    ckpt = torch.load(settings.MODEL)
    val = SNLI(
        "data/snli_1.0/",
        "dev",
        vocab=(ckpt["stoi"], ckpt["itos"]),
        unknowns=True,
    )
    val_loader = DataLoader(
        val,
        batch_size=100,
        shuffle=False,
        pin_memory=True,
        num_workers=0,
        collate_fn=data.snli.pad_collate,
    )
    searching_dead_neurons(states, [i for i in range(0,100)], model, weights, val_loader)
    
    
    print("Load predictions")
    mbase = os.path.splitext(os.path.basename(settings.MODEL))[0]
    dbase = os.path.splitext(os.path.basename(settings.DATA))[0]
    predf = f"data/analysis/preds/{mbase}_{dbase}.csv"
    # Add the feature activations so we can do correlation
    preds = pd.read_csv(predf)

    save_with_acts(preds, acts, os.path.join(settings.RESULT, "preds_acts.csv"))

    print("Visualizing features")
    from vis import sentence_report

    sentence_report.make_html(
        records,
        # Features
        toks,
        states,
        (tok_feats, tok_feats_vocab),
        idxs,
        preds,
        # General stuff
        weights,
        dataset,
        settings.RESULT,
    )

    


if __name__ == "__main__":
    main()
