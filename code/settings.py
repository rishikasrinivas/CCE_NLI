"""
Settings
"""

import os

CUDA = True
ALPHA = None  # Use "None" to use ReLU threshold (i.e., > 0)
BEAM_SIZE = 10
MAX_FORMULA_LENGTH = 5
COMPLEXITY_PENALTY = 1.00
TOPN = 5
DEBUG = False

# Choices: iou, precision, recall
METRIC = "iou"

EMBEDDING_NEIGHBORHOOD_SIZE = 5
NUM_CLUSTERS=3


PRUNE = {
    'encoder.emb.weight' : 0,
    'encoder.rnn.weight_ih_l0' : 0.0,
    'encoder.rnn.weight_hh_l0' : 0.0,
    'encoder.rnn.bias_ih_l0' : 0.0,
    'encoder.rnn.bias_hh_l0' :0.0,
    'bn.weight' : 0.0, 
    'bn.bias' : 0.0, 
    'bn.running_mean' : 0.0,
    'bn.running_var': 0.0, 
    'bn.num_batches_tracked' : 0.0,
    'mlp.0.weight' : 0.2, 
    'mlp.0.bias' : 0.0, 
    'mlp.3.weight': 0.0, 
    'mlp.3.bias': 0.0,
}
NEURONS = None
PARALLEL = 1
PRUNE_METRICS_DIR='models/snli/'
MIN_ACTS=500
SHUFFLE = False
SAVE_EVERY = 4
PRUNE_METHOD='lottery_ticket' #coices: lottery_ticket, incremental
PRUNE_AMT=0.2
# How many "maximally activating" open features to use, PER CATEGORY
MAX_OPEN_FEATS = 5
# Minimum number of activations to analyze a neuron


MODEL = "models/snli/random_inits.pth"
MODEL_TYPE = "bowman"  # choices: bowman, minimal, imdbData
RANDOM_WEIGHTS = False  # Initialize weights randomly (equivalent to an untrained model)
N_SENTENCE_FEATS = 2000  # how many of the most common sentence lemmas to keep

DATA = "data/analysis/snli_1.0_dev.feats"

assert DATA.endswith(".feats")
VECPATH = DATA.replace(".feats", ".vec")

# Overridables
if "MTDISSECT_MODEL" in os.environ:
    MODEL = os.environ["MTDISSECT_MODEL"]
if "MTDISSECT_MAX_FORMULA_LENGTH" in os.environ:
    MAX_FORMULA_LENGTH = int(os.environ["MTDISSECT_MAX_FORMULA_LENGTH"])
if "MTDISSECT_MAX_OPEN_FEATS" in os.environ:
    MAX_OPEN_FEATS = int(os.environ["MTDISSECT_MAX_OPEN_FEATS"])
if "MTDISSECT_METRIC" in os.environ:
    METRIC = os.environ["MTDISSECT_METRIC"]

mbase = os.path.splitext(os.path.basename(MODEL))[0]
dbase = os.path.splitext(os.path.basename(DATA))[0]
prune = 'LH' #or 'None' or 'Hydra'
RESULT_EXP = f"exp/{prune}/{dbase}-{mbase}-sentence-{MAX_FORMULA_LENGTH}{'-debug' if DEBUG else ''}{f'-{METRIC}' if METRIC != 'iou' else ''}{f'-random-weights' if RANDOM_WEIGHTS else ''}{f'{NUM_CLUSTERS}_clusters'}"
RESULT_MASK = f"masks/{prune}/{dbase}-{mbase}-sentence-{MAX_FORMULA_LENGTH}{'-debug' if DEBUG else ''}{f'-{METRIC}' if METRIC != 'iou' else ''}{f'-random-weights' if RANDOM_WEIGHTS else ''}{f'{NUM_CLUSTERS}_clusters'}"

