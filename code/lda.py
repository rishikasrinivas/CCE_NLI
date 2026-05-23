# download necessary libraries and packages for our topic modeling algorithm

'''
learns a distribution for the document wrt the topics (so what topics are in what documnets) and one to map words to the topic (so what words cover what topics) anduses that to group words into topics based on the document

'''
import os
import nltk
import re
import string
import gensim
import numpy as np

# for cleaning prefatory matter from Project Gutenberg texts
from gutenbergpy import textget

# for tokenization
from nltk.tokenize import word_tokenize
nltk.download("punkt_tab")
nltk.download('wordnet')

# for stopword removal
from nltk.corpus import stopwords
nltk.download('stopwords')

# for lemmatization and POS tagging
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
nltk.download('averaged_perceptron_tagger')

# for LDA
from gensim import corpora
from gensim.models import LdaModel
from gensim.models.coherencemodel import CoherenceModel

# for LDA evaluation
import pyLDAvis
import pyLDAvis.gensim_models as gensimvisualize


import pandas as pd
import re
from collections import defaultdict
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

import os
def get_indiv_concepts(formula) -> list:
    concepts = []
    concps = re.findall(r'(?<!\bNOT\s)(?:\b(?:hyp|pre|oth):[^\s)]+)', formula)
    for c in concps:
        try:
            end_idx = c.index(')')
        except:
            end_idx = len(c)
        concepts.append(c[:end_idx])
    return concepts



def load_csv_data(filepath):
    """Load CSV and extract unit-concept mappings."""
    df = pd.read_csv(filepath)
    unit_concepts = defaultdict(set)
    raw_concepts= []
    for _, row in df.iterrows():
        unit = row['unit']
        formula = row['best_name']
        concepts = get_indiv_concepts(formula)
        
        unit_concepts[unit].update(concepts)
        raw_concepts.extend(concepts)
    raw_concepts = [word.split(":")[-1] for word in raw_concepts]
        
    return unit_concepts, raw_concepts
#how concepts removal same across algortihms 

mapping, all_concepts = load_csv_data('BERT/exp/CoFi/Run0.25_5/Expls/0.0%Pruned/Cluster1IOUS1024N.csv')
stop_words = set(stopwords.words('english'))
tokens = word_tokenize(' '.join(all_concepts))
filtered_tokens = [word for word in tokens if word not in stop_words]

# remove non-alphabetic tokens
filtered_tokens_alpha = [word for word in filtered_tokens if word.isalpha()]

dictionary = corpora.Dictionary([filtered_tokens_alpha])


# generate corpus as BoW
corpus = [dictionary.doc2bow(text) for text in [filtered_tokens_alpha]]

# train LDA model
lda_model = LdaModel(corpus=corpus, id2word=dictionary, random_state=4583, chunksize=20, num_topics=4, passes=200)

# print LDA topics
for topic in lda_model.print_topics(num_topics=4, num_words=10):
    print(topic)
