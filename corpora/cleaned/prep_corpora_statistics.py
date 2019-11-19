#!/usr/bin/env python
# coding: utf-8

# In[51]:


import pandas as pd
import gensim as gs
import spacy
import gensim.models.phrases
nlp = spacy.load("en_core_web_lg")


# In[87]:


import nltk
from nltk.util import ngrams


# In[49]:


def get_statistics_lexicon(dataset_name, column_name, delimiter):
    dataset = pd.read_csv("lll/" + dataset_name + ".csv", delimiter=delimiter)
    terms = [dataset[column_name].tolist()]
    dic = gs.corpora.Dictionary(terms)
    num_unigrams = len(dic)
    print(dataset_name)
    print("...num unigrams", num_unigrams)
    
def get_statistics_corpus(dataset_name, column_name, delimiter):
    dataset = pd.read_csv("lll/" + dataset_name + ".csv", delimiter=delimiter)
    terms = dataset[column_name].tolist()
    all_sentences = []
    for sentence in terms: 
        doc = nlp(sentence)
        sent = [token.text for token in doc]
        all_sentences.append(sent)

data = {
    "crowdflower": ["content", ","],
    "emoint": ["text", "\t"],
    "tec": ["text", "\t"],
    "emotion_classification": ["text", ","]
}

data = {
	"norm_emotion": ["t", ","],
	"norm_tweet": ["t", ","],
}


for key, value in data.items():
    print(key, value)
    print(value[0])
    dataset = pd.read_csv(key + "_clean.csv", delimiter=value[1], error_bad_lines=False)
    terms = dataset[value[0]].tolist()
    all_sentences = []
    for sentence in terms: 
        doc = nlp(sentence)
        sent = [token.text for token in doc]
        all_sentences.append(sent)
    # get unigrams
    dic = gs.corpora.Dictionary(all_sentences)
    print("...num unigrams:", len(dic))
    # get bigrams
    all_bgr = []
    all_tgr = []
    for sent in all_sentences: 
        all_bgr.append(list(ngrams(sent,2)))
        all_tgr.append(list(ngrams(sent,3)))
    flat_bgr = [item for sublist in all_bgr for item in sublist]
    flat_tgr = [item for sublist in all_tgr for item in sublist]
    flat_bgr = list(dict.fromkeys(flat_bgr))
    flat_tgr = list(dict.fromkeys(flat_tgr))
    print("...num bigrams:", len(flat_bgr))
    print("...num trigrams:", len(flat_tgr))


# In[ ]:




