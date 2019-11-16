#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import spacy
import os
import nltk
from nltk.stem.porter import *
import numpy as np


# In[5]:


def log(summary):
    print(summary)
    
def combine_datasets(list_of_datasets, lexica=False):
    print("make_data", list_of_datasets)
    if not lexica: 
        datasets = []
        for dataset_name in list_of_datasets:
            datasets.append(pd.read_csv("../raw/" + dataset_name + ".csv")) 
        dataset = pd.concat(datasets, axis=0, ignore_index=True)
        return dataset
    else:
        lexica = []
        for dataset_name in list_of_datasets: 
            lexicon = pd.read_csv("../../lexica/" + dataset_name + ".csv")
            lexica.append(lexicon["stems"].tolist())
        return lexica
    
def normalize_label_length(list_of_datasets, save_name):
    print("normalize_label_length")
    df = combine_datasets(list_of_datasets)
    df = df.sample(frac=1)
    l = min(
        len(df.loc[df["affect"] == "happiness"]), 
        len(df.loc[df["affect"] == "sadness"]), 
        len(df.loc[df["affect"] == "anger"]), 
        len(df.loc[df["affect"] == "fear"]))
    norm_df = pd.concat([
        df.loc[df["affect"] == "happiness"][:l],
        df.loc[df["affect"] == "sadness"][:l],
        df.loc[df["affect"] == "anger"][:l],
        df.loc[df["affect"] == "fear"][:l]])
    df = df.sample(frac=1)
    norm_df.to_csv("../cleaned/norm_" + save_name + ".csv", index=False,  float_format='%.3f')
    
def extract_features(dataset_name, list_of_lexica, print_row): 
    affect_encoding = {"happiness": 0, "sadness": 1, "anger": 2, "fear": 3}
    cleaned_dataset = []
    wc, pos, stems = [], [], []
    dataset = pd.read_csv("../cleaned/norm_" + dataset_name + ".csv")
    # pos 97 = Satzzeichen, 103 = Leerzeichen
    print("analyzing sentence features:", dataset_name)
    for index, row in dataset.iterrows():
        if index % print_row == 0: log("... searching row " + str(index) + "/" + str(len(dataset)))
        doc = nlp(split_punct(row["text"]))
        doc = nlp(" ".join([token.text for token in doc if not token.is_stop and token.pos != 103]))
        if len(doc) != 0:
            pos.append([token.pos for token in doc])
            stems.append([stemmer.stem(token.text) for token in doc if token.pos != 97])
            emotion_words = get_emotion_words(stems[-1:][0], list_of_lexica)
            cleaned_dataset.append([
                " ".join([token.text for token in doc]), affect_encoding[row["affect"]],
                len(doc), (sum([token.text.isupper() for token in doc])/len(doc)), 
                (len(doc.ents)/len(doc)),get_cons_punct_count(pos[-1:][0]), 
                emotion_words[0]/len(doc), emotion_words[1]/len(doc), emotion_words[2]/len(doc), emotion_words[3]/len(doc)])
        
    seq_len = max([row[2] for row in cleaned_dataset])
    pos = extend_list(pos, seq_len, 0)
    stems = extend_list(stems, seq_len, " ")    
    
    df = pd.DataFrame(data=cleaned_dataset, columns=["t", "a", "wc", "uwc", "ewc", "cpc", "hc", "sc", "fc", "ac"])
    df["wc"] = [(item/seq_len) for item in df["wc"].tolist()] # normalisiert über die größte anzahl von wörtern in einem sample
    df.to_csv("../cleaned/norm_" + dataset_name + "_clean.csv", sep=",", index=False, float_format='%.3f')
    df = pd.DataFrame(data=pos)
    df.to_csv("../cleaned/norm_" + dataset_name + "_pos.csv", sep=",", index=False, float_format='%.3f')
    df = pd.DataFrame(data=stems)
    df.to_csv("../cleaned/norm_" + dataset_name + "_stems.csv", sep=",", index=False, float_format='%.3f')
    
def extend_list(l, seq_len, extension):
    for index, row in enumerate(l):
        row.extend([extension] * (seq_len - len(row)))
    return l
    
def split_punct(text):
    replacement = [(".", " . "), (",", " , "), ("!", " ! "), ("?", " ? ")]
    for k, v in replacement: 
        text = text.replace(k, v)
    return text
    
def get_emotion_words(stems, list_of_lexica):
    emotion_words = np.zeros(4)
    for index, lexicon in enumerate(list_of_lexica): 
        for stem in stems:
            if stem in lexicon:
                emotion_words[index] = emotion_words[index] + 1
    return emotion_words

def get_cons_punct_count(pos):
    cons_punct_count = 0
    for index, item in enumerate(pos[:-1]):
        if item == 97 and item == pos[index+1]:
            cons_punct_count += 1
    return cons_punct_count


# In[6]:


pd.options.mode.chained_assignment = None  # default='warn'
nlp = spacy.load("en_core_web_lg")
stemmer = nltk.stem.SnowballStemmer('english')
lexica = ["clean_happiness", "clean_sadness", "clean_anger", "clean_fear"]
list_of_lexica = combine_datasets(lexica, True)

raw_datasets = {
    "tweet": ["emoint", "crowdflower", "tec"],
    "emotion": ["emotion_classification_1", "emotion_classification_2", "emotion_classification_3", "emotion_classification_4", "emotion_classification_5","emotion_classification_6","emotion_classification_7","emotion_classification_8"]
}


# In[7]:


dataset_names = ["tweet", "emotion"]
for dataset_name in dataset_names:
    normalize_label_length(raw_datasets[dataset_name], dataset_name)
    extract_features(dataset_name, list_of_lexica, 5000)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




