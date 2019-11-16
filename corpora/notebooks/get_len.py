#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import spacy
import os
import nltk
from nltk.stem.porter import *
import numpy as np


# In[2]:


def log(summary):
    print(summary)
    
    

    
def get_len(dataset_name, print_row): 
    affect_encoding = {"happiness": 0, "sadness": 1, "anger": 2, "fear": 3}
    cleaned_dataset = []
    wc, pos, stems = [], [], []
    dataset = pd.read_csv("../cleaned/norm_" + dataset_name + ".csv")
    biggest = 0
    smallest = 0
    # pos 97 = Satzzeichen, 103 = Leerzeichen
    print("analyzing sentence features:", dataset_name)
    for index, row in dataset.iterrows():
        if index % print_row == 0: log("... searching row " + str(index) + "/" + str(len(dataset)))
        doc = nlp(split_punct(row["text"]))
        doc = nlp(" ".join([token.text for token in doc if not token.is_stop and token.pos != 103]))
        if len(doc) != 0:
            pos.append([token.pos for token in doc])
            
        
    biggest = max([row[2] for row in cleaned_dataset])
    smallest = min([row[2] for row in cleaned_dataset])
    print(biggest)
    print(smallest)

pd.options.mode.chained_assignment = None  # default='warn'
nlp = spacy.load("en_core_web_lg")
stemmer = nltk.stem.SnowballStemmer('english')


# In[4]:


dataset_names = ["tweet", "emotion"]
for dataset_name in dataset_names:
    get_len(dataset_name, 5000)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




