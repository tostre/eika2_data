#!/usr/bin/env python
# coding: utf-8

# In[6]:


import gensim.models
import gensim.corpora
import gensim as gs
import pandas as pd
from gensim.models import FastText


# In[7]:


def save_models(dataset_name, num_topics):
    print("loading topic data for", dataset_name)
    # load inputs and labels
    dataset = pd.read_csv("../cleaned/" + dataset_name + "_stems.csv").astype(str).values.tolist() 
    # remove placeholders from the stems dataset
    print("removing placeholders")
    for index, sample in enumerate(dataset): 
            dataset[index] = list(filter((" ").__ne__, sample))
    # create dic, copora and lda-model
    print(dataset)
    print("making dic")
    dic = gs.corpora.Dictionary(dataset)
    dic.save("../models/dictionary/" + dataset_name + "_dictionary")
    print("making corpus")
    corpus = [dic.doc2bow(sample) for sample in dataset]
    print("making lda")
    lda_model = gensim.models.ldamulticore.LdaMulticore(corpus=corpus, id2word=dic, num_topics=num_topics, random_state=100, chunksize=100, passes=10, per_word_topics=True)#update_every=1, 
    lda_model.save("../models/topic_models/" + dataset_name + "_ldamodel")
    print("making fasttext")
    inputs = [" ".join(sentence) for sentence in dataset]
    vector_model = FastText(size=32, window=3, min_count=1)
    vector_model.build_vocab(inputs)  
    vector_model.train(sentences=inputs, total_examples=len(inputs), total_words=vector_model.corpus_total_words, epochs=10)
    vector_model.save("../models/word_embeddings/" + dataset_name + "_fasttext")
    
def load_test(dataset_name, num_topics):
    print("loading dic")
    #dic = gs.corpora.Dictionary.load("../models/dictionary/" + dataset_name + "_dictionary")
    print("loading topic model")
    lda_model = gensim.models.ldamulticore.LdaMulticore.load("../models/topic_models/" + dataset_name + "_ldamodel")
    topics = lda_model.show_topics(num_topics = num_topics)
    print(len(topics))
    print(topics)
    print("loading fasttext")
    #vector_model = FastText.load("../models/word_embeddings/" + dataset_name + "_fasttext")
    
    


# In[9]:


num_topics_dict = {
    "norm_tweet": 79,
    "norm_emotion": 186
}

datasets = ["norm_tweet", "norm_emotion"]

for dataset in datasets: 
    save_models(dataset, num_topics_dict[dataset])
#dataset_name = "test"
#save_models(dataset_name, num_topics_dict[dataset_name])

#load_test(dataset_name, num_topics_dict[dataset_name] + 10)


# In[ ]:




