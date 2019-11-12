#!/usr/bin/env python
# coding: utf-8

# In[1]:


# https://www.youtube.com/watch?v=ok2s1vV9XW0
import gensim.models
import gensim.corpora
import gensim as gs
import pyLDAvis as pvis
import pyLDAvis.gensim
import gensim.models.coherencemodel
import warnings
warnings.filterwarnings('ignore')
import matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import seaborn as sn
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from gensim.models import FastText
import pprint
from sklearn.metrics import classification_report


# In[2]:


def load_lex_data(dataset_name, feature_set):
    print("loading lex data for", dataset_name)
    dataset = pd.read_csv("../cleaned/" + dataset_name + "_clean.csv")
    
    targets = dataset["a"]
    inputs = dataset[feature_set]
    train_x, test_x, train_y, test_y = train_test_split(inputs, targets, test_size=0.2)
    return train_x, test_x, train_y, test_y    

def load_vector_data(dataset_name):
    print("loading vector data for", dataset_name)
    sentences = pd.read_csv("../cleaned/" + dataset_name + "_stems.csv", delimiter=",").astype(str).values.tolist()
    targets = pd.read_csv("../cleaned/" + dataset_name + "_clean.csv", delimiter=",").astype(str)["a"].tolist() 
    
    # replace placeholders (" "), make one-string-sentences
    print("... replacing placeholders")
    for index, sample in enumerate(sentences): 
            sentences[index] = list(filter((" ").__ne__, sample))
    inputs = [" ".join(sentence) for sentence in sentences]
    
    # build model over sentences (size=dimension of word vectors), convert sentences to vectors
    vector_model = FastText(size=32, window=3, min_count=1)
    vector_model.build_vocab(inputs)  
    vector_model.train(sentences=inputs, total_examples=len(inputs), total_words=vector_model.corpus_total_words, epochs=10)
    inputs = [vector_model.wv[sample] for sample in inputs]
    
    # split data and return
    train_x, test_x, train_y, test_y = train_test_split(inputs, targets, test_size=0.2)
    return train_x, test_x, train_y, test_y    

def load_topic_data(dataset_name, num_topics):
    print("loading topic data for", dataset_name)
    # load inputs and labels
    dataset = pd.read_csv("../cleaned/" + dataset_name + "_stems.csv").astype(str).values.tolist() 
    targets = pd.read_csv("../cleaned/" + dataset_name + "_clean.csv")["a"].tolist()
    # remove placeholders from the stems dataset
    for index, sample in enumerate(dataset): 
            dataset[index] = list(filter((" ").__ne__, sample))
    # create dic, copora and lda-model
    dic = gs.corpora.Dictionary(dataset)
    corpus = [dic.doc2bow(sample) for sample in dataset]
    lda_model = gensim.models.ldamulticore.LdaMulticore(corpus=corpus, id2word=dic, num_topics=num_topics, random_state=100, chunksize=100, passes=10, per_word_topics=True)#update_every=1, 
    
    vecs = []
    # for every sentence in the dataset
    for i, sample in enumerate(dataset):
        # get the vector-representations from the doc
        sentence = dic.doc2bow(dataset[i])
        # get the topics from the document (they are ordered by the topic ic)
        topics = lda_model.get_document_topics(sentence, minimum_probability=0.0)
        # write the probability for every topic into a single list
        topic_vec = [topics[i][1] for i in range(num_topics)] 
        # append the prob-vector for this sentence into the all-vectors-list
        vecs.append(topic_vec)
    dataset = vecs
    
    train_x, test_x, train_y, test_y = train_test_split(dataset, targets, test_size=0.2)
    return train_x, test_x, train_y, test_y

def classify_with_rf(train_x, test_x, train_y, test_y, num_trees): 
    print("building rf model")
    rf = RandomForestClassifier(n_estimators=num_trees)
    print("... training model")
    rf.fit(train_x, train_y)
    print("... calcularing score")
    pred_y = rf.predict(test_x)
    # model metadata
    score, f1_scoore = rf.score(train_x, train_y), f1_score(test_y, pred_y, average="weighted")
    return (test_y, pred_y, score, f1_scoore, num_trees), rf.feature_importances_ 

def get_best_tree_num(dataset_name, feature_set, feature_set_name, max_trees, f="features"):
    indexes, f1 = [], []
    if f == "vec":
        data = load_vector_data(dataset_name)
    elif f == "topic":
        data = load_topic_data(dataset_name, num_topics_dict.get(dataset_name))
    else:        
        data = load_lex_data(dataset_name, features["lex"])
    
    for i in range(1,max_trees):
        print(i)
        results, importance = classify_with_rf(*data, i)
        f1.append(results[3])
        indexes.append(i)

    draw_plot(dataset_name, feature_set_name, indexes, f1, max(f1), f1.index(max(f1))+1)
    
def draw_confusion_matrix(dataset_name, feature_set_name, test_y, pred_y, score, f1_scoore, num_trees, num_topics): 
    fig = plt.figure()
    hm = sn.heatmap(confusion_matrix(test_y, pred_y), fmt="d", linewidth=0.5, annot=True, square=True, xticklabels=["h", "s", "a", "f"], yticklabels=["h", "s", "a", "f"], cmap="PuRd")
    ax1 = fig.add_axes(hm)
    ax1.set(xlabel="predicted", ylabel="target")
    desc = "dataset: {} ({}), trained over {} trees and {} topics\nscore: {}, f1_score: {}".format(dataset_name, feature_set_name, num_trees, num_topics, score, f1_scoore)
    fig.text(0.5, -0.1, desc, ha='center')
    plt.show()
    fig.savefig("../img/cm_rf_" + dataset_name + "_" + feature_set_name + ".png", bbox_inches="tight")
    
def draw_plot(dataset_name, feature_set_name, x, y, best_f1, best_index):
    fig, ax = plt.subplots()
    ax.plot(x, y)
    ax.set(xlabel="num_trees", ylabel="f1_score")
    desc = "dataset: {} ({})\nbest f1_score: {}, with num_trees: {}".format(dataset_name, feature_set_name, best_f1, best_index)
    fig.text(0.5, -0.07, desc, ha='center')
    plt.show()
    fig.savefig("../img/num_trees_" + dataset_name + "_" + feature_set_name + ".png", bbox_inches="tight")  

# achtung: Die methode plottet alle coefficients. Immer also nur ein dataset durchlaufen lassen
def draw_coefficients_plot(dataset_name, results, coefficients):
    print(coefficients)
    num_features = len(coefficients[0])
    fig = plt.figure()
    
    plt.plot(range(len(coefficients[0])), coefficients[0], "o", label=results[0][0] + "_" + results[0][1])#, label=classes[i]
    plt.plot(range(len(coefficients[1])), coefficients[1], "o", label=results[1][0] + "_" + results[1][1])#, label=classes[i]
    plt.plot(range(4, 4 + len(coefficients[2])), coefficients[2], "o", label=results[2][0] + "_" + results[2][1])#, label=classes[i]
    #for i, item in enumerate(coefficients):
    #    print(results[i][0])
    #    plt.plot(range(len(item)), item.T, "o", label=results[i][0] + "_" + results[i][1])#, label=classes[i]
    desc = "dataset: {}".format(dataset_name)
    fig.text(0.5, -0.05, desc, ha='center')
    plt.xticks(range(0, num_features), features.get("full"), rotation=90)
    #plt.legend()#loc=1
    plt.grid()
    plt.show()
    fig.savefig("../img/coef_rf_" + dataset_name + ".png", bbox_inches="tight")


# In[ ]:





# In[3]:


classes = ["happiness", "sadness", "anger", "fear"]
datasets = ["emotion", "norm_emotion", "tweet", "norm_tweet"]
features = {
    "full": ["wc", "uwc", "ewc", "cpc", "hc", "sc", "ac", "fc"],
    "nolex": ["wc", "uwc", "ewc", "cpc"],
    "lex": ["hc", "sc", "ac", "fc"]
}
classes = ["happiness", "sadness", "anger", "fear"]
trees_for_dataset = {
    "emotion_full": 100,
    "emotion_nolex": 12,
    "emotion_lex": 18,
    "emotion_vec": 1, 
    "norm_emotion_full": 18,
    "norm_emotion_nolex": 33,
    "norm_emotion_lex": 28, 
    "norm_emotion_vec": 198,
    "norm_emotion_topic": 99999,
    "tweet_full": 5,
    "tweet_nolex": 14, 
    "tweet_lex": 33,
    "tweet_vec": 1,
    "norm_tweet_full": 21,
    "norm_tweet_nolex": 12,
    "norm_tweet_lex": 10,
    "norm_tweet_vec": 150,
    "norm_tweet_topic": 87
}
num_topics_dict = {
    "norm_tweet": 79,
    "norm_emotion": 186
}


# In[5]:


# calculate optimal tree numbers
#for dataset in ["norm_emotion"]: 
#    for key, feature_set in features.items(): 
#        print(dataset, key)
#        get_best_tree_num(dataset, feature_set, key, 200, True)


# In[ ]:


#all_results = []
#importances = []

#for dataset_name in ["tweet"]: 
#    for key, feature_set in features.items(): 
#        results, importance = classify_with_rf(*load_lex_data(dataset_name, feature_set), trees_for_dataset.get(dataset_name + "_" + key, 10))
#        all_results.append([dataset_name, key, *results])
#        importances.append(importance)
    #results, importance = classify_with_rf(*load_vector_data(dataset_name), trees_for_dataset.get(dataset_name + "_vec", 10))
    #all_results.append([dataset_name, "vec", *results])
    #importances.append(importance)
    
#for index, result in enumerate(all_results):
#    with open("../img/report_rf_" + result[0] + "_"  + result[1] + ".txt", 'w') as f:
#        print((result[0] + "_" + result[1] + " (" + str(result[5]) + "):\n" + 
#          classification_report(result[2], result[3],target_names=classes)), file=f)
#    draw_confusion_matrix(*result)
    #draw_coefficients_plot(result[0], result[1], importances[index])
#draw_coefficients_plot(all_results[0][0], all_results, importances)


# In[33]:


#get_best_tree_num("norm_tweet", 1, "topic", 200, "topic")


# In[40]:


# train lrandom forests over the topic distributions
all_results = []
importances = []
for dataset_name in ["norm_emotion"]: 
    train_x, test_x, train_y, test_y = load_topic_data(dataset_name, num_topics_dict.get(dataset_name))
    results, importance = classify_with_rf(train_x, test_x, train_y, test_y, trees_for_dataset.get(dataset_name + "_topic", 10))
    all_results.append([dataset_name, "topics", *results])
    importances.append(importance)

for index, result in enumerate(all_results): 
    with open("../img/report_rf_" + result[0] + "_"  + result[1] + ".txt", 'w') as f:
        print((result[0] + "_" + result[1] + " (" + str(result[5]) + "):\n" + 
          classification_report(result[2], result[3],target_names=classes)), file=f)
    draw_confusion_matrix(*result, num_topics_dict.get(result[0]))


# In[ ]:





# In[ ]:




