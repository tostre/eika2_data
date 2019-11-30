import pandas as pd
import gensim.models
import gensim.corpora
import gensim as gs
import pyLDAvis as pvis
import pyLDAvis.gensim
import gensim.models.coherencemodel
import matplotlib.pyplot as plt


def load_topic_data(dataset_name):
    sentences = pd.read_csv("../cleaned/" + dataset_name + "_stems.csv", delimiter=",").astype(str).values.tolist()
    for index, sample in enumerate(sentences):
        sentences[index] = list(filter((" ").__ne__, sample))
    dic = gs.corpora.Dictionary(sentences)
    corpus = [dic.doc2bow(sample) for sample in sentences]
    return sentences, dic, corpus


def visualize_lda(model, corpus, dic):
    pvis.enable_notebook()
    vis = pvis.gensim.prepare(model, corpus, dic)
    vis.show()


def get_coherence_score(model, sentences, dic):
    # the higher the better it is, nutzen um versch. modelle zu vergleichen (mit untersch. topic-anzahö)
    coherence_model_lda = gensim.models.coherencemodel.CoherenceModel(model=model, texts=sentences, dictionary=dic, coherence='c_v')
    coherence_score = coherence_model_lda.get_coherence()
    return coherence_score


def draw_plot(dataset_name, x, y, best_coherence, best_num_topics):
    fig, ax = plt.subplots()
    ax.plot(x, y)
    ax.set(xlabel="num_topics", ylabel="coherence")
    desc = "dataset: {}\nbest coherence: {}, with topics: {}".format(dataset_name, best_coherence, best_num_topics)
    fig.text(0.5, -0.07, desc, ha='center')
    plt.grid()
    plt.show()
    fig.savefig("../img/num_topics_" + dataset_name + ".png", bbox_inches="tight")


def find_best_topic_num(dataset_name, lim_low, lim_high):
    coherences = []
    models = []
    sentences, dic, corpus = load_topic_data(dataset_name)
    for i in range(lim_low, lim_high + 1):
        lda_model = gensim.models.ldamulticore.LdaMulticore(corpus=corpus, id2word=dic, num_topics=i, random_state=100,
                                                            chunksize=100, passes=10, per_word_topics=True)  # update_every=1,
        models.append(lda_model)
        coherences.append(get_coherence_score(lda_model, sentences, dic))
    max_coherence_index = coherences.index(max(coherences))
    draw_plot(dataset_name, list(range(lim_low, len(coherences) + lim_low)), coherences, max(coherences), max_coherence_index + lim_low)
    models[max_coherence_index].save("../models/tm_" + dataset_name + ".model")
