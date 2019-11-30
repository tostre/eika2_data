import gensim.models
import gensim.corpora
import gensim as gs
import pandas as pd
from gensim.models import FastText
from gensim.models.phrases import Phrases, Phraser


def save_models(dataset_name, num_topics):
    # load inputs and labels
    dataset = pd.read_csv("../cleaned/" + dataset_name + "_stems.csv").astype(str).values.tolist()
    # remove placeholders from the stems dataset
    for index, sample in enumerate(dataset):
        dataset[index] = list(filter((" ").__ne__, sample))
    # create dic, copora and lda-model
    dic = gs.corpora.Dictionary(dataset)
    dic.save("../models/dictionary/" + dataset_name + "_dictionary")
    corpus = [dic.doc2bow(sample) for sample in dataset]
    lda_model = gensim.models.ldamulticore.LdaMulticore(corpus=corpus, id2word=dic, num_topics=num_topics, random_state=100, chunksize=100, passes=10, per_word_topics=True)  # update_every=1,
    lda_model.save("../models/topic_models/" + dataset_name + "_ldamodel")
    inputs = [" ".join(sentence) for sentence in dataset]
    vector_model = FastText(size=32, window=3, min_count=1)
    vector_model.build_vocab(inputs)
    vector_model.train(sentences=inputs, total_examples=len(inputs), total_words=vector_model.corpus_total_words, epochs=10)
    vector_model.save("../models/word_embeddings/" + dataset_name + "_fasttext")
    # make bigram model
    sentences = pd.read_csv("../cleaned/" + dataset_name + "_clean.csv")["t"].tolist()
    tokenized = [t.split() for t in sentences]
    phrases = Phrases(tokenized)
    bigram = Phraser(phrases)
    bigram.save("../models/bigrams/bigram_" + dataset_name + ".pkl")


num_topics_dict = {
    "norm_tweet": 79,
    "norm_emotion": 186
}
datasets = ["norm_emotion"]
