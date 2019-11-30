import pandas as pd
import spacy
import os
import nltk

pd.options.mode.chained_assignment = None  # default='warn'
nlp = spacy.load("en_core_web_lg")
stemmer = nltk.stem.SnowballStemmer('english')
stems = []

current_dir = os.path.dirname(os.path.abspath(__file__))
list_of_datasets = os.listdir(current_dir)
list_of_datasets.remove("clean_lexica.py")
list_of_datasets = ["NRC-AffectIntensity-Lexicon.csv"]

print("datasets: ", list_of_datasets)

for dataset_name in list_of_datasets:
    dataset = pd.read_csv(dataset_name, delimiter="\t", dtype={"text": str, "score": float, "AffectDimension": str})

    stems.clear()
    row_index = 0
    poss = []
    poss_ = []

    for index, row in dataset.iterrows():
        doc = nlp(row["text"])
        stems.append([stemmer.stem(token.text) for token in doc])
        poss.append(token.pos for token in doc)
        poss.append(token.pos_ for token in doc)

        if row_index % 10000 == 0 and row_index != 0:
            ("... searching row " + row_index.__str__() + "/" + len(dataset).__str__())
        row_index = row_index + 1

    dataset["stems"] = stems
    dataset["pos"] = poss
    dataset["pos_"] = poss

    dataset.to_csv("clean_" + dataset_name, sep=",", index=False, float_format='%.3f')
