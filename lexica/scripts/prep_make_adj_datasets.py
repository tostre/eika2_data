import pandas as pd
import spacy

lex_happiness = pd.read_csv("clean_happiness.csv", delimiter=",", dtype={"text": str, "affect": str, "stems": str})
lex_sadness = pd.read_csv("clean_sadness.csv", delimiter=",", dtype={"text": str, "affect": str, "stems": str})
lex_anger = pd.read_csv("clean_anger.csv", delimiter=",", dtype={"text": str, "affect": str, "stems": str})
lex_fear = pd.read_csv("clean_fear.csv", delimiter=",", dtype={"text": str, "affect": str, "stems": str})
list_happiness = lex_happiness["stems"].tolist()
list_sadness = lex_sadness["stems"].tolist()
list_anger = pd.Series(lex_anger["stems"].tolist())
list_fear = lex_fear["stems"].tolist()

emotions = ["happiness", "sadness", "anger", "fear"]
datasets = [lex_happiness, lex_sadness, lex_anger, lex_fear]

nlp = spacy.load("en_core_web_lg")

for index, dataset in enumerate(datasets):
    new_dataset = dataset.copy()
    new_dataset = new_dataset.drop(labels=["stems", "affect"], axis=1)

    text_rows = [row[1]["text"] for row in dataset.iterrows()]
    doc = nlp(" ".join(text_rows))
    pos_rows = [token.pos for token in doc]
    new_dataset["pos"] = pos_rows

    for index2, row in new_dataset.iterrows():
        if row[2] != 84:
            new_dataset = new_dataset.drop(index=index2, axis=0)
    new_dataset = new_dataset.drop(labels="pos", axis=1)

    new_dataset.to_csv("clean_" + emotions[index] + "_adj.csv", index=False, float_format='%.3f')
