import pandas as pd
import numpy as np
from scipy.stats.stats import spearmanr
import matplotlib.pyplot as plt

col = ["wc", "uwc", "ewc", "cpc", "hc", "sc", "ac", "fc"]
datasets = ["emotion", "norm_emotion", "tweet", "norm_tweet"]
features = {
    "full": ["wc", "uwc", "ewc", "cpc", "hc", "sc", "ac", "fc", "a"],
    "nolex": ["wc", "uwc", "ewc", "cpc", "a"],
    "lex": ["hc", "sc", "ac", "fc", "a"]
}

fe = ["wc", "uwc", "ewc", "cpc", "hc", "sc", "ac", "fc"]

for dataset_name in ["norm_tweet", "norm_emotion"]:
    a = []
    dataset = pd.read_csv("../cleaned/" + dataset_name + "_clean.csv").fillna(0)
    targets = dataset["a"].tolist()
    # calculate correlation between every feature and target
    for f in fe:
        inputs = dataset[f]
        a.append(spearmanr(inputs, targets)[0])

    for i, e in enumerate(a):
        if np.isnan(e):
            a[i] = 0
    # plot correlation
    fig, ax = plt.subplots()
    desc = "x-Achse: Features des Datensatzes " + dataset_name + "\ny-Achse: Pearson-Korrealtion zum Affekt-Label"
    fig.text(0.5, -0.1, desc, ha='center')
    plt.xticks(range(0, len(fe)), fe, rotation=90)
    plt.hlines(0, 0, len(fe), linestyle="dotted")
    plt.grid()
    ax.scatter(fe, a)
    for i, txt in enumerate(a):
        ax.annotate(str(round(txt, 2)), (fe[i], a[i]))
    plt.show()
    fig.savefig("../img/corr_" + dataset_name + "_" + ".png", bbox_inches="tight")
