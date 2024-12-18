import pandas as pd
import numpy as np
from itertools import combinations

irony = pd.read_csv("../datasets/crossval/irony.csv")
sarcasm = pd.read_csv("../datasets/crossval/sarcasm.csv")
polarity = pd.read_csv("../datasets/crossval/polarity.csv")
other = pd.read_csv("../datasets/crossval/other.csv")

irony = irony[irony['label'] == 1]
print(len(irony))
sarcasm = sarcasm[sarcasm['label'] == 1]
print(len(sarcasm))
polarity = polarity[polarity['label'] == 1]
# print(len(polarity))
other = other[other['label'] == 1]
# print(len(other))

datasets = [irony, sarcasm, polarity, other]
dataset_names = ['irony', 'sarcasm', 'polarity', 'other']

for (df1, name1), (df2, name2) in combinations(zip(datasets, dataset_names), 2):
    print(name1, len(df1), name2, len(df2))
    if len(df1) > len(df2):
        df1 = df1.sample(len(df2))
    else:
        df2 = df2.sample(len(df1))
    df2['label'] = 0
    combined_df = pd.concat([df1, df2]).sample(frac=1).reset_index(drop=True)
    print(len(combined_df))
    combined_df.to_csv(f"../datasets/crossval/{name1}_{name2}.csv", index=False)
    # combined_df = pd.concat([df1, df2])

