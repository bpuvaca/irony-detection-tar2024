import pandas as pd
import os
import numpy as np


def get_ece(data, n_bins=5):

    data['probability'] = np.maximum(data["probability"], 1 - data["probability"])

    data['bin'] = pd.cut(data['probability'], bins=np.linspace(0, 1, n_bins + 1), labels=range(1, n_bins + 1))

    ece = 0
    for bin in range(1, n_bins + 1):
        bin_data = data[data['bin'] == bin]
        if len(bin_data) == 0:
            continue
        bin_probability = len(bin_data) / len(data)
        accuracy = sum(bin_data['label'] == bin_data['prediction']) / len(bin_data)
        confidence = bin_data['probability'].mean()
        contribution = bin_probability* np.abs(accuracy - confidence)
        ece += contribution

    return ece


eces = {'model': []}      

n_bins = 5
for model in ['bertweet', 'roberta', 'bert']:
    # print(data)
    eces['model'].append(model)     
    for ds in ["irony_sarcasm"]:#, "irony_mix", "polarity", "sarcasm", "mix"]:
        if ds not in eces.keys():
            eces[ds] = []
        data = pd.read_csv(f"../preds/crossval4/{model}/{ds}/{ds}/{model}_trained_on_{ds}_evaluated_on_{ds}_fold_1.csv")
        for i in range(2, 6):
            data = pd.concat([data, pd.read_csv(f"../preds/crossval4/{model}/{ds}/{ds}/{model}_trained_on_{ds}_evaluated_on_{ds}_fold_{i}.csv")])
        ece = get_ece(data, n_bins=5)
        eces[ds].append(ece)
    df = pd.DataFrame(eces)
    os.makedirs(f'../results/no_negatives', exist_ok=True)
    df.to_csv(f'../results/no_negatives/eces_4e_{n_bins}bins.csv', index=False)
            
