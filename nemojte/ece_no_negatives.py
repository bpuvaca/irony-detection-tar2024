from ece import get_ece
import pandas as pd
import os
data = {'model': []}      

n_bins = 5
for model in ['bertweet', 'roberta', 'bert']:
    data['model'].append(model)     
    eces = {}
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
            
