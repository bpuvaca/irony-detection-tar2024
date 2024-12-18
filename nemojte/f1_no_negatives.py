import pandas as pd
from sklearn.metrics import f1_score
import numpy as np
import os

data = {'model': []}          

for model in ['bertweet', 'roberta', 'bert']:        
    data['model'].append(model)    
    dsets = ["irony_sarcasm"]#, "irony_mix", "polarity", "sarcasm", "mix"]:
    for ds in dsets:
        if ds not in data.keys():
            data[ds] = []
        f1s = []
        for k in range(1, 6):
            results = pd.read_csv(f"../preds/crossval4/{model}/{ds}/{ds}/{model}_trained_on_{ds}_evaluated_on_{ds}_fold_{k}.csv")
            f1s.append(f1_score(results['label'], results['prediction']))
        average_f1 = sum(f1s) / 5
        stdev = np.std(f1s)
        data[ds].append(f"{average_f1:.2f} ± {stdev:.2f}")

df = pd.DataFrame(data)
os.makedirs('../results/no_negatives', exist_ok=True)
df.to_csv(f'../results/no_negatives/f1s_no_negatives.csv', index=False)