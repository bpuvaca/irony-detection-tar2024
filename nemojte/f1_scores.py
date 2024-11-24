import pandas as pd
from sklearn.metrics import f1_score
import numpy as np
import os

for model in ['bertweet', 'roberta', 'bert']:  
    data = {'trained_on': []}          
    for train_ds in ["sarcasm", "polarity", "sarcasm_mix", "irony_mix"]:
        data['trained_on'].append(train_ds)
        for test_ds in ["sarcasm", "polarity", "irony", "other", "sarcasm_mix", "irony_mix"]:
            if test_ds not in data.keys():
                data[test_ds] = []
            if "mix" not in test_ds or test_ds == train_ds:
                f1s = []
                for k in range(1, 6):
                    results = pd.read_csv(f"../preds/crossval/{model}/{train_ds}/{test_ds}/{model}_trained_on_{train_ds}_evaluated_on_{test_ds}_fold_{k}.csv")
                    f1s.append(f1_score(results['label'], results['prediction']))
                average_f1 = sum(f1s) / 5
                stdev = np.std(f1s)
                data[test_ds].append(f"{average_f1:.2f} ± {stdev:.2f}")
            elif test_ds.startswith("irony"):
                f1s = []
                for k in range(1, 6):
                    results1 = pd.read_csv(f"../preds/crossval/{model}/{train_ds}/polarity/{model}_trained_on_{train_ds}_evaluated_on_polarity_fold_{k}.csv")
                    results2 = pd.read_csv(f"../preds/crossval/{model}/{train_ds}/irony/{model}_trained_on_{train_ds}_evaluated_on_irony_fold_{k}.csv")
                    f1s.append(f1_score(pd.concat([results1['label'], results2['label']]), pd.concat([results1['prediction'], results2['prediction']])))
                average_f1 = sum(f1s) / 5
                stdev = np.std(f1s)
                data[test_ds].append(f"{average_f1:.2f} ± {stdev:.2f}")
    df = pd.DataFrame(data)
    os.makedirs('../results', exist_ok=True)
    df.to_csv(f'../results/output_{model}5e.csv', index=False)
