import pandas as pd
from sklearn.metrics import f1_score
import numpy as np
import os

for model in ['bertweet', 'roberta', 'bert']:  
    data = {'trained_on': []}          
    train_dsets = ["sarcasm", "polarity", "sarcasm_mix", "irony_mix", "mix"]
    for train_ds in train_dsets:
        data['trained_on'].append(train_ds)
        for test_ds in ["sarcasm", "polarity", "irony", "other", "sarcasm_mix", "irony_mix", "mix"]:
            if test_ds not in data.keys():
                data[test_ds] = []
            if "mix" not in test_ds or test_ds == train_ds:
                f1s = []
                for k in range(1, 6):
                    results = pd.read_csv(f"../preds/crossval4/{model}/{train_ds}/{test_ds}/{model}_trained_on_{train_ds}_evaluated_on_{test_ds}_fold_{k}.csv")
                    f1s.append(f1_score(results['label'], results['prediction']))
                average_f1 = sum(f1s) / 5
                stdev = np.std(f1s)
                data[test_ds].append(f"{average_f1:.2f} ± {stdev:.2f}")
            else: 
                if test_ds.startswith("irony"):
                    components = ["irony", "polarity"]
                elif test_ds.startswith("sarcasm"):
                    components = ["sarcasm", "other"]
                else:
                    components = ["irony", "polarity", "sarcasm", "other"]
                f1s = []
                for k in range(1, 6):
                    results = [pd.read_csv(f"../preds/crossval4/{model}/{train_ds}/{component}/{model}_trained_on_{train_ds}_evaluated_on_{component}_fold_{k}.csv") for component in components]                    
                    labels = [result['label'] for result in results]
                    preds = [result['prediction'] for result in results]
                    f1s.append(f1_score(pd.concat(labels), pd.concat(preds)))
                average_f1 = sum(f1s) / 5
                stdev = np.std(f1s)
                data[test_ds].append(f"{average_f1:.2f} ± {stdev:.2f}")
    df = pd.DataFrame(data)
    os.makedirs('../results', exist_ok=True)
    df.to_csv(f'../results/output_{model}4e.csv', index=False)

