import pandas as pd
from sklearn.metrics import f1_score
import numpy as np
import os

for model in ['bertweet', 'roberta', 'bert']:  
    data = {'trained_on': []}          
    train_dsets = ["sarcasm", "polarity", "sarcasm_mix", "irony_mix"]
    if model != 'bert': train_dsets.append("mix")
    for train_ds in train_dsets:
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
            else: 
                is_mix = False
                if test_ds.startswith("irony"):
                    test_ds1 = "irony"
                    test_ds2 = "polarity"
                elif test_ds.startswith("sarcasm"):
                    test_ds1 = "sarcasm"
                    test_ds2 = "other"
                else:
                    test_ds1 = "sarcasm"
                    test_ds2 = "irony"
                    test_ds3 = "polarity"
                    test_ds4 = "other"
                    is_mix = True
                f1s = []
                for k in range(1, 6):
                    results1 = pd.read_csv(f"../preds/crossval/{model}/{train_ds}/{test_ds1}/{model}_trained_on_{train_ds}_evaluated_on_{test_ds1}_fold_{k}.csv")
                    results2 = pd.read_csv(f"../preds/crossval/{model}/{train_ds}/{test_ds2}/{model}_trained_on_{train_ds}_evaluated_on_{test_ds2}_fold_{k}.csv")
                    labels = [results1['label'], results2['label']]
                    preds = [results1['prediction'], results2['prediction']]
                    if is_mix:
                        results3 = pd.read_csv(f"../preds/crossval/{model}/{train_ds}/{test_ds3}/{model}_trained_on_{train_ds}_evaluated_on_{test_ds3}_fold_{k}.csv")
                        labels.append(results3['label'])
                        preds.append(results3['prediction'])
                        results4 = pd.read_csv(f"../preds/crossval/{model}/{train_ds}/{test_ds4}/{model}_trained_on_{train_ds}_evaluated_on_{test_ds4}_fold_{k}.csv")
                        labels.append(results4['label'])
                        preds.append(results4['prediction'])
                    f1s.append(f1_score(pd.concat(labels), pd.concat(preds)))
                average_f1 = sum(f1s) / 5
                stdev = np.std(f1s)
                data[test_ds].append(f"{average_f1:.2f} ± {stdev:.2f}")
    df = pd.DataFrame(data)
    os.makedirs('../results', exist_ok=True)
    df.to_csv(f'../results/output_{model}5e.csv', index=False)
