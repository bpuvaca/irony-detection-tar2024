import pandas as pd
from sklearn.metrics import f1_score
import numpy as np
import os

for model in ['bertweet', 'roberta', 'bert']:  
    data = {'trained_on': []}          
    train_dsets = ["irony_ds", "sarcasm_ds", "semeval_mix_ds"]
    for train_ds in train_dsets:
        data['trained_on'].append(train_ds)
        for test_ds in ["irony_ds", "sarcasm_ds", "semeval_mix_ds"]:
            if test_ds not in data.keys():
                data[test_ds] = []
            f1s = []
            for k in range(1, 6):
                results = pd.read_csv(f"../preds/crossval4/{model}/{train_ds}/{test_ds}/{model}_trained_on_{train_ds}_evaluated_on_{test_ds}_fold_{k}.csv")
                f1s.append(f1_score(results['label'], results['prediction']))
            if train_ds == test_ds:
                best_f1_index = np.argmax(f1s)
                print(f"Best fold for {train_ds} is {best_f1_index}")
            average_f1 = sum(f1s) / 5
            stdev = np.std(f1s)
            data[test_ds].append(f"{average_f1:.2f} ± {stdev:.2f}")
    df = pd.DataFrame(data)
    os.makedirs('../results/ds', exist_ok=True)
    df.to_csv(f'../results/ds/output_{model}4e.csv', index=False)
