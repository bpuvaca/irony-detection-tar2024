import pandas as pd
from sklearn.metrics import f1_score, accuracy_score
import numpy as np
import os

for model in ['bertweet', 'roberta', 'bert']: 
    train_ds = "irony_sarcasm_ds"
    test_ds = "irony_sarcasm_ds" 
    f1s = []
    accs = []
    for k in range(1, 6):
        results = pd.read_csv(f"../preds/crossval4/{model}/{train_ds}/{test_ds}/{model}_trained_on_{train_ds}_evaluated_on_{test_ds}_fold_{k}.csv")
        f1s.append(f1_score(results['label'], results['prediction']))
        accs.append(accuracy_score(results['label'], results['prediction']))
    average_f1 = sum(f1s) / 5
    stdev = np.std(f1s)
    print(f"for {model}, acc = {np.mean(accs):.2f} +- {np.std(accs):.2f} f1 score on irony_vs_sarcasm_ds is {average_f1:.2f} ± {stdev:.2f}")
    
    train_ds = "irony_sarcasm"
    test_ds = "irony_sarcasm" 
    f1s = []
    accs = []
    for k in range(1, 6):
        results = pd.read_csv(f"../preds/crossval4/{model}/{train_ds}/{test_ds}/{model}_trained_on_{train_ds}_evaluated_on_{test_ds}_fold_{k}.csv")
        f1s.append(f1_score(results['label'], results['prediction']))
        accs.append(accuracy_score(results['label'], results['prediction']))
    average_f1 = sum(f1s) / 5
    stdev = np.std(f1s)
    print(f"for {model}, acc = {np.mean(accs):.2f} +- {np.std(accs):.2f} f1 score on irony_vs_sarcasm is {average_f1:.2f} ± {stdev:.2f}")