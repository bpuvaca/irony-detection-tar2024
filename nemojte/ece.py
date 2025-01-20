import pandas as pd
import numpy as np
import os
# dict = {"label":[0, 1, 0, 0, 0, 0, 1, 1, 1],
#     "prediction": [0, 1, 1, 0, 1, 0, 1, 0, 1],
#     "probability": [0.22, 0.64, 0.92, 0.42, 0.51, 0.15, 0.7, 0.37, 0.83]}

# data = pd.DataFrame(dict)


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

if __name__ == "__main__":
    print("blabla")
    n_bins = 5
    for model in ['bertweet', 'roberta', 'bert']:     
        eces = {}
        eces['trained_on'] = []       
        for train_ds in ["polarity", "sarcasm", "mix", "sarcasm_mix", "irony_mix"]:
            eces['trained_on'].append(train_ds)
            for test_ds in ["sarcasm", "polarity", "irony", "other"]:
                if test_ds not in eces.keys():
                    eces[test_ds] = []
                data = pd.read_csv(f"../preds/crossval4/{model}/{train_ds}/{test_ds}/{model}_trained_on_{train_ds}_evaluated_on_{test_ds}_fold_1.csv")
                for i in range(2, 6):
                    data = pd.concat([data, pd.read_csv(f"../preds/crossval4/{model}/{train_ds}/{test_ds}/{model}_trained_on_{train_ds}_evaluated_on_{test_ds}_fold_{i}.csv")])
                ece = get_ece(data, n_bins=5)
                eces[test_ds].append(ece)
        df = pd.DataFrame(eces)
        os.makedirs(f'../results/eces/{n_bins}', exist_ok=True)
        df.to_csv(f'../results/eces/{n_bins}/ece_{model}_4e_{n_bins}bins.csv', index=False)
                
