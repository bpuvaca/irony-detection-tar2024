import pandas as pd
import numpy as np

probabilities = {}

for model in ['roberta', 'bertweet', 'bert']:
    for train_set in ['irony_mix', 'polarity', 'sarcasm', 'sarcasm_mix']:
        for test_set in ['irony', 'other', 'polarity', 'sarcasm']:
            for fold in range(1,6):
                file_path = f'../preds/crossval4/{model}/{train_set}/{test_set}/{model}_trained_on_{train_set}_evaluated_on_{test_set}_fold_{fold}.csv'
                df = pd.read_csv(file_path)
                for _, row in df.iterrows():
                    tweet = row['tweet']
                    dataset = row['dataset']
                    probability = row['probability']
                    label = row['label']
                    key = (dataset, tweet, label)
                    if key not in probabilities:
                        probabilities[key] = {}
                    model_key = f'{model}_on_{train_set}'
                    probabilities[key][model_key] = probability
            
key_names = []
for model in ['roberta', 'bertweet', 'bert']:
    for train_set in ['irony_mix', 'polarity', 'sarcasm', 'sarcasm_mix']:
        model_key = f'{model}_on_{train_set}'
        key_names.append(model_key)

data = []
for tweet, preds in probabilities.items():
    data.append([tweet[0], tweet[1], tweet[2]] + [preds[key] for key in key_names])

columns = ['dataset', 'tweet', 'label'] + key_names
df = pd.DataFrame(data, columns=columns)
df.to_csv('ALL_PROBABILITIES_4.csv', index=False)
           