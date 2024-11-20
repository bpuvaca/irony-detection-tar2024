import pandas as pd

predictions = {}

for model in ['roberta', 'bertweet', 'bert']:
    for train_set in ['irony_mix_crossval', 'polarity_crossval', 'sarcasm_crossval', 'sarcasm_mix_crossval']:
        for test_set in ['irony_crossval', 'other_crossval', 'polarity_crossval', 'sarcasm_crossval']:
            for fold in range(1,6):
                file_path = f'../preds/crossval/{model}/{train_set}/{test_set}/{model}_trained_on_{train_set}_evaluated_on_{test_set}_fold_{fold}.csv'
                df = pd.read_csv(file_path)
                for _, row in df.iterrows():
                    tweet = row['tweet']
                    dataset = row['dataset']
                    prediction = row['prediction']
                    label = row['label']
                    key = (dataset, tweet, label)
                    if key not in predictions:
                        predictions[key] = {}
                    model_key = f'{model}_on_{train_set}'
                    predictions[key][model_key] = prediction
            
key_names = []
for model in ['roberta', 'bertweet', 'bert']:
    for train_set in ['irony_mix_crossval', 'polarity_crossval', 'sarcasm_crossval', 'sarcasm_mix_crossval']:
        model_key = f'{model}_on_{train_set}'
        key_names.append(model_key)

data = []
for tweet, preds in predictions.items():
    data.append([tweet[0], tweet[1], tweet[2]] + [preds[key] for key in key_names])

columns = ['dataset', 'tweet', 'label'] + key_names
df = pd.DataFrame(data, columns=columns)
df.to_csv('ALL_PREDICTIONS.csv', index=False)
           