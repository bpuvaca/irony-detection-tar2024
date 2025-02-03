import pandas as pd 
import numpy as np

def create_liwc_data(dir_path, models, dsets):
    for model in models:
        for train_ds in dsets:
            for test_ds in dsets:
                path = f'{dir_path}/{model}/{train_ds}/{test_ds}/{model}_trained_on_{train_ds}_evaluated_on_{test_ds}_fold_'
                data = pd.read_csv(f'{path}1.csv')
                for i in range(2, 6):
                    data = pd.concat([data, pd.read_csv(f'{path}{i}.csv')], ignore_index=True)

                tn = data[(data['label'] == 0) & (data['prediction'] == 0)]
                fp = data[(data['label'] == 0) & (data['prediction'] == 1)]
                fn = data[(data['label'] == 1) & (data['prediction'] == 0)]
                tp = data[(data['label'] == 1) & (data['prediction'] == 1)]

                fp.to_csv(f'{dir_path}/{model}/{train_ds}/{test_ds}/fp.csv', index=False)
                fn.to_csv(f'{dir_path}/{model}/{train_ds}/{test_ds}/fn.csv', index=False)
                tp.to_csv(f'{dir_path}/{model}/{train_ds}/{test_ds}/tp.csv', index=False)
                tn.to_csv(f'{dir_path}/{model}/{train_ds}/{test_ds}/tn.csv', index=False)

models = ['roberta', 'bert', 'bertweet']
# dsets = ['semeval_mix_ds', 'sarcasm_ds', 'irony_ds']

# create_liwc_data('../preds/crossval4', models, dsets)


def prepare_cart_for_liwc(dir_path, model, dsets):
    for model in models:
        for train_ds in dsets:
            path = f'{dir_path}/{model}_trained_on_{train_ds}_4epoch.csv'
            data = pd.read_csv(path)
            lowest = data[data['mean_correctness'] <= 0.5]
            highest = data[data['mean_correctness'] > 0.5]
            lowest.to_csv(f'{dir_path}/{model}_trained_on_{train_ds}_4epoch_lowest.csv', index=False)
            highest.to_csv(f'{dir_path}/{model}_trained_on_{train_ds}_4epoch_highest.csv', index=False)
            
prepare_cart_for_liwc('lowest_correctness/all_predictions', models, ['semeval_mix', 'irony', 'sarcasm'])