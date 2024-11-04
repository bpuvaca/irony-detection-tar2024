import pandas as pd
from transformer_main import train_and_evaluate

filename = 'transformer_results.csv'
columns = ['irony', 'sarcasm', 'mix']
df = pd.DataFrame(columns=columns)

datasets = ['irony', 'sarcasm', 'mix']
models = ['roberta', 'bertweet', 'bert']
params_suffix = ''

for model in models:
    for dataset in datasets:
        for i in range(1,6):
            name = f"{model}_{dataset}_{i}{params_suffix}"
            train_and_evaluate(dataset, model, f"{model}_{dataset}_{i}{params_suffix}", None, 'irony sarcasm mix', False, False, name, df)

print(df)
df.to_csv('../transformer_results.csv', index=True)