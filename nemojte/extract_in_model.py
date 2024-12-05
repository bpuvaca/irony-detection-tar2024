import pandas as pd
import os

models = ['bert', 'bertweet', 'roberta']

base_dir = 'lowest_correctness'

for model in models:

    polarity_file = pd.read_csv(os.path.join(base_dir, f'{model}_trainedon_polarity_4epoch.csv'))
    sarcasm_file = pd.read_csv(os.path.join(base_dir, f'{model}_trainedon_sarcasm_4epoch.csv'))

    overlapping_tweets = pd.merge(polarity_file, sarcasm_file, on='tweet', how='inner')
    output_file_path = os.path.join(base_dir, f'{model}_overlapping_tweets.csv')
    overlapping_tweets.to_csv(output_file_path, index=False)

    print(f"Overlapping tweets for {model} saved to: {output_file_path}")
