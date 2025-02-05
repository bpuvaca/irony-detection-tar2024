import pandas as pd 
import numpy as np
from Loader import parse_dataset

# irony_fp = '../datasets/crossval/irony.csv'
# sarcasm_fp = '../datasets/crossval/sarcasm.csv'
semeval_mix_fp = '../datasets/crossval/semeval_mix.csv'
isarcasm_mix_fp = '../datasets/crossval/isarcasm_mix.csv'
mix_fp = '../datasets/crossval/mix.csv'

# irony_corpus, irony_labels = parse_dataset(irony_fp, remove_hashtags=True)
# irony_parsed = pd.DataFrame({'tweet': irony_corpus, 'label': irony_labels})
# irony_parsed.to_csv('../datasets/crossval/irony_parsed.csv', index=False)

semeval_mix_corpus, semeval_mix_labels = parse_dataset(semeval_mix_fp, remove_hashtags=True)
semeval_mix_parsed = pd.DataFrame({'tweet': semeval_mix_corpus, 'label': semeval_mix_labels})
semeval_mix_parsed.to_csv('../datasets/crossval/semeval_mix_parsed.csv', index=False)

isarcasm_mix_corpus, isarcasm_mix_labels = parse_dataset(isarcasm_mix_fp, remove_hashtags=True)
isarcasm_mix_parsed = pd.DataFrame({'tweet': isarcasm_mix_corpus, 'label': isarcasm_mix_labels})
isarcasm_mix_parsed.to_csv('../datasets/crossval/isarcasm_mix_parsed.csv', index=False)

mix_corpus, mix_labels = parse_dataset(mix_fp, remove_hashtags=True)
mix_parsed = pd.DataFrame({'tweet': mix_corpus, 'label': mix_labels})
mix_parsed.to_csv('../datasets/crossval/mix_parsed.csv', index=False)


# sarcasm_corpus, sarcasm_labels = parse_dataset(sarcasm_fp, remove_hashtags=True)

# sarcasm_parsed = pd.DataFrame({'tweet': sarcasm_corpus, 'label': sarcasm_labels})

# sarcasm_parsed.to_csv('../datasets/crossval/sarcasm_parsed.csv', index=False)

dsets = ['irony', 'sarcasm']
models = ['roberta', 'bert', 'bertweet']
lhs = ['lowest', 'highest']
for model in models:
    for ds in dsets:
        for lh in lhs:
            path = f'lowest_correctness/all_predictions/{model}_trained_on_{ds}_4epoch_{lh}.csv'
            corpus, labels = parse_dataset(path, remove_hashtags=True)
            parsed = pd.DataFrame({'tweet': corpus, 'label': labels})
            parsed.to_csv(path.replace('predictions/', 'predictions/parsed_'), index=False)