import pandas as pd 
from transformer_main import train_and_cross_validate, cross_validate

for model in ['bert', 'bertweet', 'roberta']:
    train_and_cross_validate('irony_sarcasm', model, return_all_preds=True, folds=5, save_params=True, epochs=4)
