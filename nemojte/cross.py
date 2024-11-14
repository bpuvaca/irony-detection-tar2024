
from transformer_main import cross_validate

for model in ['bertweet', 'roberta', 'bert']:            
    for train in ["isarcasm_sarc_crossval", "semeval_polarity_crossval"]:
        for valid in ["isarcasm_sarc_crossval", "semeval_polarity_crossval"]:
                cross_validate(train, model, eval_on=(None if train==valid else valid), save_to=None, folds=7)

