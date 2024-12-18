from transformer_main import train_and_cross_validate, cross_validate

for model in ['bertweet', 'roberta', 'bert']:            
    for ds in ["irony_sarcasm"]:#, "irony_mix", "polarity", "sarcasm", "mix"]:
        train_and_cross_validate(ds, model, return_all_preds=True, folds=5, save_params=True, epochs=4)	


