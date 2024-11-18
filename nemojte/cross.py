
from transformer_main import train_and_cross_validate

for model in ['bertweet', 'roberta', 'bert']:            
    for ds in ["sarcasm_crossval", "polarity_crossval", "sarcasm_mix_crossval", "irony_mix_crossval"]:
        train_and_cross_validate(ds, model, return_all_preds=True, folds=5, save_params=True, epochs=10)	



