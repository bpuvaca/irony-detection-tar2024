
from transformer_main import train_and_cross_validate, cross_validate

# for model in ['bertweet', 'roberta', 'bert']:            
#     for ds in ["sarcasm_crossval", "polarity_crossval", "sarcasm_mix_crossval", "irony_mix_crossval"]:
#         train_and_cross_validate(ds, model, return_all_preds=True, folds=5, save_params=True, epochs=10)	


for model in ['bertweet', 'roberta', 'bert']:            
    for train_ds in ["sarcasm_crossval", "polarity_crossval", "sarcasm_mix_crossval", "irony_mix_crossval"]:
        for test_ds in ["sarcasm_crossval", "polarity_crossval", "sarcasm_mix_crossval", "irony_mix_crossval", "isarcasm_irony", "semeval_other"]:
            if train_ds != test_ds:
                cross_validate(test_ds, model, train_ds, f"../params/{model}/{train_ds}/", return_all_preds=True, folds=5, fold_test_dataset=test_ds.endswith("crossval"))


