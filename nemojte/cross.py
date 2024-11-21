
from transformer_main import train_and_cross_validate, cross_validate

# for model in ['bertweet', 'roberta', 'bert']:            
#     for ds in ["sarcasm_crossval", "polarity_crossval", "sarcasm_mix_crossval", "irony_mix_crossval"]:
#         train_and_cross_validate(ds, model, return_all_preds=True, folds=5, save_params=True, epochs=10)	


# for model in ['bertweet', 'roberta', 'bert']:            
#     for train_ds in ["sarcasm_crossval", "polarity_crossval", "sarcasm_mix_crossval", "irony_mix_crossval"]:
#         for test_ds in ["sarcasm_crossval", "polarity_crossval", "sarcasm_mix_crossval", "irony_mix_crossval", "irony_crossval", "other_crossval"]:
#             if train_ds != test_ds:
#                 #f"{load_from}/{model_name}_{trained_on}_fold_{i+1}"
#                 cross_validate(dataset=test_ds, model_name=model, trained_on=train_ds, 
#                                load_from=f"crossval/{model}/{train_ds}", 
#                                return_all_preds=True, folds=5)#, fold_test_dataset=test_ds.endswith("crossval"))
                

#for model in ['bertweet', 'roberta', 'bert']:
#for model in ['bert']:
#    for ds in ["sarcasm_mix_crossval", "irony_mix_crossval"]:
#        train_and_cross_validate(ds, model, return_all_preds=True, folds=5, save_params=True, epochs=10)	


# for model in ['bertweet', 'roberta', 'bert']:            
#     for train_ds in ["sarcasm_crossval", "polarity_crossval", "sarcasm_mix_crossval", "irony_mix_crossval"]:
#         for test_ds in ["sarcasm_crossval", "polarity_crossval", "irony_crossval", "other_crossval"]:
#             #f"{load_from}/{model_name}_{trained_on}_fold_{i+1}"
#             if train_ds != test_ds:
#                 cross_validate(dataset=test_ds, model_name=model, trained_on=train_ds, 
#                     load_from=f"crossval/{model}/{train_ds}", 
#                     return_all_preds=True, folds=5)#, fold_test_dataset=test_ds.endswith("crossval"))

for model in ['roberta']:            
    for train_ds in ["sarcasm_crossval", "polarity_crossval", "sarcasm_mix_crossval", "irony_mix_crossval"]:
        for test_ds in ["sarcasm_crossval", "polarity_crossval", "irony_crossval", "other_crossval"]:
            #f"{load_from}/{model_name}_{trained_on}_fold_{i+1}"
            if train_ds != test_ds:
                cross_validate(dataset=test_ds, model_name=model, trained_on=train_ds, 
                    load_from=f"crossval/{model}/{train_ds}", 
                    return_all_preds=True, folds=5)#, fold_test_dataset=test_ds.endswith("crossval"))

