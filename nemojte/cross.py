
from transformer_main import train_and_cross_validate, cross_validate

# for model in ['bertweet', 'roberta', 'bert']:            
#     for ds in ["sarcasm_mix", "irony_mix", "polarity", "sarcasm", "mix"]:
#         train_and_cross_validate(ds, model, return_all_preds=True, folds=5, save_params=True, epochs=4)	

# for model in ['bertweet', 'roberta', 'bert']:            
#     for train_ds in ["sarcasm_crossval", "polarity_crossval", "sarcasm_mix_crossval", "irony_mix_crossval"]:
#         for test_ds in ["sarcasm_crossval", "polarity_crossval", "sarcasm_mix_crossval", "irony_mix_crossval", "irony_crossval", "other_crossval"]:
#             if train_ds != test_ds:
#                 #f"{load_from}/{model_name}_{trained_on}_fold_{i+1}"
#                 cross_validate(dataset=test_ds, model_name=model, trained_on=train_ds, 
#                                load_from=f"crossval/{model}/{train_ds}", 
#                                return_all_preds=True, folds=5)#, fold_test_dataset=test_ds.endswith("crossval"))
                

#for model in ['bert']:
#    for ds in ["sarcasm_mix", "irony_mix", "polarity", "sarcasm"]:
#        train_and_cross_validate(ds, model, return_all_preds=True, folds=5, save_params=True, epochs=5)	


# for model in ['bertweet', 'roberta', 'bert']:            
#     for train_ds in ["sarcasm_mix", "irony_mix", "polarity", "sarcasm", "mix"]:
#         for test_ds in ["sarcasm", "polarity", "irony", "other"]:
#             #if train_ds != test_ds:
#                 cross_validate(dataset=test_ds, model_name=model, trained_on=train_ds, 
#                     load_from=f"crossval/{model}/{train_ds}", 
#                     return_all_preds=True, folds=5)


#<DOWNSCALED TRAINING AND TESING
# for model in ['bertweet', 'roberta', 'bert']:            
#     for ds in ["irony", "sarcasm", "semeval_mix"]:
#         train_and_cross_validate(ds, model, return_all_preds=True, folds=5, save_params=True, epochs=4, downscale_size=350)	


# for model in ['bertweet', 'roberta', 'bert']:            
#     for train_ds in ["irony", "sarcasm", "semeval_mix"]:
#         for test_ds in ["irony", "sarcasm", "semeval_mix"]:
#             if train_ds != test_ds:
#                 train_ds = f"{train_ds}_ds"
#                 cross_validate(dataset=test_ds, model_name=model, trained_on=train_ds, 
#                     load_from=f"crossval4/{model}/{train_ds}", 
#                     return_all_preds=True, folds=5, epochs=4, downscale_size=350)
#/>


for model in ['bertweet', 'roberta', 'bert']:            
    for ds in ["semeval_mix"]:
        train_and_cross_validate(ds, model, return_all_preds=True, folds=5, save_params=True, epochs=4)	


for model in ['bertweet', 'roberta', 'bert']:            
    for train_ds in ["semeval_mix"]:
        for test_ds in ["irony", "sarcasm", "semeval_mix"]:
            if train_ds != test_ds:
                train_ds = f"{train_ds}_ds"
                cross_validate(dataset=test_ds, model_name=model, trained_on=train_ds, 
                    load_from=f"crossval4/{model}/{train_ds}", 
                    return_all_preds=True, folds=5, epochs=4)
