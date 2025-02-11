from transformer_main import train_and_cross_validate, cross_validate

# for model in ['bertweet', 'roberta', 'bert']:  
#     for ds in (["isarcasm_mix", "mix"] if model == 'bertweet' else ["irony", "sarcasm", "semeval_mix", "isarcasm_mix", "mix"]):
#         train_and_cross_validate(ds, model, return_all_preds=True, folds=5, save_params=True, epochs=4)

for model in ['bertweet', 'roberta', 'bert']:            
    for train_ds in ["irony", "sarcasm", "semeval_mix", "isarcasm_mix", "mix"]:
        for test_ds in ["irony", "sarcasm", "semeval_mix", "isarcasm_mix", "mix"]:
            if train_ds != test_ds:
                cross_validate(dataset=test_ds, model_name=model, trained_on=train_ds, 
                    load_from=f"crossval4/{model}/{train_ds}", 
                    return_all_preds=True, folds=5, epochs=4)
