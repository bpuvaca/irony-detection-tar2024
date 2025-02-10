import pandas as pd 
fold_size_dict = {
    "irony": 70,
    "polarity": 618,
    "sarcasm": 357,
    "other": 105,
    "sarcasm_mix": 357 + 105,
    "irony_mix": 70 + 618,
    "mix": 70 + 618 + 357 + 105,
    "irony_sarcasm": 70,
    "irony_polarity": 70,
    "irony_other": 70,
    "sarcasm_polarity": 357,
    "sarcasm_other": 105,
    "polarity_other": 105,
    "irony_ds": 70,
    "sarcasm_ds": 70,
    "semeval_mix": 618 + 105,
    "semeval_mix_ds": 70,
    "isarcasm_mix": 70 + 357,
    "isarcasm_mix_ds": 70,
    "mix_ds": 70,
    "irony_sarcasm_ds": 70
}

irony = pd.read_csv('../datasets/crossval/irony.csv')
sarcasm = pd.read_csv('../datasets/crossval/sarcasm.csv')
semeval_mix = pd.read_csv('../datasets/crossval/semeval_mix.csv')
isarcasm_mix = pd.read_csv('../datasets/crossval/isarcasm_mix.csv')
mix = pd.read_csv("../datasets/crossval/mix.csv")

for name, ds in zip(["irony", "sarcasm", "semeval_mix", "isarcasm_mix", "mix"], [irony, sarcasm, semeval_mix, isarcasm_mix, mix]):
    for i in range(5):
        fold = ds[i*fold_size_dict[name]:(i+1)*fold_size_dict[name]]
        for cmp_ds_name, cmp_ds in zip(["irony", "sarcasm", "semeval_mix", "isarcasm_mix", "mix"], [irony, sarcasm, semeval_mix, isarcasm_mix, mix]):
            if name == cmp_ds_name:
                continue
            cmp_fold_size = fold_size_dict[cmp_ds_name]
            cmp_fold = pd.concat([cmp_ds[:i*cmp_fold_size], cmp_ds[(i+1)*cmp_fold_size:]])
            overlap = fold['tweet'].isin(cmp_fold['tweet']).any()
            if overlap:
                print(f"\nOverlap found between {name} fold {i} and {cmp_ds_name} dataset")
                print(f"Overlapping tweets: {"\n".join(fold[fold['tweet'].isin(cmp_fold['tweet'])]['tweet'].tolist())}")
        
        

