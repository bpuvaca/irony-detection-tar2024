import pandas as pd
import math
import os

irony = pd.read_csv('../datasets/crossval/irony.csv', sep=',')
sarcasm = pd.read_csv('../datasets/crossval/sarcasm.csv', sep=',')
semeval_mix = pd.read_csv('../datasets/crossval/semeval_mix.csv', sep=',')

best_folds = {
'bertweet':
{"irony_ds" : 4,
"sarcasm_ds" : 0,
"semeval_mix_ds":2},
'roberta':
{"irony_ds":1,
"sarcasm_ds":1,
"semeval_mix_ds":0},
 'bert':
{"irony_ds":1,
"sarcasm_ds":2,
"semeval_mix_ds":1}
}

for model in ['bertweet', 'roberta', 'bert']:
    for ds_name in ["irony", "sarcasm", "semeval_mix"]:
        ds = pd.read_csv(f'../datasets/crossval/{ds_name}.csv', sep=',')
        ds_name = f"{ds_name}_ds"
        downscale_size = 350
        k = 5
        fold_size = downscale_size // k
        j_range = len(ds) // downscale_size
        i_range = math.ceil(k / j_range)
        segments = [ds[j*downscale_size:(j+1)*downscale_size] for j in range(j_range)]
        fold = best_folds[model][ds_name]
        index = fold % j_range
        os.makedirs("../datasets/cartography/ds_best_folds", exist_ok=True)
        segments[index].to_csv(f"../datasets/cartography/ds_best_folds/{model}_{ds_name}_fold{fold}.csv", index=False)