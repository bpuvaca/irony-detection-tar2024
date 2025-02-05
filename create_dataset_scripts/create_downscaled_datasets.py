import numpy as np
import pandas as pd

# irony = pd.read_csv('../datasets/crossval/irony.csv', sep=',')
# sarcasm = pd.read_csv('../datasets/crossval/sarcasm.csv', sep=',')
# semeval_mix = pd.read_csv('../datasets/crossval/semeval_mix.csv', sep=',')
# downscale_size = len(irony)

# sarcasm_ds = pd.concat([sarcasm[sarcasm['label'] == 1].sample(n=downscale_size//2, random_state=42), sarcasm[sarcasm['label'] == 0].sample(n=downscale_size//2, random_state=42)])
# sarcasm_ds = sarcasm_ds.sample(frac=1).reset_index(drop=True)
# sarcasm_ds.to_csv('../datasets/crossval/sarcasm_ds.csv', index=False)

# semeval_mix_ds = pd.concat([semeval_mix[semeval_mix['label'] == 1].sample(n=downscale_size//2, random_state=42), semeval_mix[semeval_mix['label'] == 0].sample(n=downscale_size//2, random_state=42)])
# semeval_mix_ds = semeval_mix_ds.sample(frac=1).reset_index(drop=True)
# semeval_mix_ds.to_csv('../datasets/crossval/semeval_mix_ds.csv', index=False)

# isarcasm_mix_ds = pd.DataFrame()
# for i in range(5):
#     size = 35
#     fold_size = 70
#     isarcasm_mix_ds = pd.concat([isarcasm_mix_ds, irony[i*fold_size:i*fold_size+size], sarcasm_ds[i*fold_size:i*fold_size+size]])

# isarcasm_mix_ds.to_csv('../datasets/crossval/isarcasm_mix_ds.csv', index=False)    

isarcasm_mix_ds = pd.read_csv('../datasets/crossval/isarcasm_mix_ds.csv', sep=',')
irony_ds = pd.read_csv('../datasets/crossval/irony.csv', sep=',')
sarcasm_ds = pd.read_csv('../datasets/crossval/sarcasm_ds.csv', sep=',')
semeval_mix_ds = pd.read_csv('../datasets/crossval/semeval_mix_ds.csv', sep=',')

print(isarcasm_mix_ds.shape)
print(semeval_mix_ds.shape)
mix_ds = pd.DataFrame()
for i in range(5):
    size = 35
    fold_size = 70
    mix_ds = pd.concat([mix_ds, isarcasm_mix_ds[i*fold_size:(i+1)*fold_size].sample(size), semeval_mix_ds[i*fold_size:(i+1)*fold_size].sample(size)])

mix_ds.to_csv('../datasets/crossval/mix_ds.csv', index=False)    



