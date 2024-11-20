import pandas as pd

irony = pd.read_csv('datasets/crossval/irony.csv', sep=',')
sarcasm = pd.read_csv('datasets/crossval/sarcasm.csv', sep=',')
polarity = pd.read_csv('datasets/crossval/polarity.csv', sep=',')
other = pd.read_csv('datasets/crossval/other.csv', sep=',')


k = 5
fold_size_sarcasm_mix = int((len(sarcasm) + len(other)) / k)
fold_size_sarcasm = int(len(sarcasm) / k)
fold_size_other = int(len(other) / k)
if fold_size_other != fold_size_sarcasm_mix - fold_size_sarcasm:
    print("Greska sarcasm")

fold_size_irony_mix = int((len(irony) + len(polarity)) / k)
fold_size_polarity = int(len(polarity) / k)
fold_size_irony = int(len(irony) / k)
if fold_size_irony != fold_size_irony_mix - fold_size_polarity:
    print("Greska irony")

sarcasm_mix = pd.DataFrame()

irony_mix = pd.DataFrame()
for i in range(k):
    if i == k - 1:
        sarcasm_mix = pd.concat([sarcasm_mix, sarcasm[i*fold_size_sarcasm:], other[i*fold_size_other:]])
        irony_mix = pd.concat([irony_mix, polarity[i*fold_size_polarity:], irony[i*fold_size_irony:]])
    else:
        sarcasm_mix = pd.concat([sarcasm_mix, sarcasm[i*fold_size_sarcasm:(i+1)*fold_size_sarcasm], other[i*fold_size_other:(i+1)*fold_size_other]])
        irony_mix = pd.concat([irony_mix, polarity[i*fold_size_polarity:(i+1)*fold_size_polarity], irony[i*fold_size_irony:(i+1)*fold_size_irony]])
 
sarcasm_mix['index'] = range(len(sarcasm_mix))
irony_mix['index'] = range(len(irony_mix))      

# sarcasm_mix = pd.concat([other, sarcasm])
# irony_mix = pd.concat([polarity, irony])

# sarcasm_mix.sample(frac=1).reset_index(drop=True)
# irony_mix.sample(frac=1).reset_index(drop=True)

sarcasm_mix.to_csv('datasets/crossval/sarcasm_mix.csv', index=False)
irony_mix.to_csv('datasets/crossval/irony_mix.csv', index=False)



