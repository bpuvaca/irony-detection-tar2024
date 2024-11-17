import pandas as pd

irony = pd.read_csv('datasets/iSarcasm/irony_test.csv', sep=',')
sarcasm = pd.read_csv('datasets/crossval/sarcasm.csv', sep=',')
polarity = pd.read_csv('datasets/crossval/polarity.csv', sep=',')
other = pd.read_csv('datasets/SemEval2018/other_test.csv', sep=',')

sarcasm_mix = pd.concat([other, sarcasm])
irony_mix = pd.concat([polarity, irony])

sarcasm_mix.sample(frac=1).reset_index(drop=True)
irony_mix.sample(frac=1).reset_index(drop=True)

sarcasm_mix.to_csv('datasets/crossval/sarcasm_mix.csv', index=False)
irony_mix.to_csv('datasets/crossval/irony_mix.csv', index=False)



