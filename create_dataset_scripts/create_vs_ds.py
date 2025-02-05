import pandas as pd 

irony = pd.read_csv('../datasets/crossval/irony.csv', sep=',')
sarcasm = pd.read_csv('../datasets/crossval/sarcasm.csv', sep=',')

irony_sarcasm = irony[irony['label'] == 1]
irony_sarcasm.loc[:, 'label'] = 0

irony_sarcasm = pd.concat([irony_sarcasm, sarcasm[sarcasm['label'] == 1].sample(len(irony_sarcasm))])
irony_sarcasm = irony_sarcasm.sample(frac=1).reset_index(drop=True)
irony_sarcasm = irony_sarcasm[['label', 'tweet']]
irony_sarcasm.to_csv('../datasets/crossval/irony_sarcasm_ds.csv', index=False)
