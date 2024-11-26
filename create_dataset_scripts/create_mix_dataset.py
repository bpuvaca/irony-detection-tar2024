import pandas as pd

irony = pd.read_csv('../datasets/crossval/irony.csv')
polarity = pd.read_csv('../datasets/crossval/polarity.csv')
sarcasm = pd.read_csv('../datasets/crossval/sarcasm.csv')
other = pd.read_csv('../datasets/crossval/other.csv')

mix = pd.concat([irony, polarity, sarcasm, other])
mix.to_csv('../datasets/crossval/mix.csv', index=False)
