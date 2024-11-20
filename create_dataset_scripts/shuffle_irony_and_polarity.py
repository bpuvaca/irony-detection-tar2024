import pandas as pd
irony = pd.read_csv('datasets/crossval/irony_not_shuffled.csv', sep=',')
other = pd.read_csv('datasets/crossval/other_not_shuffled.csv', sep=',')

irony = irony.sample(frac=1).reset_index(drop=True)
other = other.sample(frac=1).reset_index(drop=True)

irony.to_csv('datasets/crossval/irony.csv', index=False)
other.to_csv('datasets/crossval/other.csv', index=False)