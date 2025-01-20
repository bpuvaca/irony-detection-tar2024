import pandas as pd 

other = pd.read_csv('datasets/crossval/other.csv', sep=',')
polarity = pd.read_csv('datasets/crossval/polarity.csv', sep=',')

semeval_mix = pd.concat([other, polarity], ignore_index=True)
print(len(semeval_mix[semeval_mix['label'] == 1]))
print(len(semeval_mix[semeval_mix['label'] == 0]))
semeval_mix = semeval_mix.sample(frac=1).reset_index(drop=True)
semeval_mix['index'] = range(len(semeval_mix))
semeval_mix.to_csv('datasets/crossval/semeval_mix.csv', index=False)

