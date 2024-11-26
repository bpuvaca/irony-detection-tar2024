import pandas as pd

irony = pd.read_csv('../datasets/crossval/irony.csv')
polarity = pd.read_csv('../datasets/crossval/polarity.csv')
sarcasm = pd.read_csv('../datasets/crossval/sarcasm.csv')
other = pd.read_csv('../datasets/crossval/other.csv')

# mix = pd.concat([irony, polarity, sarcasm, other])
# mix.to_csv('../datasets/crossval/mix.csv', index=False)

mix = pd.read_csv('../datasets/crossval/mix.csv')
# print(len(mix[mix['label'] == 1]))
# print(len(mix[mix['label'] == 0]))
# print(len(mix))
# print(len(mix['tweet'].unique()))
tc = mix.tweet.value_counts()
tc = tc[tc > 1]

irony_tc = irony.tweet.value_counts()
sarcasm_tc = sarcasm.tweet.value_counts()
polarity_tc = polarity.tweet.value_counts()
other_tc = other.tweet.value_counts()


for tweet in tc.items():
    print(f'Tweet \n"{tweet[0]}"\n occurs {tweet[1]} times in the dataset')
    count = irony_tc.get(tweet[0], 0)
    if count != 0: print(f'Appears {count} times in Irony')
    count = sarcasm_tc.get(tweet[0], 0)
    if count != 0: print(f'Appears {count} times in Sarcasm')
    count = polarity_tc.get(tweet[0], 0)
    if count != 0: print(f'Appears {count} times in Polarity')
    count = other_tc.get(tweet[0], 0)
    if count != 0: print(f'Appears {count} times in Other')
    print("\n")

