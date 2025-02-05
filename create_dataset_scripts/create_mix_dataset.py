import pandas as pd

irony = pd.read_csv('../datasets/crossval/irony.csv')
polarity = pd.read_csv('../datasets/crossval/polarity.csv')
sarcasm = pd.read_csv('../datasets/crossval/sarcasm.csv')
other = pd.read_csv('../datasets/crossval/other.csv')

isarcasm_mix = pd.concat([irony, sarcasm])
isarcasm_mix = isarcasm_mix.sample(frac=1).reset_index(drop=True)
isarcasm_mix['index'] = range(len(isarcasm_mix))
isarcasm_mix.to_csv('../datasets/crossval/isarcasm_mix.csv', index=False)

sarcasm_ds = pd.read_csv('../datasets/crossval/sarcasm_ds.csv')



isarcasm_mix_ds = pd.concat([irony[irony], sarcasm])




# k = 5
# fold_size_irony = int(len(irony) / k)
# fold_size_polarity = int(len(polarity) / k)
# fold_size_sarcasm = int(len(sarcasm) / k)
# fold_size_other = int(len(other) / k)

# mix = pd.DataFrame()

# for i in range(k):
#     if i == k - 1:
#         mix = pd.concat([mix, irony[i*fold_size_irony:], polarity[i*fold_size_polarity:], sarcasm[i*fold_size_sarcasm:], other[i*fold_size_other:]])
#     else:
#         mix = pd.concat([mix, irony[i*fold_size_irony:(i+1)*fold_size_irony], 
#                                  polarity[i*fold_size_polarity:(i+1)*fold_size_polarity], 
#                                  sarcasm[i*fold_size_sarcasm:(i+1)*fold_size_sarcasm], 
#                                  other[i*fold_size_other:(i+1)*fold_size_other]])
        
 
# mix['index'] = range(len(mix))

# # Rearrange columns
# columns_order = ["index", "label", "tweet"]
# mix = mix[columns_order]
# mix.to_csv('../datasets/crossval/mix.csv', index=False)

# mix = pd.read_csv('../datasets/crossval/mix.csv')
# # print(len(mix[mix['label'] == 1]))
# # print(len(mix[mix['label'] == 0]))
# # print(len(mix))
# # print(len(mix['tweet'].unique()))
# tc = mix.tweet.value_counts()
# tc = tc[tc > 1]

# irony_tc = irony.tweet.value_counts()
# sarcasm_tc = sarcasm.tweet.value_counts()
# polarity_tc = polarity.tweet.value_counts()
# other_tc = other.tweet.value_counts()


# for tweet in tc.items():
#     print(f'Tweet \n"{tweet[0]}"\n occurs {tweet[1]} times in the dataset')
#     count = irony_tc.get(tweet[0], 0)
#     if count != 0: print(f'Appears {count} times in Irony')
#     count = sarcasm_tc.get(tweet[0], 0)
#     if count != 0: print(f'Appears {count} times in Sarcasm')
#     count = polarity_tc.get(tweet[0], 0)
#     if count != 0: print(f'Appears {count} times in Polarity')
#     count = other_tc.get(tweet[0], 0)
#     if count != 0: print(f'Appears {count} times in Other')
#     print("\n")

