import pandas as pd

fold_size_dict = {
    "irony": 70,
    "polarity": 618,
    "sarcasm": 357,
    "other": 105,
    "sarcasm_mix": 357 + 105,
    "irony_mix": 70 + 618,
    "mix": 70 + 618 + 357 + 105,
    "irony_sarcasm": 70,
    "irony_polarity": 70,
    "irony_other": 70,
    "sarcasm_polarity": 357,
    "sarcasm_other": 105,
    "polarity_other": 105,
    "irony_ds": 70,
    "sarcasm_ds": 70,
    "semeval_mix": 618 + 105,
    "semeval_mix_ds": 70,
    "isarcasm_mix": 70 + 357,
    "isarcasm_mix_ds": 70,
    "mix_ds": 70,
    "irony_sarcasm_ds": 70
}

irony = pd.read_csv('../datasets/crossval/irony.csv')
polarity = pd.read_csv('../datasets/crossval/polarity.csv')
sarcasm = pd.read_csv('../datasets/crossval/sarcasm.csv')
other = pd.read_csv('../datasets/crossval/other.csv')
semeval_mix = pd.read_csv('../datasets/crossval/semeval_mix.csv')

isarcasm_mix = pd.DataFrame()
for i in range(5):
    irony_fold_size = fold_size_dict["irony"]
    sarcasm_fold_size = fold_size_dict["sarcasm"]
    isarcasm_mix = pd.concat([isarcasm_mix, irony[i*irony_fold_size:(i+1)*irony_fold_size], sarcasm[i*sarcasm_fold_size:(i+1)*sarcasm_fold_size]])

isarcasm_mix.to_csv('../datasets/crossval/isarcasm_mix.csv', index=False)

mix = pd.DataFrame()
for i in range(5):
    semeval_mix_fold_size = fold_size_dict["sememeval_mix"]
    isarcasm_mix_fold_size = fold_size_dict["isarcasm_mix"]
    mix = pd.concat([mix, semeval_mix[i*semeval_mix_fold_size:(i+1)*semeval_mix_fold_size], isarcasm_mix[i*isarcasm_mix_fold_size:(i+1)*isarcasm_mix_fold_size]])

mix.to_csv('../datasets/crossval/mix.csv', index=False)

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

