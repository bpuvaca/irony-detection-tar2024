import pandas as pd

csv1 = "irony_train_taskA.csv"
csv2 = "irony_validation_taskA.csv"
csv3 = "irony_test_taskA.csv"

df1 = pd.read_csv(csv1)
df2 = pd.read_csv(csv2)
df3 = pd.read_csv(csv3)

df1['source'] = 'csv1'
df2['source'] = 'csv2'
df3['source'] = 'csv3'

combined_df = pd.concat([df1, df2, df3])

duplicates = combined_df[combined_df.duplicated(['tweet'], keep=False)]

duplicate_groups = duplicates.groupby('tweet')['source'].apply(list).reset_index()

result = pd.merge(duplicates, duplicate_groups, on='tweet')

result = result.drop_duplicates(subset=['tweet', 'label'])

result.columns = ['index', 'tweet', 'label', 'original_source', 'duplicate_sources']

print(result)

result.to_csv("duplicate_tweets.csv", index=False)
