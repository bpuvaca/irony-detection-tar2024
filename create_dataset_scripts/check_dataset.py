import pandas as pd

# Load the datasets
csv1 = "irony_train_taskA.csv"
csv2 = "irony_validation_taskA.csv"
csv3 = "irony_test_taskA.csv"

df1 = pd.read_csv(csv1)
df2 = pd.read_csv(csv2)
df3 = pd.read_csv(csv3)

# Adding a source column to identify the origin of the tweets
df1['source'] = 'csv1'
df2['source'] = 'csv2'
df3['source'] = 'csv3'

# Concatenate the dataframes
combined_df = pd.concat([df1, df2, df3])

# Find duplicates
duplicates = combined_df[combined_df.duplicated(['tweet'], keep=False)]

# Group by tweet and aggregate the sources
duplicate_groups = duplicates.groupby('tweet')['source'].apply(list).reset_index()

# Merge the grouped sources back with the duplicate records to include all columns
result = pd.merge(duplicates, duplicate_groups, on='tweet')

# Drop the duplicate 'source' column and keep unique rows
result = result.drop_duplicates(subset=['tweet', 'label'])

# Rename columns for clarity
result.columns = ['index', 'tweet', 'label', 'original_source', 'duplicate_sources']

# Display the result
print(result)

# Optionally, save the result to a new CSV file
result.to_csv("duplicate_tweets.csv", index=False)
