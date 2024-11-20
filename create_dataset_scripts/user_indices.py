import pandas as pd
from collections import Counter
import re

df = pd.read_csv('datasets/iSarcasm/sarcasm.csv', sep=',')
df.columns = ['tweet', 'label', 'index']

# Extract user mentions from the 'tweet' column
user_mentions = df['tweet'].apply(lambda x: re.findall(r'@\w+', x))

# Flatten the list of lists and count the occurrences of each user mention
user_mentions_flat = [mention for sublist in user_mentions for mention in sublist]
mention_counts = Counter(user_mentions_flat)

# Convert the counts to a DataFrame
mention_counts_df = pd.DataFrame(mention_counts.items(), columns=['user_mention', 'count'])

# Save the result to a CSV file
# Sort the DataFrame by 'count' in descending order
mention_counts_df = mention_counts_df.sort_values(by='count', ascending=False)

# Save the sorted result to a CSV file
mention_counts_df.to_csv('sarcasm_user_mentions_counts_sorted.csv', index=False)


df = pd.read_csv('datasets/SemEval2018/polarity.csv', sep=',')
df.columns = ['index', 'label', 'tweet']

# Extract user mentions from the 'tweet' column
user_mentions = df['tweet'].apply(lambda x: re.findall(r'@\w+', x))

# Flatten the list of lists and count the occurrences of each user mention
user_mentions_flat = [mention for sublist in user_mentions for mention in sublist]
mention_counts = Counter(user_mentions_flat)

# Convert the counts to a DataFrame
mention_counts_df = pd.DataFrame(mention_counts.items(), columns=['user_mention', 'count'])

# Save the result to a CSV file
# Sort the DataFrame by 'count' in descending order
mention_counts_df = mention_counts_df.sort_values(by='count', ascending=False)

# Save the sorted result to a CSV file
mention_counts_df.to_csv('polarityy_user_mentions_counts_sorted.csv', index=False)




