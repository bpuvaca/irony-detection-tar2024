import pandas as pd
from sklearn.model_selection import train_test_split

# Load the dataset
df = pd.read_csv('datasets/train/SemEval2018-T3-train-taskB_emoji_ironyHashtags.txt', sep='\t', header=None)

columns = ['index', 'label', 'tweet']

df.columns = columns

df_non_ironic = df[df['label'] == '0']
df_polarity = df[df['label'] == '1']

print("Number of rows in df_non_ironic:", len(df_non_ironic))
print("Number of rows in df_polarity:", len(df_polarity))


df_test = pd.read_csv('datasets/goldtest_TaskB/SemEval2018-T3_gold_test_taskB_emoji.txt', sep='\t', header=None)

columns = ['index', 'label', 'tweet']

df_test.columns = columns

df_non_ironic_test = df_test[df_test['label'] == '0']
df_polarity_test = df_test[df_test['label'] == '1']

print("Number of rows in df_non_ironic_test:", len(df_non_ironic_test))
print("Number of rows in df_polarity_test:", len(df_polarity_test))
# Display the first few rows of the dataframe

df_polarity = pd.concat([df_polarity, df_polarity_test])
df_polarity.dropna(inplace=True, subset=['tweet'])
df_polarity = pd.concat([df_polarity, df_non_ironic.head(len(df_polarity))])
df_polarity = df_polarity.sample(frac=1).reset_index(drop=True)
df_polarity['index'] = range(len(df_polarity))
df_polarity.to_csv('datasets/SemEval2018/polarity.csv', index=False)


# polarity_test = pd.concat([df_polarity_test, df_non_ironic_test.head(len(df_polarity_test))])
# polarity_test['index'] = range(len(polarity_test))
# polarity_test.to_csv('../datasets/SemEval2018/polarity_test.csv', index=False)
# start_index += len(df_polarity_test)

# other = pd.concat([df_other, df_other_test])
# other['label'] = 1
# other_test = pd.concat([other, df_non_ironic_test.iloc[start_index:start_index + len(other)]])
# other_test['index'] = range(len(other_test))
# other_test.to_csv('../datasets/SemEval2018/other_test.csv', index=False)

# print(len(polarity_train))
# print(len(polarity_val))
# print(len(polarity_test))
# print(len(other_test))
