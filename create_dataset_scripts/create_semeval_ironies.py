import pandas as pd
from sklearn.model_selection import train_test_split

# Load the dataset
df = pd.read_csv('../datasets/train/SemEval2018-T3-train-taskB_emoji_ironyHashtags.txt', sep='\t', header=None)

columns = ['index', 'label', 'tweet']

df.columns = columns
print(df.head())

df_non_ironic = df[df['label'] == '0']
df_polarity = df[df['label'] == '1']
df_polarity_train, df_polarity_val = train_test_split(df_polarity, test_size=0.2, random_state=42)
df_sit = df[df['label'] == '2']
df_other = df[df['label'] == '3']
print(len(df_polarity))
print()
# Display the first few rows of the dataframe

start_index = 0

polarity_train = pd.concat([df_polarity_train, df_non_ironic.head(len(df_polarity_train))])
polarity_train['index'] = range(len(polarity_train))
polarity_train.to_csv('../datasets/SemEval2018/polarity_train.csv', index=False)
start_index += len(df_polarity_train)

polarity_val = pd.concat([df_polarity_val, df_non_ironic.iloc[start_index:start_index + len(df_polarity_val)]])
polarity_val['index'] = range(len(polarity_val))
polarity_val.to_csv('../datasets/SemEval2018/polarity_valid.csv', index=False)
start_index += len(df_polarity_val)

# sit = pd.concat(df_sit, df_non_ironic.iloc[start_index:start_index + len(df_sit)])
# sit['index'] = range(len(sit))
# sit.to_csv('../datasets/SemEval2018/sit_train.csv', index=False)
# start_index += len(df_sit)

# other = pd.concat(df_other, df_non_ironic.iloc[start_index:start_index + len(df_other)])
# other['index'] = range(len(other))
# other.to_csv('../datasets/SemEval2018/other_train.csv', index=False)
# start_index += len(df_other)

# Load the dataset
df_test = pd.read_csv('../datasets/goldtest_TaskB/SemEval2018-T3_gold_test_taskB_emoji.txt', sep='\t', header=None)

columns = ['index', 'label', 'tweet']

df_test.columns = columns
print(df_test.head())

df_non_ironic_test = df_test[df_test['label'] == '0']
df_polarity_test = df_test[df_test['label'] == '1']
df_sit = df_test[df_test['label'] == '2']
df_other_test = df_test[df_test['label'] == '3']
# Display the first few rows of the dataframe
start_index = 0

polarity_test = pd.concat([df_polarity_test, df_non_ironic_test.head(len(df_polarity_test))])
polarity_test['index'] = range(len(polarity_test))
polarity_test.to_csv('../datasets/SemEval2018/polarity_test.csv', index=False)
start_index += len(df_polarity_test)

other = pd.concat([df_other, df_other_test])
other['label'] = 1
other_test = pd.concat([other, df_non_ironic_test.iloc[start_index:start_index + len(other)]])
other_test['index'] = range(len(other_test))
other_test.to_csv('../datasets/SemEval2018/other_test.csv', index=False)

print(len(polarity_train))
print(len(polarity_val))
print(len(polarity_test))
print(len(other_test))