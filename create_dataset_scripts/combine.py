
import pandas as pd
from sklearn.model_selection import train_test_split

df1 = pd.read_csv('train_taskA.txt', sep='\t')
df2 = pd.read_csv('test_taskA.txt', sep='\t')

combined_df = pd.concat([df1, df2])

combined_df.columns = ['index', 'label', 'tweet']
combined_df = combined_df[['index', 'tweet', 'label']]

train_df, temp_df = train_test_split(combined_df, train_size=0.7, stratify=combined_df['label'], random_state=42)
val_df, test_df = train_test_split(temp_df, train_size=0.5, stratify=temp_df['label'], random_state=42)

train_df.to_csv('irony_train_taskA.csv', index=False)
val_df.to_csv('irony_validation_taskA.csv', index=False)
test_df.to_csv('irony_test_taskA.csv', index=False)


"""
import pandas as pd

train_df = pd.read_csv('iSarcasm_train.csv')
test_df = pd.read_csv('iSarcasm_test.csv')

combined_df = pd.concat([train_df, test_df])

filtered_df = combined_df[combined_df['irony'] == 1]

filtered_df.to_csv('two.csv', index=False)

filtered_df.shape, filtered_df['irony'].value_counts()



import pandas as pd

one_df = pd.read_csv('one.csv', sep=',', on_bad_lines='skip')
two_df = pd.read_csv('two.csv', sep=',')

one_df.columns = ['raw']  # Since we can't parse the columns properly, treat it as a single column
one_df = one_df['raw'].str.split('\t', expand=True)
one_df.columns = ['index', 'label', 'tweet']

two_df = two_df.rename(columns={'Unnamed: 0': 'index', 'tweet': 'tweet'})
two_df['label'] = 1
two_df = two_df[['index', 'tweet', 'label']]

one_df['index'] = pd.to_numeric(one_df['index'], errors='coerce')

combined_df = pd.concat([one_df[['index', 'tweet', 'label']], two_df[['index', 'tweet', 'label']]])

combined_df = combined_df.reset_index(drop=True)
combined_df['index'] = combined_df.index

combined_df.to_csv('irony.csv', index=False)

combined_df.shape, combined_df.head()

"""