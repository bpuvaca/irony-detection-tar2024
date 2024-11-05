import pandas as pd
from sklearn.model_selection import train_test_split

## TRAIN
df_iSarcasm = pd.read_csv("datasets/iSarcasm/train.En.csv", sep=",")
df_iSarcasm.dropna(inplace=True, subset=['tweet'])


iSarcasm_column_names = ["i","tweet","sarcastic","rephrase","sarcasm","irony","satire","understatement","overstatement","rhetorical_question"]
df_iSarcasm.columns = iSarcasm_column_names

columns_to_drop_iSarcasm = ["i","rephrase","satire","understatement","overstatement","rhetorical_question"]
df_iSarcasm.drop(columns=columns_to_drop_iSarcasm, inplace=True)

df_sarcasm = df_iSarcasm[(df_iSarcasm['sarcasm'] == 1)].drop(columns=["sarcastic","irony"]).rename(columns={'sarcasm': 'label'})
df_irony = df_iSarcasm[(df_iSarcasm['irony'] == 1)].drop(columns=["sarcastic","sarcasm"]).rename(columns={'irony': 'label'})
df_not_sarcastic = df_iSarcasm[(df_iSarcasm['sarcastic'] == 0)].drop(columns=["sarcasm", "irony"]).rename(columns={'sarcastic': 'label'})

df_sarcasm_train, df_sarcasm_val = train_test_split(df_sarcasm, test_size=0.2, random_state=42)

# df_sarcasm_train.reset_index(inplace=True)

# df_sarcasm_val.reset_index(inplace=True)

# df_not_sarcastic.reset_index(inplace=True)

# df_irony.reset_index(inplace=True)
# df_irony.rename(columns={'index': 'index', 'text': 'tweet'}, inplace=True)

# print("Number of rows in df_sarcasm:", len(df_sarcasm))
# print("Number of rows in df_irony:", len(df_irony))
# print("Number of rows in df_not_sarcastic:", len(df_not_sarcastic))

##TEST
df_iSarcasm_test = pd.read_csv("datasets/iSarcasm/task_B_En_test.csv", sep=",")
# df_iSarcasm_test.dropna(axis=1, inplace=True)
df_iSarcasm_test.dropna(inplace=True, subset=['text'])
iSarcasm_test_column_names = "tweet,sarcasm,irony,satire,understatement,overstatement,rhetorical_question".split(",")
df_iSarcasm_test.columns = iSarcasm_test_column_names

columns_to_drop_iSarcasm_test = "satire,understatement,overstatement,rhetorical_question".split(",")
df_iSarcasm_test.drop(columns=columns_to_drop_iSarcasm_test, inplace=True)

df_sarcasm_test = df_iSarcasm_test[(df_iSarcasm_test['sarcasm'] == 1)].drop(columns=["irony"]).rename(columns={'sarcasm': 'label'})
df_irony_test = df_iSarcasm_test[(df_iSarcasm_test['irony'] == 1)].drop(columns=["sarcasm"]).rename(columns={'irony': 'label'})
df_not_sarcastic_test = df_iSarcasm_test[(df_iSarcasm_test['sarcasm'] != 1) & (df_iSarcasm_test['irony'] != 1)].drop(columns=["sarcasm"]).rename(columns={'irony': 'label'})

sarcasm_train = pd.concat([df_sarcasm_train, df_not_sarcastic.head(len(df_sarcasm_train))])
sarcasm_train['index'] = range(len(sarcasm_train))
sarcasm_train.to_csv("datasets/iSarcasm/sarcasm_train.csv", index=False)

sarcasm_val = pd.concat([df_sarcasm_val, df_not_sarcastic.iloc[len(df_sarcasm_train):len(df_sarcasm_train)+len(df_sarcasm_val)]])
sarcasm_val['index'] = range(len(sarcasm_val))
sarcasm_val.to_csv("datasets/iSarcasm/sarcasm_valid.csv", index=False)

sarcasm_test = pd.concat([df_sarcasm_test, df_not_sarcastic.iloc[len(df_sarcasm_train)+len(df_sarcasm_val):len(df_sarcasm_train)+len(df_sarcasm_val)+len(df_sarcasm_test)]])
sarcasm_test['index'] = range(len(sarcasm_test))
sarcasm_test.to_csv("datasets/iSarcasm/sarcasm_test.csv", index=False)

irony_start_index = len(df_sarcasm_train)+len(df_sarcasm_val)+len(df_sarcasm_test)
df_irony = pd.concat([df_irony, df_irony_test])
irony_test = pd.concat([df_irony, df_not_sarcastic.iloc[irony_start_index:irony_start_index+len(df_irony)]])
pd.DataFrame().to_csv("datasets/iSarcasm/irony_train.csv", index=False)
pd.DataFrame().to_csv("datasets/iSarcasm/irony_valid.csv", index=False)
irony_test['index'] = range(len(irony_test))
irony_test.to_csv("datasets/iSarcasm/irony_test.csv", index=False)
# Create a new and empty dataframe
# print(len(sarcasm_train))
# print(len(sarcasm_train[sarcasm_train['label'] == 1]))
# print(len(sarcasm_train[sarcasm_train['label'] == 0]))

# print(df_sarcasm.head(5))
# print(df_not_sarcastic.head(5))
# print(len(sarcasm_val))
# print(len(sarcasm_val[sarcasm_val['label'] == 1]))
# print(len(sarcasm_val[sarcasm_val['label'] == 0]))
# print(len(sarcasm_test))
# print(len(sarcasm_test[sarcasm_test['label'] == 1]))
# print(len(sarcasm_test[sarcasm_test['label'] == 0]))
# print(len(df_sarcasm_test))
# print(len(df_not_sarcastic_test))
# print(sarcasm_train.head(5))

print(len(irony_test))
print(len(irony_test[irony_test['label'] == 1]))
print(len(irony_test[irony_test['label'] == 0]))
# df_other_irony['label'] = 1
# df_other_irony.reset_index(inplace=True)
# df_other_irony.rename(columns={'index': 'index', 'tweet_text': 'tweet'}, inplace=True)

# df_not_sarcastic['label'] = 0
# df_not_sarcastic.reset_index(inplace=True)
# df_not_sarcastic.rename(columns={'index': 'index', 'text': 'tweet'}, inplace=True)

# #neutral vs sarcasm => 2601 vs 915

# df_neutral_vs_sarcasm = pd.concat([df_sarcastic_and_sarcasm[['index', 'tweet', 'label']], 
#                          df_other_irony[['index', 'tweet', 'label']], 
#                          df_not_sarcastic[['index', 'tweet', 'label']]], ignore_index=True)


# df_neutral_vs_sarcasm.to_csv("sarcasm_test.csv", index=False)