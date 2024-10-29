import pandas as pd

## TRAIN
df_iSarcasm = pd.read_csv("datasets/iSarcasm/train.En.csv", sep=",")

iSarcasm_column_names = ["i","tweet","sarcastic","rephrase","sarcasm","irony","satire","understatement","overstatement","rhetorical_question"]
df_iSarcasm.columns = iSarcasm_column_names

columns_to_drop_iSarcasm = ["i", "sarcastic","rephrase","satire","understatement","overstatement","rhetorical_question"]
df_iSarcasm.drop(columns=columns_to_drop_iSarcasm, inplace=True)

df_sarcasm = df_iSarcasm[(df_iSarcasm['sarcasm'] == 1)]
df_irony = df_iSarcasm[(df_iSarcasm['irony'] == 1)]
df_not_sarcastic = df_iSarcasm[(df_iSarcasm['sarcasm'] != 0) & (df_iSarcasm['irony'] != 0)]

df_sarcasm.reset_index(inplace=True)
df_sarcasm.rename(columns={'index': 'index', 'text': 'tweet'}, inplace=True)

df_irony.reset_index(inplace=True)
df_irony.rename(columns={'index': 'index', 'text': 'tweet'}, inplace=True)

df_irony.reset_index(inplace=True)
df_irony.rename(columns={'index': 'index', 'text': 'tweet'}, inplace=True)

print("Number of rows in df_sarcasm:", len(df_sarcasm))
print("Number of rows in df_irony:", len(df_irony))
print("Number of rows in df_not_sarcastic:", len(df_not_sarcastic))

##TEST
df_iSarcasm_test = pd.read_csv("datasets/iSarcasm/task_B_En_test.csv", sep=",")
print(len(df_iSarcasm_test))

iSarcasm_test_column_names = "text,sarcasm,irony,satire,understatement,overstatement,rhetorical_question".split(",")
df_iSarcasm_test.columns = iSarcasm_test_column_names

columns_to_drop_iSarcasm_test = "satire,understatement,overstatement,rhetorical_question".split(",")
df_iSarcasm_test.drop(columns=columns_to_drop_iSarcasm_test, inplace=True)

df_sarcasm_test = df_iSarcasm_test[(df_iSarcasm_test['sarcasm'] == 1)]
df_irony_test = df_iSarcasm_test[(df_iSarcasm_test['irony'] == 1)]
df_not_sarcastic_test = df_iSarcasm_test[(df_iSarcasm_test['sarcasm'] != 1) & (df_iSarcasm_test['irony'] != 1)]

df_sarcasm_test.reset_index(inplace=True)
df_sarcasm_test.rename(columns={'index': 'index', 'text': 'tweet'}, inplace=True)

df_irony_test.reset_index(inplace=True)
df_irony_test.rename(columns={'index': 'index', 'text': 'tweet'}, inplace=True)

df_irony_test.reset_index(inplace=True)
df_irony_test.rename(columns={'index': 'index', 'text': 'tweet'}, inplace=True)

print("Number of rows in df_sarcasm_test:", len(df_sarcasm_test))
print("Number of rows in df_irony_test:", len(df_irony_test))
print("Number of rows in df_not_sarcastic_test:", len(df_not_sarcastic_test))

sarcasm_train = pd.concat([df_sarcasm, df_not_sarcastic.head(len(df_sarcasm))])
sarcasm_train.to_csv("datasets/iSarcasm/sarcasm_train.csv", index=False)

sarcasm_test = pd.concat([df_sarcasm_test, df_not_sarcastic_test.iloc[len(df_sarcasm):len(df_sarcasm)+len(df_sarcasm_test)]])
sarcasm_test.to_csv("datasets/iSarcasm/sarcasm_test.csv", index=False)

print(len(sarcasm_train))
# print(df_sarcasm.head(5))
# print(df_not_sarcastic.head(5))

print(len(sarcasm_test))

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