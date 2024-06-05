import pandas as pd


df_iSarcasm = pd.read_csv("task_B_En_test.csv", sep=",")
df_semeval2018 = pd.read_csv("irony-detection-tar2024/datasets/goldtest_TaskB/SemEval2018-T3_gold_test_taskB_emoji.txt", sep="\t")

iSarcasm_column_names = ["text","sarcasm","irony","satire","understatement","overstatement","rhetorical_question"]
df_iSarcasm.columns = iSarcasm_column_names
semeval2018_column_names = ["tweet index","label","tweet_text"]
df_semeval2018.columns = semeval2018_column_names

"""
df_other_irony = df_semeval2018[df_semeval2018['label'] == 3]
df_sarcastic_and_sarcasm = df_iSarcasm[(df_iSarcasm['sarcasm'] == 1)]
df_not_sarcastic = df_iSarcasm[(df_iSarcasm['sarcasm'] == 0) & (df_iSarcasm['irony'] == 0) & (df_iSarcasm['satire'] == 0) & (df_iSarcasm['understatement'] == 0) & (df_iSarcasm['overstatement'] == 0) &(df_iSarcasm['rhetorical_question'] == 0)]


columns_to_drop_iSarcasm = ["sarcasm","irony","satire","understatement","overstatement","rhetorical_question"]
df_sarcastic_and_sarcasm.drop(columns=columns_to_drop_iSarcasm, inplace=True)
df_not_sarcastic.drop(columns=columns_to_drop_iSarcasm, inplace=True)

columns_to_drop_semeval = ["tweet index", "label"]
df_other_irony.drop(columns=columns_to_drop_semeval, inplace=True)


df_sarcastic_and_sarcasm['label'] = 1
df_sarcastic_and_sarcasm.reset_index(inplace=True)
df_sarcastic_and_sarcasm.rename(columns={'index': 'index', 'text': 'tweet'}, inplace=True)

df_other_irony['label'] = 1
df_other_irony.reset_index(inplace=True)
df_other_irony.rename(columns={'index': 'index', 'tweet_text': 'tweet'}, inplace=True)

df_not_sarcastic['label'] = 0
df_not_sarcastic.reset_index(inplace=True)
df_not_sarcastic.rename(columns={'index': 'index', 'text': 'tweet'}, inplace=True)

#neutral vs sarcasm => 2601 vs 915

df_neutral_vs_sarcasm = pd.concat([df_sarcastic_and_sarcasm[['index', 'tweet', 'label']], 
                         df_other_irony[['index', 'tweet', 'label']], 
                         df_not_sarcastic[['index', 'tweet', 'label']]], ignore_index=True)


df_neutral_vs_sarcasm.to_csv("sarcasm_test.csv", index=False)

"""

df_semeval2018A = pd.read_csv("irony-detection-tar2024/datasets/goldtest_TaskA/SemEval2018-T3_gold_test_taskA_emoji.txt", sep="\t")
semeval2018A_column_names = ["tweet index","label","tweet text"]
df_semeval2018A.columns = semeval2018A_column_names

df_irony = df_iSarcasm[(df_iSarcasm['irony']==1)]
df_irony_s = df_semeval2018A[(df_semeval2018A['label']==1)]
df_not_irony_s = df_semeval2018A[(df_semeval2018A['label']==0)]

df_irony.drop(columns=["sarcasm","irony","satire","understatement","overstatement","rhetorical_question"])
df_irony_s.drop(columns=["tweet index", "tweet text"])
df_not_irony_s.drop(columns=["tweet index", "tweet text"])

df_irony['label'] = 1
df_irony.reset_index(inplace=True)
df_irony.rename(columns={'index': 'index', 'text': 'tweet'}, inplace=True)

df_irony_s['label'] = 1
df_irony_s.reset_index(inplace=True)
df_irony_s.rename(columns={'index': 'index', 'tweet text': 'tweet'}, inplace=True)

df_not_irony_s['label'] = 0
df_not_irony_s.reset_index(inplace=True)
df_not_irony_s.rename(columns={'index': 'index', 'tweet text': 'tweet'}, inplace=True)

df_neutral_vs_irony = pd.concat([df_irony[['index', 'tweet', 'label']], 
                         df_irony_s[['index', 'tweet', 'label']], 
                         df_not_irony_s[['index', 'tweet', 'label']]], ignore_index=True)


df_neutral_vs_irony.to_csv("test_irony.csv", index=False)













