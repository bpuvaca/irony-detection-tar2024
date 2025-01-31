import pandas as pd 
import numpy as np

path = '../preds/crossval4/roberta/semeval_mix_ds/semeval_mix_ds/roberta_trained_on_semeval_mix_ds_evaluated_on_semeval_mix_ds_fold_'
data = pd.read_csv(f'{path}1.csv')
for i in range(2, 6):
    data = pd.concat([data, pd.read_csv(f'{path}{i}.csv')], ignore_index=True)

data['hashtag'] = data['tweet'].apply(lambda x: 1 if '#' in x else 0)
data['mention'] = data['tweet'].apply(lambda x: 1 if '@' in x else 0)
data['hashtag_count'] = data['tweet'].apply(lambda x: x.count('#'))
data['mention_count'] = data['tweet'].apply(lambda x: x.count('@'))
data['link'] = data['tweet'].apply(lambda x: 1 if 'HTTPURL' in x else 0)
data['link_count'] = data['tweet'].apply(lambda x: x.count('HTTPURL'))
data['any'] = (data['hashtag'] == 1) | (data['mention']==1) | (data['link']==1).apply(lambda x: 1 if x else 0)
data['sum'] = data['hashtag_count'] + data['mention_count'] + data['link_count']
data['emoji'] = data['tweet'].apply(lambda x: 1 if x.count(':') >=2 else 0)
data['emoji_count'] = data['tweet'].apply(lambda x: x.count(':') / 2)
print(data)
tn = data[(data['label'] == 0) & (data['prediction'] == 0)]
fp = data[(data['label'] == 0) & (data['prediction'] == 1)]
fn = data[(data['label'] == 1) & (data['prediction'] == 0)]
tp = data[(data['label'] == 1) & (data['prediction'] == 1)]

fp.to_csv('fp.csv', index=False)
fn.to_csv('fn.csv', index=False)

correlation_matrix = data[['hashtag', 'mention', 'hashtag_count', 'mention_count', 'link', 'link_count', 'any', 'sum', 'emoji', 'emoji_count', 'prediction', 'probability']].corr()
print(correlation_matrix)


# path = '../preds/crossval4/roberta/semeval_mix_ds/sarcasm_ds/roberta_trained_on_semeval_mix_ds_evaluated_on_sarcasm_ds_fold_'
# data = pd.read_csv(f'{path}1.csv')
# for i in range(2, 6):
#     data = pd.concat([data, pd.read_csv(f'{path}{i}.csv')], ignore_index=True)

# data['hashtag'] = data['tweet'].apply(lambda x: 1 if '#' in x else 0)
# data['mention'] = data['tweet'].apply(lambda x: 1 if '@' in x else 0)
# data['hashtag_count'] = data['tweet'].apply(lambda x: x.count('#'))
# data['mention_count'] = data['tweet'].apply(lambda x: x.count('@'))
# data['link'] = data['tweet'].apply(lambda x: 1 if 'HTTPURL' in x else 0)
# data['link_count'] = data['tweet'].apply(lambda x: x.count('HTTPURL'))
# data['any'] = (data['hashtag'] == 1) | (data['mention']==1) | (data['link']==1)
# data['sum'] = data['hashtag_count'] + data['mention_count'] + data['link_count']
# print(data)
# tn = data[(data['label'] == 0) & (data['prediction'] == 0)]
# fp = data[(data['label'] == 0) & (data['prediction'] == 1)]
# fn = data[(data['label'] == 1) & (data['prediction'] == 0)]
# tp = data[(data['label'] == 1) & (data['prediction'] == 1)]

# positives = data[data['probability'] > 0.25]
# print(len(positives))
# positives.to_csv('positives.csv', index=False)

# fn.to_csv('fn.csv', index=False)


# correlation_matrix = data[['hashtag', 'mention', 'hashtag_count', 'mention_count', 'link', 'link_count', 'any', 'sum', 'prediction', 'probability']].corr()
# print(correlation_matrix)