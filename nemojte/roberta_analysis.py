import pandas as pd 
import numpy as np

# path = '../preds/crossval4/roberta/semeval_mix_ds/semeval_mix_ds/roberta_trained_on_semeval_mix_ds_evaluated_on_semeval_mix_ds_fold_'
# data = pd.read_csv(f'{path}1.csv')
# for i in range(2, 6):
#     data = pd.concat([data, pd.read_csv(f'{path}{i}.csv')], ignore_index=True)

# data['hashtag'] = data['tweet'].apply(lambda x: 1 if '#' in x else 0)
# data['mention'] = data['tweet'].apply(lambda x: 1 if '@' in x else 0)
# data['hashtag_count'] = data['tweet'].apply(lambda x: x.count('#'))
# data['mention_count'] = data['tweet'].apply(lambda x: x.count('@'))
# data['link'] = data['tweet'].apply(lambda x: 1 if 'HTTPURL' in x else 0)
# data['link_count'] = data['tweet'].apply(lambda x: x.count('HTTPURL'))
# data['any'] = (data['hashtag'] == 1) | (data['mention']==1) | (data['link']==1).apply(lambda x: 1 if x else 0)
# data['sum'] = data['hashtag_count'] + data['mention_count'] + data['link_count']
# data['emoji'] = data['tweet'].apply(lambda x: 1 if x.count(':') >=2 else 0)
# data['emoji_count'] = data['tweet'].apply(lambda x: x.count(':') / 2)
# print(data)
# tn = data[(data['label'] == 0) & (data['prediction'] == 0)]
# fp = data[(data['label'] == 0) & (data['prediction'] == 1)]
# fn = data[(data['label'] == 1) & (data['prediction'] == 0)]
# tp = data[(data['label'] == 1) & (data['prediction'] == 1)]

# fp.to_csv('fp.csv', index=False)
# fn.to_csv('fn.csv', index=False)

# correlation_matrix = data[['hashtag', 'mention', 'hashtag_count', 'mention_count', 'link', 'link_count', 'any', 'sum', 'emoji', 'emoji_count', 'prediction', 'probability']].corr()
# print(correlation_matrix)


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

irony_parsed = pd.read_csv('../datasets/crossval/irony_parsed.csv')
sarcasm_parsed = pd.read_csv('../datasets/crossval/sarcasm_parsed.csv')
semeval_mix_parsed = pd.read_csv('../datasets/crossval/semeval_mix_parsed.csv')

for name, ds in zip(["irony", "sarcasm", "semeval_mix"], [irony_parsed, sarcasm_parsed, semeval_mix_parsed]):
    print(f"analysis for {name}")
    ds_pos = ds[ds['label'] == 1]
    ds_neg = ds[ds['label'] == 0]
    hash_pos = ds_pos['tweet'].apply(lambda x: 1 if "#" in x else 0).sum()
    hash_neg = ds_neg['tweet'].apply(lambda x: 1 if "#" in x else 0).sum()
    hash_avg_pos = ds_pos['tweet'].apply(lambda x: x.count('#')).mean()
    hash_avg_neg = ds_neg['tweet'].apply(lambda x: x.count('#')).mean()
    hash_std_pos = ds_pos['tweet'].apply(lambda x: x.count('#')).std()
    hash_std_neg = ds_neg['tweet'].apply(lambda x: x.count('#')).std()
    print(f"{hash_pos/len(ds_pos)*100:.2f}% of positive tweets contain hashtags, {hash_avg_pos:.2f} +- {hash_std_pos:.2f} hashtags per tweet")
    print(f"{hash_neg/len(ds_neg)*100:.2f}% of negative tweets contain hashtags, {hash_avg_neg:.2f} +- {hash_std_neg:.2f} hashtags per tweet")
    
    
    mention_pos = ds_pos['tweet'].apply(lambda x: 1 if "@" in x else 0).sum()
    mention_neg = ds_neg['tweet'].apply(lambda x: 1 if "@" in x else 0).sum()
    mention_avg_pos = ds_pos['tweet'].apply(lambda x: x.count('@')).mean()
    mention_avg_neg = ds_neg['tweet'].apply(lambda x: x.count('@')).mean()
    mention_std_pos = ds_pos['tweet'].apply(lambda x: x.count('@')).std()
    mention_std_neg = ds_neg['tweet'].apply(lambda x: x.count('@')).std()
    print(f"{mention_pos/len(ds_pos)*100:.2f}% of positive tweets contain mentions, {mention_avg_pos:.2f} +- {mention_std_pos:.2f} mentions per tweet")
    print(f"{mention_neg/len(ds_neg)*100:.2f}% of negative tweets contain mentions, {mention_avg_neg:.2f} +- {mention_std_neg:.2f} mentions per tweet")    
    
    link_pos = ds_pos['tweet'].apply(lambda x: 1 if "HTTPURL" in x else 0).sum()
    link_neg = ds_neg['tweet'].apply(lambda x: 1 if "HTTPURL" in x else 0).sum()
    link_avg_pos = ds_pos['tweet'].apply(lambda x: x.count('HTTPURL')).mean()
    link_avg_neg = ds_neg['tweet'].apply(lambda x: x.count('HTTPURL')).mean()
    link_std_pos = ds_pos['tweet'].apply(lambda x: x.count('HTTPURL')).std()
    link_std_neg = ds_neg['tweet'].apply(lambda x: x.count('HTTPURL')).std()
    print(f"{link_pos/len(ds_pos)*100:.2f}% of positive tweets contain links, {link_avg_pos:.2f} +- {link_std_pos:.2f} links per tweet")
    print(f"{link_neg/len(ds_neg)*100:.2f}% of negative tweets contain links, {link_avg_neg:.2f} +- {link_std_neg:.2f} links per tweet")
    
    emoji_pos = ds_pos['tweet'].apply(lambda x: 1 if x.count(':') >= 2 else 0).sum()
    emoji_neg = ds_neg['tweet'].apply(lambda x: 1 if x.count(':') >= 2 else 0).sum()
    emoji_avg_pos = ds_pos['tweet'].apply(lambda x: x.count(':') // 2).mean()
    emoji_avg_neg = ds_neg['tweet'].apply(lambda x: x.count(':') // 2).mean()
    emoji_std_pos = ds_pos['tweet'].apply(lambda x: x.count(':') // 2).std()
    emoji_std_neg = ds_neg['tweet'].apply(lambda x: x.count(':') // 2).std()
    print(f"{emoji_pos/len(ds_pos)*100:.2f}% of positive tweets contain emojis, {emoji_avg_pos:.2f} +- {emoji_std_pos:.2f} emojis per tweet")
    print(f"{emoji_neg/len(ds_neg)*100:.2f}% of negative tweets contain emojis, {emoji_avg_neg:.2f} +- {emoji_std_neg:.2f} emojis per tweet")
    print()
