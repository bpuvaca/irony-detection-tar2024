import pandas as pd

def count_common_tweets(file_path1, file_path2):
    df1 = pd.read_csv(file_path1)
    df2 = pd.read_csv(file_path2)

    tweets1 = df1['tweet'].tolist()
    tweets2 = df2['tweet'].tolist()

    set1 = set(tweets1)
    set2 = set(tweets2)

    common_tweets = set1.intersection(set2)

    return len(common_tweets)

file_path1 = '../datasets/taskA/taskA_test.csv'
file_path2 = '../datasets/sarcasm/sarcasm_train.csv'

common_tweets_count = count_common_tweets(file_path1, file_path2)
print(f"Number of common tweets: {common_tweets_count}")
