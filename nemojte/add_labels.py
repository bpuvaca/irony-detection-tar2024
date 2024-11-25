import json
import pandas as pd
from nltk.tokenize import TweetTokenizer
from emoji import demojize
from difflib import get_close_matches, SequenceMatcher

tokenizer = TweetTokenizer()

def normalizeToken(token):
    lowercased_token = token.lower()
    if token.startswith("@"):
        return "@USER"
    elif lowercased_token.startswith("http") or lowercased_token.startswith("www"):
        return "HTTPURL"
    elif len(token) == 1:
        return demojize(token)
    else:
        if token == "’":
            return "'"
        elif token == "…":
            return "..."
        else:
            return token

def normalizeTweet(tweet):
    tokens = tokenizer.tokenize(tweet.replace("’", "'").replace("…", "..."))
    normTweet = " ".join([normalizeToken(token) for token in tokens])
    normTweet = (
        normTweet.replace("cannot ", "can not ")
        .replace("n't ", " n't ")
        .replace("n 't ", " n't ")
        .replace("ca n't", "can't")
        .replace("ai n't", "ain't")
    )
    normTweet = (
        normTweet.replace("'m ", " 'm ")
        .replace("'re ", " 're ")
        .replace("'s ", " 's ")
        .replace("'ll ", " 'll ")
        .replace("'d ", " 'd ")
        .replace("'ve ", " 've ")
    )
    normTweet = (
        normTweet.replace(" p . m .", " p.m.")
        .replace(" p . m ", " p.m ")
        .replace(" a . m .", " a.m.")
        .replace(" a . m ", " a.m ")
    )
    return " ".join(normTweet.split())

json_file_path = 'training_dynamics/bertweet_trainedon_semeval_polarity_10epoch.json'
with open(json_file_path, 'r') as f:
    json_data = json.load(f)

csv_file_path = '../datasets/crossval/polarity.csv'
sarcasm_data = pd.read_csv(csv_file_path)

sarcasm_data['normalized_tweet'] = sarcasm_data['tweet'].apply(normalizeTweet)

tweet_to_label = dict(zip(sarcasm_data['normalized_tweet'], sarcasm_data['label']))

unmatched_tweets = []

for entry in json_data.values():
    json_tweet = entry['tweet_text']
    entry['label'] = tweet_to_label.get(json_tweet, None)
    if entry['label'] is None:
        unmatched_tweets.append(json_tweet)
    elif entry['label'] is not None:
        del tweet_to_label[json_tweet]


with open("tweet_matches.txt", "w", encoding='utf-8') as f:
    for entry in json_data.values():
        close_matches = []
        if entry['label'] is None:
            json_tweet = entry['tweet_text']
            close_matches = get_close_matches(json_tweet, tweet_to_label.keys(), n=1, cutoff=0.6)
            if len(close_matches)==0:
                print(json_tweet)
            if close_matches:
                closest_match = close_matches[0]
                similarity_score = SequenceMatcher(None, json_tweet, closest_match).ratio()
                entry["label"] = tweet_to_label[closest_match]
                f.write(f"JSON Tweet: {json_tweet}\nClosest Match: {closest_match}\nScore: {similarity_score}\n\n")


updated_json_path = 'training_dynamics/bertweet_trainedon_semeval_polarity_10epoch.json'
with open(updated_json_path, 'w', encoding='utf-8') as f:
    json.dump(json_data, f)

print(f"Updated JSON saved to: {updated_json_path}")
