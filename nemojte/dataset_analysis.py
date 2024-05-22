import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
from torchtext.vocab import GloVe
from sklearn import metrics
from Loader import Loader, parse_dataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

corpus, labels = parse_dataset("../datasets/train/SemEval2018-T3-train-taskB_emoji_ironyHashtags.txt")
label_distribution = torch.bincount(torch.tensor(labels))
index_to_label = {0: "not irony", 1: "irony by clash", 2: "situational irony", 3: "other irony (sarcasm,...)"}
for i, count in enumerate(label_distribution):
   print(f"{index_to_label[i]}: {count}")

sarcastic_tweets = [corpus[i] for i, label in enumerate(labels) if label == 3]
tweets_including_hashtag_sarcasm = [tweet for tweet in sarcastic_tweets if "#sarcasm" in tweet]
print(f"Number of allegedly sarcastic tweets: {len(sarcastic_tweets)}")
print(f"Number of tweets including #sarcasm: {len(tweets_including_hashtag_sarcasm)}")
