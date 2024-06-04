import torch
from torch.nn.utils.rnn import pad_sequence
import pandas as pd
from torch.utils.data import Dataset
from transformerUtils import TransformerDataset
import re
from TweetNormalizer import normalizeTweet
import re

def clean_hashtags(sentence):
    hashtags_to_remove = ["#sarcasm", "#irony", "#not"]
    for hashtag in hashtags_to_remove:
        sentence = re.sub(fr"(?i){re.escape(hashtag)}", "", sentence)
    return sentence

def parse_dataset(fp, remove_hashtags=False):
    df = pd.read_csv(fp)
    corpus = df['tweet'].tolist()
    corpus = [normalizeTweet(tweet) for tweet in corpus]
    if remove_hashtags:
        corpus = [clean_hashtags(tweet) for tweet in corpus]
        corpus = [tweet for tweet in corpus if tweet.strip()]
    labels = df['label'].tolist()
    return corpus, labels

class TweetDataset(Dataset):
    def __init__(self, tweets, labels, glove, device):
        self.tweets = tweets
        self.labels = torch.tensor(labels).to(device)
        self.glove = glove
        self.device = device
        self.tokenized_tweets = [self.tokenize(tweet) for tweet in tweets]
        self.padded_sequences = pad_sequence([torch.stack(tweet) for tweet in self.tokenized_tweets], batch_first=True).to(device)
        
    def tokenize(self, tweet):
        return [self.glove[word] for word in tweet.split()]
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.padded_sequences[idx], self.labels[idx]

class GloveLoader():
    def load_dataset(self, device, train_fp, valid_fp, test_fp, glove, remove_hashtags=True):
        train_corpus, train_labels = parse_dataset(train_fp, remove_hashtags=remove_hashtags)
        self.train_dataset = TweetDataset(train_corpus, train_labels, glove, device)
        self.input_size = self.train_dataset.padded_sequences.size(-1)
        valid_corpus, valid_labels = parse_dataset(valid_fp, remove_hashtags=remove_hashtags)
        self.valid_dataset = TweetDataset(valid_corpus, valid_labels, glove, device)
        test_corpus, test_labels = parse_dataset(test_fp, remove_hashtags=remove_hashtags)
        self.test_dataset = TweetDataset(test_corpus, test_labels, glove, device)
        
class TransformerLoader():
    def load_dataset(self, train_fp, valid_fp ,test_fp, tokenizer, remove_hashtags=True):
        train_corpus, train_labels = parse_dataset(train_fp, remove_hashtags=remove_hashtags)
        self.train_dataset = TransformerDataset(train_corpus, train_labels, tokenizer)
        valid_corpus, valid_labels = parse_dataset(valid_fp, remove_hashtags=remove_hashtags)
        self.valid_dataset = TransformerDataset(valid_corpus, valid_labels, tokenizer)
        test_corpus, test_labels = parse_dataset(test_fp, remove_hashtags=remove_hashtags)
        self.test_dataset = TransformerDataset(test_corpus, test_labels, tokenizer)

