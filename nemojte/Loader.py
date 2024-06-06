import torch
from torch.nn.utils.rnn import pad_sequence
import pandas as pd
from torch.utils.data import Dataset
import re
from TweetNormalizer import normalizeTweet
import random
from transformerUtils import TransformerDataset

BALANCED_TRAIN_1S = 810
BALANCED_TRAIN_0S = 1338
BALANCED_TEST_VALID_1S = 173
BALANCED_TEST_VALID_0S = 287

def reduce_dataset(corpus, labels, num_1s, num_0s):
    reduced_corpus = []
    reduced_labels = []
    zip_corpus_labels = list(zip(corpus, labels))
    random.shuffle(zip_corpus_labels)
    for tweet, label in zip_corpus_labels:
        if label == 1 and num_1s > 0:
            reduced_corpus.append(tweet)
            reduced_labels.append(label)
            num_1s -= 1
        elif label == 0 and num_0s > 0:
            reduced_corpus.append(tweet)
            reduced_labels.append(label)
            num_0s -= 1
    return reduced_corpus, reduced_labels

def clean_hashtags(sentence):
    hashtags_to_remove = ["#sarcasm", "#irony", "#not"]
    for hashtag in hashtags_to_remove:
        sentence = re.sub(fr"(?i){re.escape(hashtag)}", "", sentence)
    return sentence

def parse_dataset(fp, remove_hashtags=False, balance=False, dataset_type='train'):
    df = pd.read_csv(fp)
    corpus = df['tweet'].tolist()
    corpus = [normalizeTweet(tweet) for tweet in corpus]
    if remove_hashtags:
        corpus = [clean_hashtags(tweet) for tweet in corpus]
        corpus = [tweet for tweet in corpus if tweet.strip()]
    labels = df['label'].tolist()
    
    if balance:
        if dataset_type == 'train':
            corpus, labels = reduce_dataset(corpus, labels, BALANCED_TRAIN_1S, BALANCED_TRAIN_0S)
        else:
            corpus, labels = reduce_dataset(corpus, labels, BALANCED_TEST_VALID_1S, BALANCED_TEST_VALID_0S)
    
    print(f"Parsed dataset type {dataset_type} with {len(corpus)} tweets, {labels.count(1)} 1s and {labels.count(0)} 0s")
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
    def load_dataset(self, device, train_fp, valid_fp, test_fp, glove, remove_hashtags=True, balance=False):
        train_corpus, train_labels = parse_dataset(train_fp, remove_hashtags=remove_hashtags, balance=balance, dataset_type='train')
        self.train_dataset = TweetDataset(train_corpus, train_labels, glove, device)
        self.input_size = self.train_dataset.padded_sequences.size(-1)
        
        valid_corpus, valid_labels = parse_dataset(valid_fp, remove_hashtags=remove_hashtags, balance=balance, dataset_type='valid')
        self.valid_dataset = TweetDataset(valid_corpus, valid_labels, glove, device)
        
        test_corpus, test_labels = parse_dataset(test_fp, remove_hashtags=remove_hashtags, balance=balance, dataset_type='test')
        self.test_dataset = TweetDataset(test_corpus, test_labels, glove, device)
    
    def load_test_dataset(self, device, test_fp, glove, remove_hashtags=True, balance=False):
        test_corpus, test_labels = parse_dataset(test_fp, remove_hashtags=remove_hashtags, balance=balance, dataset_type='test')
        self.test_dataset = TweetDataset(test_corpus, test_labels, glove, device)
        
class TransformerLoader():
    def load_dataset(self, train_fp, valid_fp, test_fp, tokenizer, remove_hashtags=True, balance=False):
        train_corpus, train_labels = parse_dataset(train_fp, remove_hashtags=remove_hashtags, balance=balance, dataset_type='train')
        self.train_dataset = TransformerDataset(train_corpus, train_labels, tokenizer)
        
        valid_corpus, valid_labels = parse_dataset(valid_fp, remove_hashtags=remove_hashtags, balance=balance, dataset_type='valid')
        self.valid_dataset = TransformerDataset(valid_corpus, valid_labels, tokenizer)
        
        test_corpus, test_labels = parse_dataset(test_fp, remove_hashtags=remove_hashtags, balance=balance, dataset_type='test')
        self.test_dataset = TransformerDataset(test_corpus, test_labels, tokenizer)
        
    def load_test_dataset(self, test_fp, tokenizer, remove_hashtags=True, balance=False):
        test_corpus, test_labels = parse_dataset(test_fp, remove_hashtags=remove_hashtags, balance=balance, dataset_type='test')
        self.test_dataset = TransformerDataset(test_corpus, test_labels, tokenizer)
