import torch
from torch.nn.utils.rnn import pad_sequence
from torchtext.vocab import GloVe
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformerUtils import TransformerDataset

def parse_dataset(fp):
    df = pd.read_csv(fp)
    corpus = df['tweet'].tolist()
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
    def load_dataset(self, device, train_fp, test_fp, glove):
        train_corpus, train_labels = parse_dataset(train_fp)
        self.train_dataset = TweetDataset(train_corpus, train_labels, glove, device)
        self.input_size = self.train_dataset.padded_sequences.size(-1)
        test_corpus, test_labels = parse_dataset(test_fp)
        self.test_dataset = TweetDataset(test_corpus, test_labels, glove, device)
        
class TransformerLoader():
    def load_dataset(self, train_fp, test_fp, tokenizer):
        train_corpus, train_labels = parse_dataset(train_fp)
        self.train_dataset = TransformerDataset(train_corpus, train_labels, tokenizer)
        test_corpus, test_labels = parse_dataset(test_fp)
        self.test_dataset = TransformerDataset(test_corpus, test_labels, tokenizer)

