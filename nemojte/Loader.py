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

file_path_dict = {
    #"train_sarcasm": "../datasets/sarcasm/sarcasm_train.csv",
    #"test_sarcasm": "../datasets/sarcasm/sarcasm_test.csv",
    #"valid_sarcasm": "../datasets/sarcasm/sarcasm_valid.csv",
    #"train_irony": "../datasets/irony/irony_train.csv",
    #"test_irony": "../datasets/irony/irony_test.csv",
    #"valid_irony": "../datasets/irony/irony_valid.csv",
    # "train_mix": "../datasets/mix/mix_train.csv",
    # "test_mix": "../datasets/mix/mix_test.csv",
    # "valid_mix": "../datasets/mix/mix_valid.csv",
    "train_task_a": "../datasets/taskA/taskA_train.csv",
    "test_task_a": "../datasets/taskA/taskA_test.csv",
    "valid_task_a": "../datasets/taskA/taskA_valid.csv",
    "train_isarcasm_sarc": "../datasets/iSarcasm/sarcasm_train.csv",
    "valid_isarcasm_sarc": "../datasets/iSarcasm/sarcasm_valid.csv",
    "test_isarcasm_sarc": "../datasets/iSarcasm/sarcasm_test.csv",
    "train_isarcasm_irony": "../datasets/iSarcasm/irony_train.csv",
    "valid_isarcasm_irony": "../datasets/iSarcasm/irony_valid.csv",
    "test_isarcasm_irony": "../datasets/iSarcasm/irony_test.csv",
    "train_semeval_polarity": "../datasets/SemEval2018/polarity_train.csv",
    "valid_semeval_polarity": "../datasets/SemEval2018/polarity_valid.csv",
    "test_semeval_polarity": "../datasets/SemEval2018/polarity_test.csv",
    "test_semeval_other": "../datasets/SemEval2018/other_test.csv",
    
    
    "valid_semeval_polarity": "../datasets/crossval/polarity.csv",
    "test_semeval_polarity": "../datasets/crossval/polarity.csv",
    "valid_isarcasm_sarc": "../datasets/crossval/sarcasm.csv",
    "test_isarcasm_sarc": "../datasets/crossval/sarcasm.csv",
    
    
    #crossval
    "train_polarity": "../datasets/crossval/polarity.csv",
    "valid_polarity": "../datasets/crossval/polarity.csv",
    "test_polarity": "../datasets/crossval/polarity.csv",
    
    "train_sarcasm": "../datasets/crossval/sarcasm.csv",
    "valid_sarcasm": "../datasets/crossval/sarcasm.csv",
    "test_sarcasm": "../datasets/crossval/sarcasm.csv",
    
    "train_other": "../datasets/crossval/other.csv",
    
    "train_irony": "../datasets/crossval/irony.csv",
    
    
    "train_sarcasm_mix": "../datasets/crossval/sarcasm_mix.csv",
    
    "train_irony_mix": "../datasets/crossval/irony_mix.csv",
    
    "train_mix": "../datasets/crossval/mix.csv",
    "valid_mix": "../datasets/crossval/mix.csv",
    "test_mix": "../datasets/crossval/mix.csv",
    
}

fold_size_dict = {
    "irony": 70,
    "polarity": 618,
    "sarcasm": 357,
    "other": 105,
    "sarcasm_mix": 357 + 105,
    "irony_mix": 70 + 618,
    "mix": 70 + 618 + 357 + 105
}

def reduce_dataset(corpus, labels, num_1s, num_0s, shuffle = True):
    reduced_corpus = []
    reduced_labels = []
    zip_corpus_labels = list(zip(corpus, labels))
    if shuffle:
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
        elif dataset_type == 'valid':
            corpus, labels = reduce_dataset(corpus, labels, BALANCED_TEST_VALID_1S, BALANCED_TEST_VALID_0S)
        elif dataset_type == 'test':
            corpus, labels = reduce_dataset(corpus, labels, BALANCED_TEST_VALID_1S, BALANCED_TEST_VALID_0S, shuffle=False)
    
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
    def __init__(self, task):
        task = task.lower()
        task = 'task_a' if task == 'taska' else task
        self.task = task
        self.balance = False if task == 'task_a' else True
        self.train_fp = file_path_dict[f"train_{task}"]
        self.valid_fp = file_path_dict[f"valid_{task}"]
        self.test_fp = file_path_dict[f"test_{task}"]

    def load_dataset(self, device, glove, remove_hashtags=True):
        train_corpus, train_labels = parse_dataset(self.train_fp, remove_hashtags=remove_hashtags, balance=self.balance, dataset_type='train')
        self.train_dataset = TweetDataset(train_corpus, train_labels, glove, device)
        self.input_size = self.train_dataset.padded_sequences.size(-1)
        
        valid_corpus, valid_labels = parse_dataset(self.valid_fp, remove_hashtags=remove_hashtags, balance=self.balance, dataset_type='valid')
        self.valid_dataset = TweetDataset(valid_corpus, valid_labels, glove, device)
        
        test_corpus, test_labels = parse_dataset(self.test_fp, remove_hashtags=remove_hashtags, balance=self.balance, dataset_type='test')
        self.test_dataset = TweetDataset(test_corpus, test_labels, glove, device)
    
    def load_test_dataset(self, device, glove, remove_hashtags=True):
        test_corpus, test_labels = parse_dataset(self.test_fp, remove_hashtags=remove_hashtags, balance=self.balance, dataset_type='test')
        self.test_dataset = TweetDataset(test_corpus, test_labels, glove, device)
        
class TransformerLoader():
    def __init__(self, task):
        task = task.lower()
        self.task = task
        task = 'task_a' if task == 'taska' else task
        # self.balance = False if task == 'task_a' else True
        self.balance = False
        try:
            self.train_fp = file_path_dict[f"train_{task}"]
        except:
            self.train_fp = None
        
        try:
            self.valid_fp = file_path_dict[f"valid_{task}"]
        except:
            self.valid_fp = None
            
        try:
            self.test_fp = file_path_dict[f"test_{task}"]
        except:
            self.test_fp = None
        #print("Balance: {}".format(self.balance))
        
    def load_dataset(self, tokenizer, remove_hashtags=True, balance_train=True):
        
        train_corpus, train_labels = parse_dataset(self.train_fp, remove_hashtags=remove_hashtags, balance=(self.balance and balance_train), dataset_type='train')
        self.train_dataset = TransformerDataset(train_corpus, train_labels, tokenizer)
        
        valid_corpus, valid_labels = parse_dataset(self.valid_fp, remove_hashtags=remove_hashtags, balance=self.balance, dataset_type='valid')
        self.valid_dataset = TransformerDataset(valid_corpus, valid_labels, tokenizer)
        
        test_corpus, test_labels = parse_dataset(self.test_fp, remove_hashtags=remove_hashtags, balance=self.balance, dataset_type='test')
        self.test_dataset = TransformerDataset(test_corpus, test_labels, tokenizer)

        #self.test_texts = test_corpus
        self.test_texts = list(zip(test_corpus, test_labels))

        
    def load_test_dataset(self, tokenizer, remove_hashtags=True):
        test_corpus, test_labels = parse_dataset(self.test_fp, remove_hashtags=remove_hashtags, balance=self.balance, dataset_type='test')
        self.test_dataset = TransformerDataset(test_corpus, test_labels, tokenizer)

        #self.test_texts = test_corpus
        self.test_texts = list(zip(test_corpus, test_labels))
        
    def load_crossval_dataset(self, tokenizer, remove_hashtags=True, k=5):
        corpus, labels = parse_dataset(self.train_fp, remove_hashtags=remove_hashtags, balance=False, dataset_type='train')
        self.train_datasets = [None for _ in range(k)]
        self.valid_datasets = [None for _ in range(k)]
        self.test_texts = [None for _ in range(k)]
        fold_size = fold_size_dict[self.task]
        for i in range(k):
            train_corpus = corpus[0:i*fold_size] + (corpus[(i+1)*fold_size:] if i < k-1 else [])
            train_labels = labels[0:i*fold_size] + (labels[(i+1)*fold_size:] if i < k-1 else [])
            self.train_datasets[i] = TransformerDataset(train_corpus, train_labels, tokenizer)
            valid_corpus = corpus[i*fold_size:(i+1)*fold_size] if i < k-1 else corpus[i*fold_size:]
            valid_labels = labels[i*fold_size:(i+1)*fold_size] if i < k-1 else labels[i*fold_size:]
            self.valid_datasets[i] = TransformerDataset(valid_corpus, valid_labels, tokenizer)
            self.test_texts[i] = list(zip(valid_corpus, valid_labels))
        
        
    
