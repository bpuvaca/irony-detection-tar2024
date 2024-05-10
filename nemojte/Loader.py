
import torch
from torch.nn.utils.rnn import pad_sequence
from torchtext.vocab import GloVe

class Loader():
    
    def parse_dataset(fp):
        '''
        Loads the dataset .txt file with label-tweet on each line and parses the dataset.
        :param fp: filepath of dataset
        :return:
            corpus: list of tweet strings of each tweet.
            y: list of labels
        '''
        y = []
        corpus = []
        with open(fp, 'rt', encoding='utf-8') as data_in:
            for line in data_in:
                if not line.lower().startswith("tweet index"): # discard first line if it contains metadata
                    line = line.rstrip() # remove trailing whitespace
                    label = int(line.split("\t")[1])
                    tweet = line.split("\t")[2]
                    y.append(label)
                    corpus.append(tweet)

        return corpus, y
    
    def load(self, device, train_fp, test_fp):

        corpus, labels = self.parse_dataset(train_fp)
        test_corpus, test_labels = self.parse_dataset(test_fp)
        labels = torch.tensor(labels).to(device)

        glove = GloVe(name='6B', dim=300)

        tokenized_tweets = [[glove[word] for word in tweet.split()] for tweet in corpus]
        tokenized_tweets = [torch.stack(tweet) for tweet in tokenized_tweets]
        padded_sequences = pad_sequence(tokenized_tweets, batch_first=True).to(device)
        
        tokenized_test_tweets = [[glove[word] for word in tweet.split()] for tweet in test_corpus]
        tokenized_test_tweets = [torch.stack(tweet) for tweet in tokenized_test_tweets]
        padded_test_sequences = pad_sequence(tokenized_test_tweets, batch_first=True).to(device)
        
        return padded_sequences, padded_test_sequences
