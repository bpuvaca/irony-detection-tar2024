import torch
from torch.nn.utils.rnn import pad_sequence
from torchtext.vocab import GloVe

def parse_dataset(fp):
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
    
class Loader():
    def load(self, device, train_fp, test_fp, glove):

        corpus, labels = parse_dataset(train_fp)
        labels = torch.tensor(labels).to(device)
        tokenized_tweets = [[glove[word] for word in tweet.split()] for tweet in corpus]
        tokenized_tweets = [torch.stack(tweet) for tweet in tokenized_tweets]
        padded_sequences = pad_sequence(tokenized_tweets, batch_first=True).to(device)
        self.train_dataset = torch.utils.data.TensorDataset(padded_sequences, labels)
        self.input_size = padded_sequences.size(-1)

        # train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        self.test_corpus, self.test_labels = parse_dataset(test_fp)
        

