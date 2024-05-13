import torch
from sklearn import metrics
from Loader import pad_sequence

class Evaluator:
    def evaluate(self, device, glove, test_corpus, test_labels, model):
        test_labels = torch.tensor(test_labels).to(device)
        with torch.no_grad():
            tokenized_test_tweets = [[glove[word] for word in tweet.split()] for tweet in test_corpus]
            tokenized_test_tweets = [torch.stack(tweet) for tweet in tokenized_test_tweets]
            padded_test_sequences = pad_sequence(tokenized_test_tweets, batch_first=True).to(device)
            test_outputs = model(padded_test_sequences)
            _, predicted = torch.max(test_outputs, 1)
            f1 = metrics.f1_score(test_labels.cpu(), predicted.cpu(), average='macro')
            print("F1: ", f1)