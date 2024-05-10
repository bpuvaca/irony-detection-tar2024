import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
from torchtext.vocab import GloVe
from sklearn import metrics

class Train:
    def __init__(self) -> None:
        pass
            
    def train(self, model, learning_rate, padded_sequences, labels, batch_size, num_epochs, device, test_corpus, criterion, optimizer):



        train_dataset = torch.utils.data.TensorDataset(padded_sequences, labels)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        # Training loop
        for epoch in range(num_epochs):
            total_loss = 0
            for batch_inputs, batch_labels in train_loader:
                
                # Forward pass
                outputs = model(batch_inputs)
                loss = criterion(outputs, batch_labels)
                
                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss / len(train_loader)}")


        test_labels = torch.tensor(test_labels).to(device)
        with torch.no_grad():
            tokenized_test_tweets = [[glove[word] for word in tweet.split()] for tweet in test_corpus]
            tokenized_test_tweets = [torch.stack(tweet) for tweet in tokenized_test_tweets]
            padded_test_sequences = pad_sequence(tokenized_test_tweets, batch_first=True).to(device)
            test_outputs = model(padded_test_sequences)
            _, predicted = torch.max(test_outputs, 1)
            f1 = metrics.f1_score(test_labels.cpu(), predicted.cpu(), average='macro')
            print("F1: ", f1)
    