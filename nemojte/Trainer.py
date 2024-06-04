import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
from torchtext.vocab import GloVe
from sklearn import metrics

class Trainer:
    def __init__(self) -> None:
        pass
            
    def train(self, model, learning_rate, batch_size, num_epochs, train_dataset, criterion):

        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

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
        
        return model