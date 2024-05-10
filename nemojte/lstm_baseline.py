import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
from torchtext.vocab import GloVe
from sklearn import metrics

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(BiLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size*2, num_classes)  # Multiply by 2 for bidirectional
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(device)  # 2 for bidirectional
        c0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(device)
        
        out, _ = self.lstm(x, (h0, c0))
        
        # Concat the final forward and backward hidden states
        out = torch.cat((out[:, -1, :self.hidden_size], out[:, 0, self.hidden_size:]), dim=1)
        
        out = self.fc(out)
        return out
   

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

corpus, labels = parse_dataset("../datasets/train/SemEval2018-T3-train-taskA_emoji_ironyHashtags.txt")
test_corpus, test_labels = parse_dataset("..\datasets\goldtest_TaskA\SemEval2018-T3_gold_test_taskA_emoji.txt")
labels = torch.tensor(labels).to(device)

glove = GloVe(name='6B', dim=300)

tokenized_tweets = [[glove[word] for word in tweet.split()] for tweet in corpus]
tokenized_tweets = [torch.stack(tweet) for tweet in tokenized_tweets]

padded_sequences = pad_sequence(tokenized_tweets, batch_first=True).to(device)

input_size = padded_sequences.size(-1)

hidden_size = 16
num_layers = 2
num_classes = 2
batch_size = 64
learning_rate = 0.0005
num_epochs = 10

model = BiLSTM(input_size, hidden_size, num_layers, num_classes).to(device)

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(), lr=learning_rate)

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
    