import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
from torchtext.vocab import GloVe
from sklearn import metrics
from Loader import Loader
from Evaluator import Evaluator
from Trainer import Trainer

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
   

train_fp = "../datasets/train/SemEval2018-T3-train-taskA_emoji_ironyHashtags.txt"
test_fp = "..\datasets\goldtest_TaskA\SemEval2018-T3_gold_test_taskA_emoji.txt"

glove = GloVe(name='6B', dim=300)

loader = Loader()
loader.load(device, train_fp, test_fp, glove)


hidden_size = 16
num_layers = 2
num_classes = 2
batch_size = 64
learning_rate = 0.0005
num_epochs = 10

model = BiLSTM(loader.input_size, hidden_size, num_layers, num_classes).to(device)

criterion = nn.CrossEntropyLoss()

model = Trainer().train(model, learning_rate, batch_size, num_epochs, loader.train_dataset, criterion)

Evaluator().evaluate(device, glove, loader.test_corpus, loader.test_labels, model)

