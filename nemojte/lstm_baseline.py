import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
from torchtext.vocab import GloVe
from sklearn import metrics
from Loader import GloveLoader
import train
import evaluate

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
        out = torch.cat((out[:, -1, :self.hidden_size], out[:, 0, self.hidden_size:]), dim=1)
        
        out = self.fc(out)
        return out

glove = GloVe(name='6B', dim=300)

loader_irony = GloveLoader('irony')
loader_irony.load_dataset(device, glove)
loader_sarcasm = GloveLoader('sarcasm')
loader_sarcasm.load_dataset(device, glove)
loader_mix = GloveLoader('mix')
loader_mix.load_dataset(device, glove)
loader_taskA = GloveLoader('taskA')
loader_taskA.load_dataset(device, glove)

hidden_size = 16
num_layers = 4
num_classes = 2
batch_size = 32
learning_rate = 0.0005
num_epochs = 40

# loader_sarcasm.load_dataset(device, train_sarcasm, valid_sarcasm, test_sarcasm, glove, balance=True)
# loader_irony.load_dataset(device, train_irony, valid_irony, test_irony, glove, balance=True)
# loader_mix.load_dataset(device, train_mix, valid_mix, test_mix, glove, balance=True)
# loader_taskA.load_dataset(device, train_taskA, valid_taskA, test_taskA, glove, balance=False)

for loader, dataset in zip([loader_taskA, loader_sarcasm, loader_irony, loader_mix], ["taskA", "sarcasm", "irony", "mix"]):
    
    sum_f1 = 0; sum_acc = 0; sum_prec = 0; sum_rec = 0
    
    for i in range(5):
        
        print(f"run {i + 1}/5 for {dataset}")

        model = BiLSTM(loader.input_size, hidden_size, num_layers, num_classes).to(device)

        criterion = nn.CrossEntropyLoss()

        model = train.train_baseline(device, model, learning_rate, batch_size, num_epochs, loader.train_dataset, loader.valid_dataset, criterion)

        f1, acc, prec, rec = evaluate.evaluate_baseline(device, loader.test_dataset, model)
        
        sum_f1 += f1; sum_acc += acc; sum_prec += prec; sum_rec += rec
    
    print(f"\n\nAverage f1 for {dataset} is {sum_f1 / 5}\nAverage acc for {dataset} is {sum_acc / 5}\nAverage prec for {dataset} is {sum_prec / 5}\nAverage rec for {dataset} is {sum_rec / 5}\n\n")

