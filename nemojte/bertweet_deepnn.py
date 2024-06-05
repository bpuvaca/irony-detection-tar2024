import torch
from transformers import BertTokenizer, BertModel
import torch.nn as nn
from torch.utils.data import DataLoader

from Loader import TransformerLoader
from transformerUtils import train_and_evaluate_deep_bertweet

from Trainer import Trainer
from torch.nn import BCEWithLogitsLoss

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class BertweetDeepModel(nn.Module):
    def __init__(self, bertweet_model, hidden_size=64, dropout=0.1):
        super(BertweetDeepModel, self).__init__()
        self.bertweet_model = bertweet_model
        self.hidden_size = hidden_size
        self.bilstm = nn.LSTM(bidirectional=True, input_size=768, hidden_size=self.hidden_size, num_layers=2, batch_first=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(768 + 2*self.hidden_size, 1)

    def forward(self, input_ids):
        outputs = self.bertweet_model(input_ids)
        embeddings = outputs.last_hidden_state
        lstm_output, _ = self.bilstm(embeddings)
        concatenated_output = torch.cat((lstm_output, embeddings), dim=2)
        pooled_output = concatenated_output.max(dim=1).values
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output).squeeze(-1)
        return logits
    
tokenizer = BertTokenizer.from_pretrained('vinai/bertweet-base')
bertweet_model = BertModel.from_pretrained('vinai/bertweet-base').to(device)
bertweet_model.to(device)

train_sarcasm = "../datasets/sarcasm/sarcasm_train.csv"
test_sarcasm = "../datasets/sarcasm/sarcasm_test.csv"
valid_sarcasm = "../datasets/sarcasm/sarcasm_valid.csv"
train_irony = "../datasets/irony/irony_train.csv"
test_irony = "../datasets/irony/irony_test.csv"
valid_irony = "../datasets/irony/irony_valid.csv"

loader = TransformerLoader()
#loader.load_dataset(train_sarcasm, valid_sarcasm, test_sarcasm, tokenizer, remove_hashtags=True)
loader.load_dataset(train_irony, valid_irony, test_irony, tokenizer, remove_hashtags=True)

batch_size = 16

train_dataloader = DataLoader(loader.train_dataset, batch_size=batch_size, shuffle=True)
valid_dataloader = DataLoader(loader.valid_dataset, batch_size=128, shuffle=False)

model = BertweetDeepModel(bertweet_model).to(device)

trainer = Trainer()
criterion = BCEWithLogitsLoss()

train_and_evaluate_deep_bertweet(model, train_dataloader, valid_dataloader, criterion, 3)