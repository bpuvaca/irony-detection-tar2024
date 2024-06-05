import torch
from transformers import BertTokenizer, BertModel
import torch.nn as nn
from torch.utils.data import DataLoader

from Loader import TransformerLoader
from transformerUtils import train_and_evaluate_deep_bertweet

from Trainer import Trainer
from torch.nn import BCELoss

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class BertweetDeepModel(nn.Module):
   def __init__(self, bertweet_model):
      super(BertweetDeepModel, self).__init__()
      self.bertweet_model = bertweet_model
      self.bilstm = nn.LSTM(bidirectional=True, input_size=768, hidden_size=256, num_layers=2, batch_first=True)
      self.maxpool = nn.AdaptiveMaxPool1d(output_size=1024)

   def forward(self, input_ids):
      outputs = self.bertweet_model(input_ids)
      embeddings = outputs.last_hidden_state
      lstm_output, _ = self.bilstm(embeddings)
      concatenated_output = torch.cat((lstm_output, embeddings), dim=2)
      pooled_output = self.maxpool(concatenated_output.transpose(1, 2)).squeeze(2)
      return pooled_output

# Load the BERTweet model
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
loader.load_dataset(train_sarcasm, valid_sarcasm, test_sarcasm, tokenizer, remove_hashtags=True)

batch_size = 16

train_dataloader = DataLoader(loader.train_dataset, batch_size=batch_size, shuffle=True)
valid_dataloader = DataLoader(loader.valid_dataset, batch_size=batch_size, shuffle=False)

model = BertweetDeepModel(bertweet_model).to(device)

trainer = Trainer()
criterion = BCELoss()

train_and_evaluate_deep_bertweet(model, train_dataloader, valid_dataloader, criterion, 3)