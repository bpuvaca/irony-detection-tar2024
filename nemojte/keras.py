from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformerUtils import train_eval_test_roberta
from Loader import TransformerLoader


class ConvRobertaModel(nn.Module):
    def __init__(self, base_model, num_labels, max_len):
        super(ConvRobertaModel, self).__init__()
        self.bert = base_model
        self.conv1 = nn.Conv1d(in_channels=768, out_channels=256, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=256, out_channels=128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(4096, num_labels)  

    def forward(self, input_ids, attention_mask=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state.transpose(1, 2) 
        conv_output = self.pool(nn.ReLU()(self.conv1(sequence_output)))
        conv_output = self.pool(nn.ReLU()(self.conv2(conv_output)))
        flat_output = conv_output.view(conv_output.size(0), -1)  
        dropout_output = self.dropout(flat_output)
        logits = self.fc(dropout_output)
        return logits

tokenizer = AutoTokenizer.from_pretrained("roberta-base")
base_model = AutoModel.from_pretrained("roberta-base")

num_labels = 2
max_len = 200
model = ConvRobertaModel(base_model, num_labels, max_len)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

train_sarcasm = "../datasets/irony_taskA/irony_train_taskA.csv"
test_sarcasm = "../datasets/irony_taskA/irony_test_taskA.csv"
valid_sarcasm = "../datasets/irony_taskA/irony_validation_taskA.csv"

loader = TransformerLoader()
loader.load_dataset(train_sarcasm, valid_sarcasm, test_sarcasm, tokenizer, remove_hashtags=True)

batch_size = 16

train_dataloader = DataLoader(loader.train_dataset, batch_size=batch_size, shuffle=True)
valid_dataloader = DataLoader(loader.valid_dataset, batch_size=128, shuffle=False)
test_dataloader = DataLoader(loader.test_dataset, batch_size=128, shuffle=False)

train_eval_test_roberta(model, train_dataloader, valid_dataloader, test_dataloader, epochs=5, early_stopping=True)
