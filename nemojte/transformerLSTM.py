from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from Loader import TransformerLoader
import train
import evaluate

class TransformerBiLSTMModel(nn.Module):
    def __init__(self, base_model, num_labels, hidden_size=64, num_layers=2, dropout=0.1):
        super(TransformerBiLSTMModel, self).__init__()
        self.bert = base_model
        self.bilstm = nn.LSTM(input_size=768, hidden_size=hidden_size, num_layers=num_layers, bidirectional=True, batch_first=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size * 2, num_labels)

    def forward(self, input_ids, attention_mask=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        lstm_output, _ = self.bilstm(sequence_output)
        pooled_output = lstm_output.mean(dim=1)
        dropout_output = self.dropout(pooled_output)
        logits = self.fc(dropout_output)
        return logits

def main():
    transformer_model = "vinai/bertweet-base"
    tokenizer = AutoTokenizer.from_pretrained(transformer_model)
    base_model = AutoModel.from_pretrained(transformer_model)

    num_labels = 2
    model = TransformerBiLSTMModel(base_model, num_labels)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    train_sarcasm = "../datasets/sarcasm/sarcasm_train.csv"
    test_sarcasm = "../datasets/sarcasm/sarcasm_test.csv"
    valid_sarcasm = "../datasets/sarcasm/sarcasm_valid.csv"
    train_irony = "../datasets/irony/irony_train.csv"
    test_irony = "../datasets/irony/irony_test.csv"
    valid_irony = "../datasets/irony/irony_valid.csv"
    train_mix = "../datasets/mix/mix_train.csv"
    test_mix = "../datasets/mix/mix_test.csv"
    valid_mix = "../datasets/mix/mix_valid.csv"

    loader = TransformerLoader()
    #loader.load_dataset(train_sarcasm, valid_sarcasm, test_sarcasm, tokenizer, remove_hashtags=True, balance=True)
    loader.load_dataset(train_irony, valid_irony, test_irony, tokenizer, remove_hashtags=True, balance=True)
    #loader.load_dataset(train_mix, valid_mix, test_mix, tokenizer, remove_hashtags=True, balance=True)

    batch_size = 16

    train_dataloader = DataLoader(loader.train_dataset, batch_size=batch_size, shuffle=True)
    valid_dataloader = DataLoader(loader.valid_dataset, batch_size=128, shuffle=False)
    test_dataloader = DataLoader(loader.test_dataset, batch_size=128, shuffle=False)
    save_path = "bertweet_bilstm/irony"

    train.train_transformer_deep(model, train_dataloader, valid_dataloader, epochs=10, early_stopping=True, save_path=save_path)
    evaluate.evaluate_transformer_deep(model, test_dataloader)

if __name__ == "__main__":
    main()