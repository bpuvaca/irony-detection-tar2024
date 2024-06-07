from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from Loader import TransformerLoader
import train
import evaluate

class TransformerCNNModel(nn.Module):
    def __init__(self, base_model, num_labels):
        super(TransformerCNNModel, self).__init__()
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

def main():
<<<<<<< HEAD
    #transformer_model = "vinai/bertweet-base"
    transformer_model = "roberta-base"
=======
    transformer_model = "bert-base-uncased"
    #transformer_model = "vinai/bertweet-base"
    #transformer_model = "roberta-base"
>>>>>>> 780fda9e8703737678554b32aadf3b2c13555017
    tokenizer = AutoTokenizer.from_pretrained(transformer_model)
    base_model = AutoModel.from_pretrained(transformer_model)

    num_labels = 2
    model = TransformerCNNModel(base_model, num_labels)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    loader = TransformerLoader('mix')
    loader.load_dataset(tokenizer, remove_hashtags=True)

    batch_size = 16

    train_dataloader = DataLoader(loader.train_dataset, batch_size=batch_size, shuffle=True)
    valid_dataloader = DataLoader(loader.valid_dataset, batch_size=128, shuffle=False)
    test_dataloader = DataLoader(loader.test_dataset, batch_size=128, shuffle=False)

<<<<<<< HEAD
    save_path = "roberta_cnn/mix"
=======
    save_path = "bert_cnn/mix"
>>>>>>> 780fda9e8703737678554b32aadf3b2c13555017

    train.train_transformer_deep(model, train_dataloader, valid_dataloader, epochs=10, early_stopping=True, save_path=save_path)
    evaluate.evaluate_transformer_deep(model, test_dataloader)

if __name__ == "__main__":
    main()
