from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
import pandas as pd
from Loader import TransformerLoader
from torch.utils.data import DataLoader
import torch
import train
import evaluate

#transformer_model = "bert-base-uncased"
transformer_model = "vinai/bertweet-base"
tokenizer = AutoTokenizer.from_pretrained(transformer_model)
model = AutoModelForSequenceClassification.from_pretrained(transformer_model, num_labels=2)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

loader = TransformerLoader('mix', mixed_not_balanced=True)
loader.load_dataset(tokenizer, remove_hashtags=True)

batch_size = 16

# Create DataLoaders
train_dataloader = DataLoader(loader.train_dataset, batch_size=batch_size, shuffle=True)
valid_dataloader = DataLoader(loader.valid_dataset, batch_size=128, shuffle=False)
test_dataloader = DataLoader(loader.test_dataset, batch_size=128, shuffle=False)

save_path = "bertweet/mix_not_b"

# Train and evaluate the model
train.train_bertweet(model, train_dataloader, valid_dataloader, epochs=10, early_stopping=True, save_path=save_path)
evaluate.evaluate_bertweet(model, test_dataloader)