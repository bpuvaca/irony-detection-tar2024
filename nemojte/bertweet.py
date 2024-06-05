from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
import pandas as pd
from Loader import TransformerLoader
from torch.utils.data import DataLoader
import torch
import train
import evaluate

tokenizer = AutoTokenizer.from_pretrained("vinai/bertweet-base")
model = AutoModelForSequenceClassification.from_pretrained("vinai/bertweet-base", num_labels=2)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

train_sarcasm = "../datasets/sarcasm/sarcasm_train.csv"
test_sarcasm = "../datasets/sarcasm/sarcasm_test.csv"
valid_sarcasm = "../datasets/sarcasm/sarcasm_valid.csv"
train_irony = "../datasets/irony/irony_train.csv"
test_irony = "../datasets/irony/irony_test.csv"
valid_irony = "../datasets/irony/irony_valid.csv"

loader = TransformerLoader()
loader.load_dataset(train_sarcasm, valid_sarcasm, test_sarcasm, tokenizer, remove_hashtags=True)
#loader.load_dataset(train_irony, valid_irony, test_irony, tokenizer, remove_hashtags=True)

batch_size = 16

# Create DataLoaders
train_dataloader = DataLoader(loader.train_dataset, batch_size=batch_size, shuffle=True)
valid_dataloader = DataLoader(loader.valid_dataset, batch_size=128, shuffle=False)
test_dataloader = DataLoader(loader.test_dataset, batch_size=128, shuffle=False)

# Train and evaluate the model
train.train_bertweet(model, train_dataloader, valid_dataloader, epochs=10, early_stopping=True)
evaluate.evaluate_bertweet(model, test_dataloader)