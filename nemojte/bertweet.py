from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
import pandas as pd
from Loader import TransformerLoader
from torch.utils.data import DataLoader
import torch
from transformerUtils import train_and_evaluate_bertweet

tokenizer = AutoTokenizer.from_pretrained("vinai/bertweet-base")
model = AutoModelForSequenceClassification.from_pretrained("vinai/bertweet-base", num_labels=2)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

train_sarcasm = "../sarcasm_dataset/sarcasm_train.csv"
test_sarcasm = "../sarcasm_dataset/sarcasm_test.csv"
valid_sarcasm = "../sarcasm_dataset/sarcasm_validation.csv"
train_irony = "../irony_dataset/irony_train.csv"
test_irony = "../irony_dataset/irony_test.csv"
valid_irony = "../irony_dataset/irony_validation.csv"

loader = TransformerLoader()
#loader.load_dataset(train_sarcasm, valid_sarcasm, test_sarcasm, tokenizer, remove_hashtags=True)
loader.load_dataset(train_irony, test_irony, valid_irony, tokenizer, remove_hashtags=False)

batch_size = 16

# Create DataLoaders
train_dataloader = DataLoader(loader.train_dataset, batch_size=batch_size, shuffle=True)
valid_dataloader = DataLoader(loader.valid_dataset, batch_size=128, shuffle=True)

# Train and evaluate the model
train_and_evaluate_bertweet(model, train_dataloader, valid_dataloader, epochs=5)