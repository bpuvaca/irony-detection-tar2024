from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
import pandas as pd
from Loader import TransformerLoader
from torch.utils.data import DataLoader
import torch
from transformerUtils import train_and_evaluate

tokenizer = AutoTokenizer.from_pretrained("vinai/bertweet-base")
model = AutoModelForSequenceClassification.from_pretrained("vinai/bertweet-base", num_labels=2)  # Binary classification

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

train_sarcasm = "../new_datasets/sarcasm/train_sarcasm.csv"
test_sarcasm = "../new_datasets/sarcasm/test_sarcasm.csv"
train_irony = "../new_datasets/irony/train_irony.csv"
test_irony = "../new_datasets/irony/test_irony.csv"

loader = TransformerLoader()
loader.load_dataset(train_irony, test_irony, tokenizer)

batch_size = 32

# Create DataLoaders
dataloader = DataLoader(loader.train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(loader.test_dataset, batch_size=batch_size, shuffle=False)

# Train and evaluate the model
train_and_evaluate(model, dataloader, val_dataloader, epochs=5)