import torch
from torch.utils.data import Dataset
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import f1_score
import numpy as np
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_score, recall_score

class TransformerDataset(Dataset):
    def __init__(self, tweets, labels, tokenizer, max_len=128):
        self.tweets = tweets
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.tweets)

    def __getitem__(self, idx):
        tweet = self.tweets[idx]
        label = self.labels[idx]
        encoding = self.tokenizer.encode_plus(
            tweet,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        return {
            'tweet_text': tweet,
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def train_and_evaluate(model, train_dataloader, val_dataloader, epochs=3):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    total_steps = len(train_dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    for epoch in range(epochs):
        # Training phase
        model.train()
        total_train_loss = 0

        for step, batch in enumerate(train_dataloader):
            batch_input_ids = batch['input_ids'].to(device)
            batch_attention_masks = batch['attention_mask'].to(device)
            batch_labels = batch['labels'].to(device)

            model.zero_grad()

            outputs = model(
                batch_input_ids,
                token_type_ids=None,
                attention_mask=batch_attention_masks,
                labels=batch_labels
            )

            loss = outputs.loss
            total_train_loss += loss.item()

            loss.backward()
            optimizer.step()
            scheduler.step()

            if step % 10 == 0 and step != 0:
                print(f"Epoch {epoch + 1}, Step {step}, Loss {loss.item()}")

        avg_train_loss = total_train_loss / len(train_dataloader)
        print(f"Epoch {epoch + 1}, Average Training Loss: {avg_train_loss}")

        # Evaluation phase
        model.eval()
        total_eval_loss = 0
        all_preds = []
        all_labels = []

        for batch in val_dataloader:
            batch_input_ids = batch['input_ids'].to(device)
            batch_attention_masks = batch['attention_mask'].to(device)
            batch_labels = batch['labels'].to(device)

            with torch.no_grad():
                outputs = model(
                    batch_input_ids,
                    token_type_ids=None,
                    attention_mask=batch_attention_masks,
                    labels=batch_labels
                )

                loss = outputs.loss
                logits = outputs.logits

            total_eval_loss += loss.item()

            logits = logits.detach().cpu().numpy()
            label_ids = batch_labels.to('cpu').numpy()

            all_preds.extend(np.argmax(logits, axis=1).flatten())
            all_labels.extend(label_ids.flatten())

        avg_val_loss = total_eval_loss / len(val_dataloader)
        val_f1_score = f1_score(all_labels, all_preds, average='macro')
        val_accuracy = accuracy_score(all_labels, all_preds)
        val_precision = precision_score(all_labels, all_preds, average='macro')
        val_recall = recall_score(all_labels, all_preds, average='macro')

        print(f"Epoch {epoch + 1}, Validation Loss: {avg_val_loss}")
        print(f"Epoch {epoch + 1}, Validation F1 Score: {val_f1_score}")
        print(f"Epoch {epoch + 1}, Validation Accuracy: {val_accuracy}")
        print(f"Epoch {epoch + 1}, Validation Precision: {val_precision}")
        print(f"Epoch {epoch + 1}, Validation Recall: {val_recall}")

def train_and_evaluate_deep_bertweet(model, train_dataloader, val_dataloader, criterion, epochs=3):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-8, weight_decay=1e-5, eps=1e-8)
    model.to(device)

    for epoch in range(epochs):
        # Training phase
        model.train()
        total_train_loss = 0

        for step, batch in enumerate(train_dataloader):
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()

            logits = model(input_ids)
            loss = criterion(logits, labels.float())
            total_train_loss += loss.item()

            loss.backward()
            optimizer.step()
            if step % 10 == 0 and step != 0:
                print(f"Epoch {epoch + 1}, Step {step}, Loss {loss.item()}")

        avg_train_loss = total_train_loss / len(train_dataloader)
        print(f"Epoch {epoch + 1}, Average Training Loss: {avg_train_loss}")

        # Evaluation phase
        model.eval()
        total_eval_loss = 0
        all_preds = []
        all_labels = []

        for batch in val_dataloader:
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)

            with torch.no_grad():
                logits = model(input_ids)
                loss = criterion(logits, labels.float())
                total_eval_loss += loss.item()

                preds = torch.sigmoid(logits).round().cpu().numpy()
                label_ids = labels.cpu().numpy()

                all_preds.extend(preds)
                all_labels.extend(label_ids)

        avg_val_loss = total_eval_loss / len(val_dataloader)
        val_f1_score = f1_score(all_labels, all_preds, average='macro')
        val_accuracy = accuracy_score(all_labels, all_preds)
        val_precision = precision_score(all_labels, all_preds, average='macro')
        val_recall = recall_score(all_labels, all_preds, average='macro')

        print(f"Epoch {epoch + 1}, Validation Loss: {avg_val_loss}")
        print(f"Epoch {epoch + 1}, Validation F1 Score: {val_f1_score}")
        print(f"Epoch {epoch + 1}, Validation Accuracy: {val_accuracy}")
        print(f"Epoch {epoch + 1}, Validation Precision: {val_precision}")
        print(f"Epoch {epoch + 1}, Validation Recall: {val_recall}")

