from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np

# Load BERTweet tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("vinai/bertweet-base")
model = AutoModelForSequenceClassification.from_pretrained("vinai/bertweet-base", num_labels=2)  # Binary classification

#Prepare the Dataset
from torch.utils.data import DataLoader, TensorDataset
import torch

def preprocess_data(texts, labels, tokenizer, max_len):
    input_ids = []
    attention_masks = []

    for text in texts:
        encoded_dict = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])

    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    labels = torch.tensor(labels)

    return TensorDataset(input_ids, attention_masks, labels)

# Example data
texts = ["I love using BERTweet!", "I dislike this model."]
labels = [1, 0]  # Binary labels

dataset = preprocess_data(texts, labels, tokenizer, max_len=128)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

#Define the Training Loop:

from transformers import AdamW, get_linear_schedule_with_warmup

def train(model, dataloader, epochs=3):
    # Set the model in training mode
    model.train()

    optimizer = AdamW(model.parameters(), lr=2e-5)
    total_steps = len(dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    for epoch in range(epochs):
        for step, batch in enumerate(dataloader):
            batch_input_ids, batch_attention_masks, batch_labels = batch
            
            model.zero_grad()
            outputs = model(
                batch_input_ids,
                token_type_ids=None,
                attention_mask=batch_attention_masks,
                labels=batch_labels
            )
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            scheduler.step()

            if step % 10 == 0 and step != 0:
                print(f"Epoch {epoch + 1}, Step {step}, Loss {loss.item()}")

# Train the model
train(model, dataloader)

#evaulate the model
def evaluate(model, dataloader):
    model.eval()
    total_eval_accuracy = 0
    total_eval_loss = 0

    for batch in dataloader:
        batch_input_ids, batch_attention_masks, batch_labels = batch

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
        total_eval_accuracy += flat_accuracy(logits, label_ids)

    avg_val_accuracy = total_eval_accuracy / len(dataloader)
    print(f"Validation Accuracy: {avg_val_accuracy}")
    avg_val_loss = total_eval_loss / len(dataloader)
    print(f"Validation Loss: {avg_val_loss}")

# Function to calculate the accuracy of our predictions vs labels
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

# Assuming you have a validation dataloader
evaluate(model, val_dataloader)
