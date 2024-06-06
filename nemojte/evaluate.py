import torch
from sklearn import metrics
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np

def evaluate_baseline(device, test_dataset, model):
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    all_labels = []
    all_predictions = []
    
    with torch.no_grad():
        for padded_sequences, labels in test_loader:
            padded_sequences = padded_sequences.to(device)
            labels = labels.to(device)
            
            outputs = model(padded_sequences)
            _, predicted = torch.max(outputs, 1)
            
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
    
    # Calculate F1 score
    f1 = metrics.f1_score(all_labels, all_predictions, average='macro')
    print("Test F1: ", f1)
    return f1
    
def evaluate_bertweet(model, test_dataloader):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    total_test_loss = 0
    all_preds = []
    all_labels = []

    for batch in test_dataloader:
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

        total_test_loss += loss.item()

        logits = logits.detach().cpu().numpy()
        label_ids = batch_labels.to('cpu').numpy()

        all_preds.extend(np.argmax(logits, axis=1).flatten())
        all_labels.extend(label_ids.flatten())

    avg_test_loss = total_test_loss / len(test_dataloader)
    test_f1_score = f1_score(all_labels, all_preds, average='macro')
    test_accuracy = accuracy_score(all_labels, all_preds)
    test_precision = precision_score(all_labels, all_preds, average='macro')
    test_recall = recall_score(all_labels, all_preds, average='macro')

    print(f"Test Loss: {avg_test_loss:.3f}")
    print(f"Test F1 Score: {test_f1_score:.3f}")
    print(f"Test Accuracy: {test_accuracy:.3f}")
    print(f"Test Precision: {test_precision:.3f}")
    print(f"Test Recall: {test_recall:.3f}")


def evaluate_transformer_deep(model, test_dataloader):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    total_test_loss = 0
    all_preds = []
    all_labels = []

    for batch in test_dataloader:
        batch_input_ids = batch['input_ids'].to(device)
        batch_attention_masks = batch['attention_mask'].to(device)
        batch_labels = batch['labels'].to(device)

        with torch.no_grad():
            outputs = model(
                input_ids=batch_input_ids,
                attention_mask=batch_attention_masks
            )

            loss = nn.CrossEntropyLoss()(outputs, batch_labels)
            logits = outputs

        total_test_loss += loss.item()

        logits = logits.detach().cpu().numpy()
        label_ids = batch_labels.to('cpu').numpy()

        all_preds.extend(np.argmax(logits, axis=1).flatten())
        all_labels.extend(label_ids.flatten())

    avg_test_loss = total_test_loss / len(test_dataloader)
    test_f1_score = f1_score(all_labels, all_preds, average='macro')
    test_accuracy = accuracy_score(all_labels, all_preds)
    test_precision = precision_score(all_labels, all_preds, average='macro')
    test_recall = recall_score(all_labels, all_preds, average='macro')

    print(f"Test Loss: {avg_test_loss:.3f}")
    print(f"Test F1 Score: {test_f1_score:.3f}")
    print(f"Test Accuracy: {test_accuracy:.3f}")
    print(f"Test Precision: {test_precision:.3f}")
    print(f"Test Recall: {test_recall:.3f}")
