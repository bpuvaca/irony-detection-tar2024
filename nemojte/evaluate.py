import torch
from sklearn import metrics
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
import csv

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
    test_accuracy = accuracy_score(all_labels, all_predictions)
    test_precision = precision_score(all_labels, all_predictions, average='macro')
    test_recall = recall_score(all_labels, all_predictions, average='macro')

    print(f"Test F1 Score: {f1:.3f}")
    print(f"Test Accuracy: {test_accuracy:.3f}")
    print(f"Test Precision: {test_precision:.3f}")
    print(f"Test Recall: {test_recall:.3f}")
    return f1, test_accuracy, test_precision, test_recall
    
def evaluate_transformer(model, test_dataloader, model_name="", trained_on="", eval_on="", return_wrong_preds=False, return_all_preds=False, dataset_texts=None, load_from=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    total_test_loss = 0
    all_preds = []
    all_labels = []
    wrong_preds_saver = []
    all_preds_saver = []

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

        preds = np.argmax(logits, axis=1).flatten()

        all_preds.extend(np.argmax(logits, axis=1).flatten())
        all_labels.extend(label_ids.flatten())

        if return_wrong_preds and dataset_texts is not None:
            for i, (pred, label) in enumerate(zip(preds, label_ids)):
                if pred != label:
                    wrong_preds_saver.append((i+1, (dataset_texts[i], pred)))
        
        # if return_all_preds and dataset_texts is not None:
        #     for i, (pred, label) in enumerate(zip(preds, label_ids)):
        #         all_preds_saver.append((i+1, (dataset_texts[i], pred)))
                
    if return_all_preds and dataset_texts is not None:
        for i, (pred, label, tweet) in enumerate(zip(all_preds, all_labels, dataset_texts)):
            all_preds_saver.append((i+1, tweet[0], label, pred))
    
    avg_test_loss = total_test_loss / len(test_dataloader)
    test_f1_score = f1_score(all_labels, all_preds, average='macro')
    test_accuracy = accuracy_score(all_labels, all_preds)
    test_precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    test_recall = recall_score(all_labels, all_preds, average='macro')

    print(f"Test Loss: {avg_test_loss:.3f}")
    print(f"Test F1 Score: {test_f1_score:.3f}")
    print(f"Test Accuracy: {test_accuracy:.3f}")
    print(f"Test Precision: {test_precision:.3f}")
    print(f"Test Recall: {test_recall:.3f}")

    # if wrong_preds_saver:
    #     filename = model_name + "+" + trained_on + "_test_on_" + eval_on + ".csv"
    #     with open("wrong_preds/" + filename, "w") as file:
    #         file.write("index,tweet,label,prediction\n")
    #         for wrong_pred in wrong_preds_saver:
    #             file.write(f"{wrong_pred[0]},\"{wrong_pred[1][0][0]}\",{wrong_pred[1][0][1]},{wrong_pred[1][1]}\n")

    #     print(f"Check nemojte/wrong_preds/{filename} for wrong preds")
    
    # if all_preds_saver:
    #     if load_from:
    #         filename = load_from + "_test_on_" + eval_on + ".csv"
    #     else:
    #         filename = model_name + "+" + trained_on + "_test_on_" + eval_on + ".csv"
    #     with open("all_preds/" + filename, "w", encoding="utf-8", newline='') as csvfile:
    #         csvwriter = csv.writer(csvfile)
    #         csvwriter.writerow(["index", "tweet", "label", "prediction"])
    #         for pred in all_preds_saver:
    #             csvwriter.writerow([pred[0], pred[1], pred[2], pred[3]])
        
    #     print(f"Check nemojte/all_preds/{filename} for all preds")
    if all_preds_saver:
        return all_preds_saver, (test_f1_score, test_accuracy, test_precision, test_recall)


def get_probabilities(model, test_dataloader):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    probabilities = []

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

        for i, logit in enumerate(logits):
            prob_1 = torch.softmax(torch.tensor(logit), dim=-1).numpy()[1]
            probabilities.append((batch['ids'][i].item(), prob_1))
    
    return probabilities


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
    test_precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    test_recall = recall_score(all_labels, all_preds, average='macro')

    print(f"Test Loss: {avg_test_loss:.3f}")
    print(f"Test F1 Score: {test_f1_score:.3f}")
    print(f"Test Accuracy: {test_accuracy:.3f}")
    print(f"Test Precision: {test_precision:.3f}")
    print(f"Test Recall: {test_recall:.3f}")
