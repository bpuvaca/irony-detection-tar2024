import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import f1_score
import numpy as np
from sklearn import metrics
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import accuracy_score, precision_score, recall_score
import os
import json
            
#import weave
#import wandb
#weave.init('irony-detection')

def train_baseline(device, model, learning_rate, batch_size, num_epochs, train_dataset, val_dataset, criterion):

    optimizer = optim.Adam(model.parameters(), lr=learning_rate, )

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    prev_val_f1 = 0
    for epoch in range(num_epochs):
        total_loss = 0
        for batch_inputs, batch_labels in train_loader:
            
            outputs = model(batch_inputs)
            loss = criterion(outputs, batch_labels)  
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Average Training Loss: {total_loss / len(train_loader)}")
        
        #Evaluation phase
        if (epoch + 1) % 3 == 0:
            model.eval()
            
            all_labels = []
            all_predictions = []

            with torch.no_grad():
                for padded_sequences, labels in val_loader:
                    padded_sequences = padded_sequences.to(device)
                    labels = labels.to(device)
                    
                    outputs = model(padded_sequences)
                    _, predicted = torch.max(outputs, 1)
                    
                    all_labels.extend(labels.cpu().numpy())
                    all_predictions.extend(predicted.cpu().numpy())

            # Calculate F1 score
            f1 = metrics.f1_score(all_labels, all_predictions, average='macro')
            print(f"Epoch {epoch + 1} Validation F1: ", f1)
            if f1 < prev_val_f1:
                print("F1 on validation set is declining, early stopping is activated")
                break
            else:
                prev_val_f1 = f1
                prev_params = model.state_dict()

    
    model.load_state_dict(prev_params)
    return model

def save_model(model, path):
    if not path.endswith(".pt"):
        path += ".pt"
    full_path = "../params/" + path
    torch.save(model.state_dict(), full_path)
    print(f"Model parameters saved to {full_path}")
    
#@weave.op()
def train_transformer(model, train_dataloader, val_dataloader, epochs=3, early_stopping=False, save_path: str = None, model_name="", trained_on="", return_all_preds=None, dataset_texts=None, fold_num=1):
    #run = wandb.init(project='irony-detection', name=f"{model_name}_{trained_on}_{fold_num+1}", group=f"{model_name}_{trained_on}", job_type='train')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    total_steps = len(train_dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
    prev_params = model.state_dict()
    prev_val_f1 = 0
    all_preds_saver = []
    
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
        if epoch != epochs - 1: continue
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
        
        if (epoch == epochs - 1) and return_all_preds and dataset_texts is not None:
            for i, (pred, label, tweet) in enumerate(zip(all_preds, all_labels, dataset_texts)):
                all_preds_saver.append((i+1, tweet[0], label, pred))
    

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

        #run.log({
        #    "epoch": epoch + 1,
        #    "f1": val_f1_score,
        #    "accuracy": val_accuracy,
        #})

        if early_stopping:
            if val_f1_score < prev_val_f1:
                print("F1 on validation set is declining, early stopping is activated")
                break
            else:
                prev_val_f1 = val_f1_score
                prev_params = model.state_dict()

    if early_stopping:
        model.load_state_dict(prev_params)

    #run.finish()

    if save_path is not None:
        save_model(model, save_path)
    if return_all_preds:
        return all_preds_saver, (val_f1_score, val_accuracy, val_precision, val_recall)
    else:        
        return (val_f1_score, val_accuracy, val_precision, val_recall)


def train_transformer_deep(model, train_dataloader, val_dataloader, epochs=10, early_stopping=False, save_path: str = None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    total_steps = len(train_dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
    prev_params = model.state_dict()
    prev_val_f1 = 0

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0

        for step, batch in enumerate(train_dataloader):
            batch_input_ids = batch['input_ids'].to(device)
            batch_attention_masks = batch['attention_mask'].to(device)
            batch_labels = batch['labels'].to(device)

            model.zero_grad()

            outputs = model(
                input_ids=batch_input_ids,
                attention_mask=batch_attention_masks
            )

            loss = nn.CrossEntropyLoss()(outputs, batch_labels)
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
                    input_ids=batch_input_ids,
                    attention_mask=batch_attention_masks
                )

                loss = nn.CrossEntropyLoss()(outputs, batch_labels)
                logits = outputs

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

        if early_stopping:
            if val_f1_score < prev_val_f1:
                print("F1 on validation set is declining, early stopping is activated")
                break
            else:
                prev_val_f1 = val_f1_score
                prev_params = model.state_dict()

    if early_stopping:
        model.load_state_dict(prev_params)
    
    if save_path is not None:
        if not save_path.endswith(".pt"):
            save_path += ".pt"
        torch.save(model.state_dict(), "../params/" + save_path)

    if save_path is not None:
        save_model(model, save_path)


def train_cartography(model, train_dataloader, epochs=3, save_path: str = None, model_name="", trained_on=""):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    total_steps = len(train_dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    dynamics_output_dir = "./training_dynamics_4epochs"
    os.makedirs(dynamics_output_dir, exist_ok=True)
    training_dynamics = {}

    for epoch in range(epochs):
        # Training phase
        model.train()
        total_train_loss = 0

        for step, batch in enumerate(train_dataloader):
            batch_input_ids = batch['input_ids'].to(device)
            batch_attention_masks = batch['attention_mask'].to(device)
            batch_labels = batch['labels'].to(device)
            batch_tweet_texts = batch['tweet_text']

            model.zero_grad()

            outputs = model(
                batch_input_ids,
                token_type_ids=None,
                attention_mask=batch_attention_masks,
                labels=batch_labels
            )

            loss = outputs.loss
            total_train_loss += loss.item()


            logits = outputs.logits

            probabilities = torch.softmax(logits, dim=-1)
            confidence = probabilities[range(probabilities.size(0)), batch_labels]
            predictions = torch.argmax(probabilities, dim=-1)

            for i in range(len(batch_labels)):

                data_id = f"{step * train_dataloader.batch_size + i}"
                if data_id not in training_dynamics:
                    training_dynamics[data_id] = {'confidence': [], 'correctness': [], 'label': batch_labels[i].item(), 'tweet_text': batch_tweet_texts[i]}

                training_dynamics[data_id]['confidence'].append(confidence[i].item())

                is_correct = int(predictions[i] == batch_labels[i])
                training_dynamics[data_id]['correctness'].append(is_correct)


            loss.backward()
            optimizer.step()
            scheduler.step()

            if step % 10 == 0 and step != 0:
                print(f"Epoch {epoch + 1}, Step {step}, Loss {loss.item()}")
        
        if epoch == epochs-1:
            with open(f"{dynamics_output_dir}/{model_name}_trainedon_{trained_on}_{epochs}epoch.json", 'w') as f:
                json.dump(training_dynamics, f)

        avg_train_loss = total_train_loss / len(train_dataloader)
        print(f"Epoch {epoch + 1}, Average Training Loss: {avg_train_loss}")

        
    if save_path is not None:
        save_model(model, save_path)
