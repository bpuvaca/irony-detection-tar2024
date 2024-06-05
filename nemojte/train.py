import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import f1_score
import numpy as np
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_score, recall_score
            
def train_baseline(self, model, learning_rate, batch_size, num_epochs, train_dataset, criterion):

    optimizer = optim.Adam(model.parameters(), lr=learning_rate, )

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(num_epochs):
        total_loss = 0
        for batch_inputs, batch_labels in train_loader:
            
            outputs = model(batch_inputs)
            loss = criterion(outputs, batch_labels)  
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss / len(train_loader)}")
    
    return model

def train_eval_test_bertweet(model, train_dataloader, val_dataloader, test_dataloader, epochs=3, early_stopping=False, save_path:str=None):
    """

    Args:
    save_path(str): parameters of the trained model will be saved to params/[save_path], 
    first folder should describe the model and second should describe 
    the dataset used for training, 
    e.g. save_path="bertweet/sarcasm.pt"
    default value is None and the params won't be saved 
    
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    total_steps = len(train_dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
    prev_params = model.state_dict()
    prev_val_f1 = 0

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

        if (early_stopping):
            if (val_f1_score < prev_val_f1):
                print("F1 on validation set is declining, early stopping is activated")
                break
            else:
                prev_val_f1 = val_f1_score
                prev_params = model.state_dict()
            
    if (early_stopping): model.load_state_dict(prev_params)
    
    if save_path is not None:
        if not save_path.endswith(".pt"):
            save_path += ".pt"
        torch.save(model.state_dict(), "../params/" + save_path)


    # Test phase
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

    print(f"Test Loss: {avg_test_loss}")
    print(f"Epoch {epoch + 1}, Validation F1 Score: {test_f1_score}")
    print(f"Epoch {epoch + 1}, Validation Accuracy: {test_accuracy}")
    print(f"Epoch {epoch + 1}, Validation Precision: {test_precision}")
    print(f"Epoch {epoch + 1}, Validation Recall: {test_recall}")

def train_eval_test_transformer_deep(model, train_dataloader, val_dataloader, test_dataloader, epochs=10, early_stopping=False, save_path:str=None):
    """

    Args:
    save_path(str): parameters of the trained model will be saved to params/[save_path], 
    first folder should describe the model and second should describe 
    the dataset used for training, 
    e.g. save_path="transformerCNN/sarcasm"
    default value is None and the params won't be saved 

    """
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

    # Test phase
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

    print(f"Test Loss: {avg_test_loss}")
    print(f"Test F1 Score: {test_f1_score}")
    print(f"Test Accuracy: {test_accuracy}")
    print(f"Test Precision: {test_precision}")
    print(f"Test Recall: {test_recall}")