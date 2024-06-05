import torch
from sklearn import metrics

class Evaluator:
    def evaluate(self, device, test_dataset, model):
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
        print("F1: ", f1)