import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import sys

if len(sys.argv) != 2:
    print("Usage: python metrics.py <csv_filename>")
    sys.exit(1)

csv_filename = sys.argv[1]

def calculate_metrics(csv_filename):
    data = pd.read_csv(csv_filename)
    
    y_true = data['Label'].astype(int)
    y_pred = data['Prediction'].astype(int)
    
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    
    print(f'Accuracy: {accuracy}')
    print(f'Precision: {precision}')
    print(f'Recall: {recall}')
    print(f'F1 Score: {f1}')

calculate_metrics(csv_filename)
