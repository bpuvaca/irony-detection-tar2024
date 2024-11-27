import json
import pandas as pd
import math
import os
from tabulate import tabulate

def wrap_text(text, width=50):
    return '\n'.join([text[i:i+width] for i in range(0, len(text), width)])

folder_path = "training_dynamics"
output_file = "lowest_sums_output.txt"

with open(output_file, 'w', encoding='utf-8') as out_file:
    for filename in os.listdir(folder_path):
        if filename.endswith(".json"):
            filepath = os.path.join(folder_path, filename)
            
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            results = []

            for data_id, dynamics in data.items():
                confidences = dynamics['confidence']
                correctnesses = dynamics['correctness']
                tweet = dynamics['tweet_text']
                label = dynamics['label']
                
                mean_confidence = sum(confidences) / len(confidences)
                variability = math.sqrt(sum((c - mean_confidence) ** 2 for c in confidences) / len(confidences))
                mean_correctness = sum(correctnesses) / len(correctnesses)
                
                tezinska_suma = mean_confidence + mean_correctness - variability
                
                results.append({
                    'tweet': tweet,
                    'label': label,
                    'sum': tezinska_suma,
                    'mean_confidence': mean_confidence,
                    'variability': variability,
                    'mean_correctness': mean_correctness
                })

            df = pd.DataFrame(results)
            lowest_sums = df.nsmallest(15, 'sum')

            table_data = [
                [
                    wrap_text(row['tweet'], width=50),
                    row['label'],
                    f"{row['sum']:.4f}",
                    f"{row['mean_confidence']:.4f}",
                    f"{row['variability']:.4f}",
                    f"{row['mean_correctness']:.4f}"
                ]
                for _, row in lowest_sums.iterrows()
            ]

            out_file.write(f"File: {filename}\n")
            out_file.write(tabulate(
                table_data,
                headers=["Tweet", "Label", "Sum", "Mean Confidence", "Variability", "Mean Correctness"],
                tablefmt="grid"
            ))
            out_file.write("\n\n")
