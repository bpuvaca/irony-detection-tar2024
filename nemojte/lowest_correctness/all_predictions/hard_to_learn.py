import json
import pandas as pd
import math
import os

folder_path = "training_dynamics_4epochs"
output_path = "lowest_correctness"

for filename in os.listdir(folder_path):
    if 'mix' not in filename:
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

                results.append({
                    'tweet': tweet,
                    'label': label,
                    'mean_confidence': mean_confidence,
                    'variability': variability,
                    'mean_correctness': mean_correctness,
                    'correctnesses': correctnesses
                })

            df = pd.DataFrame(results)
            lowest = df.nsmallest(100, mean_correctness)
            
            output_csv_path = os.path.join(f"lowest_correctness/{filename.replace('.json', '.csv')}")
            df.to_csv(output_csv_path, index=False, encoding='utf-8')
