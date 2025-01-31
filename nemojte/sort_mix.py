import json
import pandas as pd
import math
import os

json_file_path = "training_dynamics_4epochs/bertweet_trainedon_semeval_mix_4epoch.json"
output_csv_path = "lowest_correctness/all_predictions/bertweet_trained_on_semeval_mix_4epoch.csv"

with open(json_file_path, 'r', encoding='utf-8') as f:
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
df_sorted = df.sort_values(by='mean_correctness')

df_sorted.to_csv(output_csv_path, index=False, encoding='utf-8')
