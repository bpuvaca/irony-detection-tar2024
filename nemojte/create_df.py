import json
import pandas as pd
import math


filename = "bertweet_trainedon_semeval_mix_4epoch"


with open(f'training_dynamics_4epochs/{filename}.json', 'r') as f:
    data = json.load(f)


#data_ids = []
mean_confidences = []
mean_variabilities = []
all_correctnesses = []

for data_id, dynamics in data.items():
    confidences = dynamics['confidence']
    correctnesses = dynamics['correctness']
    
    mean_confidence = sum(confidences) / len(confidences)
    variability = math.sqrt(sum((c - mean_confidence) ** 2 for c in confidences) / len(confidences))
    correctness = float(sum(correctnesses) / len(correctnesses))
    
    #data_ids.append(data_id)
    mean_confidences.append(mean_confidence)
    mean_variabilities.append(variability)
    all_correctnesses.append(correctness)


df = pd.DataFrame({
    'confidence': mean_confidences,
    'variability': mean_variabilities,
    'correctness': all_correctnesses
})

df.to_csv(f'training_dynamics_4epochs/{filename}.csv', index=False)
