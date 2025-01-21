import pandas as pd 
import math

other = pd.read_csv('datasets/crossval/other.csv', sep=',')
polarity = pd.read_csv('datasets/crossval/polarity.csv', sep=',')

semeval_mix = pd.concat([other, polarity], ignore_index=True)
print(len(semeval_mix[semeval_mix['label'] == 1]))
print(len(semeval_mix[semeval_mix['label'] == 0]))
semeval_mix = semeval_mix.sample(frac=1).reset_index(drop=True)
semeval_mix['index'] = range(len(semeval_mix))
semeval_mix.to_csv('datasets/crossval/semeval_mix.csv', index=False)

irony = pd.read_csv('datasets/crossval/irony.csv', sep=',')
sarcasm = pd.read_csv('datasets/crossval/sarcasm.csv', sep=',')

isarcasm_mix = pd.concat([irony, sarcasm], ignore_index=True)

downscale_size = len(irony)
k = 5


fold_size = downscale_size // k
j_range = len(corpus) // downscale_size
i_range = math.ceil(k / j_range)
segments = [(corpus[j*downscale_size:(j+1)*downscale_size], labels[j*downscale_size:(j+1)*downscale_size]) for j in range(j_range)]
total_cnt = 0
for i in range(i_range):
    for j in range(j_range):
        if total_cnt >= k:
            break
        start_index = j * downscale_size
        train_corpus = segments[j][0][:i*fold_size] + (segments[j][0][(i+1)*fold_size:] if i < k-1 else [])
        train_labels = segments[j][1][:i*fold_size] + (segments[j][1][(i+1)*fold_size:] if i < k-1 else [])
        self.train_datasets[total_cnt] = TransformerDataset(train_corpus, train_labels, tokenizer)
        valid_corpus = segments[j][0][i*fold_size:(i+1)*fold_size] if i < k-1 else corpus[i*fold_size:]
        valid_labels = segments[j][1][i*fold_size:(i+1)*fold_size] if i < k-1 else labels[i*fold_size:]
        self.valid_datasets[total_cnt] = TransformerDataset(valid_corpus, valid_labels, tokenizer)
        self.test_texts[total_cnt] = list(zip(valid_corpus, valid_labels))
        total_cnt += 1

