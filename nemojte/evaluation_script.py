import sys
from transformers import AutoTokenizer, AutoModel
import torch
from torch.utils.data import DataLoader
from Loader import TransformerLoader, GloveLoader
import evaluate
from transformerLSTM import TransformerBiLSTMModel
from transformerCNN import TransformerCNNModel

save_path = sys.argv[1]
save_path = "../params/" + save_path

test_sarcasm = "../datasets/sarcasm/sarcasm_test.csv"
test_irony = "../datasets/irony/irony_test.csv"
test_mix = "../datasets/mix/mix_test.csv"
test_taskA = "../datasets/taskA/taskA_test.csv"

transformer_model = "vinai/bertweet-base"
#transformer_model = "roberta-base"
tokenizer = AutoTokenizer.from_pretrained(transformer_model)
base_model = AutoModel.from_pretrained(transformer_model)

loader = TransformerLoader()
#loader = GloveLoader()

model = TransformerBiLSTMModel(base_model, num_labels=2)
model.load_state_dict(torch.load(save_path))

print("Testing on Task A")
loader.load_test_dataset(test_taskA, tokenizer, remove_hashtags=True, balance=False)
evaluate.evaluate_transformer_deep(model, DataLoader(loader.test_dataset, batch_size=128, shuffle=False))

print("Testing on irony")
loader.load_test_dataset(test_irony, tokenizer, remove_hashtags=True, balance=True)
evaluate.evaluate_transformer_deep(model, DataLoader(loader.test_dataset, batch_size=128, shuffle=False))

print("Testing on sarcasm")
loader.load_test_dataset(test_sarcasm, tokenizer, remove_hashtags=True, balance=True)
evaluate.evaluate_transformer_deep(model, DataLoader(loader.test_dataset, batch_size=128, shuffle=False))

print("Testing on mix")
loader.load_test_dataset(test_mix, tokenizer, remove_hashtags=True, balance=True)
evaluate.evaluate_transformer_deep(model, DataLoader(loader.test_dataset, batch_size=128, shuffle=False))