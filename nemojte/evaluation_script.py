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

transformer_model = "vinai/bertweet-base"
#transformer_model = "roberta-base"
tokenizer = AutoTokenizer.from_pretrained(transformer_model)
base_model = AutoModel.from_pretrained(transformer_model)

model = TransformerBiLSTMModel(base_model, num_labels=2)
model.load_state_dict(torch.load(save_path))

print("Testing on Task A")
loader = TransformerLoader('task_A')
loader.load_test_dataset(tokenizer, remove_hashtags=True)
evaluate.evaluate_transformer_deep(model, DataLoader(loader.test_dataset, batch_size=128, shuffle=False))

print("Testing on irony")
loader = TransformerLoader('irony')
loader.load_test_dataset(tokenizer, remove_hashtags=True)
evaluate.evaluate_transformer_deep(model, DataLoader(loader.test_dataset, batch_size=128, shuffle=False))

print("Testing on sarcasm")
loader = TransformerLoader('sarcasm')
loader.load_test_dataset(tokenizer, remove_hashtags=True)
evaluate.evaluate_transformer_deep(model, DataLoader(loader.test_dataset, batch_size=128, shuffle=False))

print("Testing on mix")
loader = TransformerLoader('mix')
loader.load_test_dataset(tokenizer, remove_hashtags=True)
evaluate.evaluate_transformer_deep(model, DataLoader(loader.test_dataset, batch_size=128, shuffle=False))