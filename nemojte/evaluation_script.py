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

transformer_model = "vinai/bertweet-base"
#transformer_model = "roberta-base"
tokenizer = AutoTokenizer.from_pretrained(transformer_model)
base_model = AutoModel.from_pretrained(transformer_model)

loader = TransformerLoader()
#loader = GloveLoader()
loader.load_test_dataset(test_irony, tokenizer, remove_hashtags=True)

model = TransformerBiLSTMModel(base_model, num_labels=2)

model.load_state_dict(torch.load(save_path))

evaluate.evaluate_transformer_deep(model, DataLoader(loader.test_dataset, batch_size=128, shuffle=False))