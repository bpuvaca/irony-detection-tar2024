import torch 
import torch.nn as nn 
import torch.optim as optim 
from transformers import DebertaModel, DebertaTokenizer  
from Loader import Loader 
from Evaluator import Evaluator
from Trainer import Trainer
from torchtext.vocab import GloVe

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 

class DebertaClassifier(nn.Module): 
    def __init__(self, model_name, num_classes): 
        super(DebertaClassifier, self).__init__() 
        self.deberta = DebertaModel.from_pretrained(model_name) 
        self.classifier = nn.Linear(self.deberta.config.hidden_size, num_classes)
    
    def forward(self, input_ids, attention_mask): 
        outputs = self.deberta(input_ids=input_ids, attention_mask=attention_mask) 
        pooled_output = outputs[1] 
        logits = self.classifier(pooled_output) 
        return logits 
        
glove = GloVe(name='6B', dim=300)
        
train_fp = "./new_datasets/irony/train_irony.csv" 
test_fp = "./new_datasets/irony/test_irony.csv" 

loader = Loader() 
loader.load_dataset(device, train_fp, test_fp, glove)

num_classes = 2 
batch_size = 16 
learning_rate = 2e-5 
num_epochs = 3 

model = DebertaClassifier("microsoft/deberta-base", num_classes).to(device) 
criterion = nn.CrossEntropyLoss()

trainer = Trainer() 
trainer.train(model, learning_rate, batch_size, num_epochs, loader.train_dataset, criterion) 

evaluator = Evaluator() 
evaluator.evaluate(model, loader.test_dataset, batch_size, device)


