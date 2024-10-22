import argparse
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
import pandas as pd
from Loader import TransformerLoader
from torch.utils.data import DataLoader
import torch
import train
import evaluate

def parse_args():
    parser = argparse.ArgumentParser(description='Process some arguments.')
    parser.add_argument('--ds', type=str, required=False, help='Name of the dataset')
    parser.add_argument('--eval_on', type=str, required=False, help='Dataset(s) to evaluate on')
    parser.add_argument('--model', type=str, required=True, help='Model name')
    parser.add_argument('--load_from', type=str, required=False, help='Source')
    parser.add_argument('--save_to', type=str, required=False, help='Destination')

    args = parser.parse_args()
    return args

def map_model_name(model_name):
    model_dict = {
        "roberta": "roberta-base",
        "bertweet": "vinai/bertweet-base",
        "bert": "bert-base-uncased"
    }
    if model_name not in model_dict:
        raise ValueError("Model not found.")
    return model_dict[model_name]

def load_model(transformer_model, load_from):
    model = AutoModelForSequenceClassification.from_pretrained(transformer_model, num_labels=2)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    if load_from:
        load_path = f"../params/{load_from}.pt"
        model.load_state_dict(torch.load(load_path, map_location=device, weights_only=True))
    
    return model

def load_dataset(dataset_name, tokenizer):
    loader = TransformerLoader(dataset_name)
    loader.load_dataset(tokenizer, remove_hashtags=True)

    batch_size = 16

    # Create DataLoaders
    train_dataloader = DataLoader(loader.train_dataset, batch_size=batch_size, shuffle=True)
    valid_dataloader = DataLoader(loader.valid_dataset, batch_size=128, shuffle=False)
    test_dataloader = DataLoader(loader.test_dataset, batch_size=128, shuffle=False)

    return train_dataloader, valid_dataloader, test_dataloader

def train_and_evaluate(dataset, model_name, load_from=None, save_to=None, eval_on=None):
    print(f"Training {model_name} on {dataset}")
    transformer_model = map_model_name(model_name)
    model = load_model(transformer_model, load_from)
    tokenizer = AutoTokenizer.from_pretrained(transformer_model)

    if not load_from:
        train_dataloader, valid_dataloader, test_dataloader = load_dataset(dataset, tokenizer)
        save_path = save_to if save_to else None
        train.train_transformer(model, train_dataloader, valid_dataloader, epochs=10, early_stopping=True, save_path=save_path)

    if not eval_on:
        evaluate.evaluate_transformer(model, test_dataloader)
    else:
        eval_on = eval_on.split(" ")
        for dataset in eval_on:
            _, _, test_dataloader = load_dataset(dataset, tokenizer)
            print(f"Evaluating on {dataset}")
            evaluate.evaluate_transformer(model, test_dataloader)

if __name__ == "__main__":
    args = parse_args()
    if args.load_from and args.save_to:
        raise ValueError("Choose either save path or load path, not both.")
    train_and_evaluate(args.ds, args.model, args.load_from, args.save_to, args.eval_on)
    
    
    
    