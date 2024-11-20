import argparse
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
import pandas as pd
from Loader import TransformerLoader
from torch.utils.data import DataLoader
import torch
import train
import evaluate
import os
import csv

def parse_args():
    parser = argparse.ArgumentParser(description='Process some arguments.')
    parser.add_argument('--ds', type=str, required=False, help='Name of the dataset')
    parser.add_argument('--eval_on', type=str, required=False, help='Dataset(s) to evaluate on')
    parser.add_argument('--model', type=str, required=True, help='Model name')
    parser.add_argument('--load_from', type=str, required=False, help='Source')
    parser.add_argument('--save_to', type=str, required=False, help='Destination')
    parser.add_argument('--cart', type=bool, required=False, help='Cartography', default=False)

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

def load_dataset(dataset_name, tokenizer, test_only=False):
    loader = TransformerLoader(dataset_name)
    if test_only:
        loader.load_test_dataset(tokenizer, remove_hashtags=True)
    else:
        loader.load_dataset(tokenizer, remove_hashtags=True)

    batch_size = 16

    # Create DataLoaders
    test_dataloader = DataLoader(loader.test_dataset, batch_size=128, shuffle=False)
    if test_only:
        return test_dataloader, loader.test_texts

    train_dataloader = DataLoader(loader.train_dataset, batch_size=batch_size, shuffle=True)
    valid_dataloader = DataLoader(loader.valid_dataset, batch_size=128, shuffle=False)
    return train_dataloader, valid_dataloader, test_dataloader, loader.test_texts

def train_and_evaluate(dataset, model_name, load_from=None, save_to=None, eval_on=None, return_wrong_preds=True, return_all_preds=True, cartography=False):
    print(f"Training {model_name} on {dataset}")
    transformer_model = map_model_name(model_name)
    model = load_model(transformer_model, load_from)
    tokenizer = AutoTokenizer.from_pretrained(transformer_model)

    if not load_from:
        train_dataloader, valid_dataloader, test_dataloader, tweets = load_dataset(dataset, tokenizer)
        save_path = save_to if save_to else None
        if cartography:
            train.train_cartography(model, train_dataloader, epochs=3, save_path=save_path, model_name=model_name, trained_on=dataset)
 
        else:
            train.train_transformer(model, train_dataloader, valid_dataloader, epochs=1, early_stopping=False, save_path=save_path)

    if not cartography:
        if not eval_on:
            evaluate.evaluate_transformer(model, test_dataloader, model_name=model_name, trained_on=dataset, eval_on=dataset, 
                                        return_wrong_preds=return_wrong_preds, return_all_preds=return_all_preds, dataset_texts=tweets)
        else:
            eval_on = eval_on.split(" ")
            for dataset in eval_on:
                _, _, test_dataloader, tweets = load_dataset(dataset, tokenizer)
                print(f"Evaluating on {dataset}")
                evaluate.evaluate_transformer(model, test_dataloader, model_name=model_name, trained_on=dataset, eval_on=dataset,
                                        return_wrong_preds=return_wrong_preds, return_all_preds=return_all_preds, dataset_texts=tweets)

def evaluate_only(model_name, load_from, eval_on=None, return_wrong_preds=True, return_all_preds=True):
    print(f"Evaluating {model_name}")
    transformer_model = map_model_name(model_name)
    model = load_model(transformer_model, load_from)
    tokenizer = AutoTokenizer.from_pretrained(transformer_model)

    eval_on = eval_on.split(" ")
    for dataset in eval_on:
        test_dataloader, tweets = load_dataset(dataset, tokenizer, test_only=True)
        print(f"Evaluating on {dataset}")
        evaluate.evaluate_transformer(model, test_dataloader, model_name=model_name, trained_on=dataset, eval_on=dataset,
                                return_wrong_preds=return_wrong_preds, return_all_preds=return_all_preds, dataset_texts=tweets, load_from=load_from)

def train_and_cross_validate(dataset, model_name, save_params=False, return_all_preds=True, folds=5, epochs=10):
    print(f"Training {model_name} on {dataset}")
    transformer_model = map_model_name(model_name)
    tokenizer = AutoTokenizer.from_pretrained(transformer_model)
    #iskoristi save_to

    loader = TransformerLoader(dataset)
    loader.load_crossval_dataset(tokenizer, remove_hashtags=True, k=folds)
    
    batch_size = 16
    
    f1_score = 0
    
    for i in range(folds):
        # Create DataLoaders
        print("\nFold: ", i)
        model = load_model(transformer_model, None)    
        train_dataloader = DataLoader(loader.train_datasets[i], batch_size=batch_size, shuffle=True)
        valid_dataloader = DataLoader(loader.valid_datasets[i], batch_size=128, shuffle=False)
        result = train.train_transformer(model, train_dataloader, valid_dataloader, epochs=epochs, early_stopping=False, return_all_preds=return_all_preds, dataset_texts=loader.test_texts[i], model_name=model_name, trained_on=dataset, cartography=True)
        if save_params:
            filepath = f"../params/crossval/{model_name}/{dataset}/"
            filename = f"{model_name}_{dataset}_fold_{i+1}.pt"
            os.makedirs(filepath, exist_ok=True)
            fullpath = filepath + filename
            torch.save(model.state_dict(), fullpath)
            print(f"Model parameters saved to {fullpath}")
    
        if return_all_preds:
            all_preds, (f1, acc, prec, rec) = result
            filepath = f'../preds/crossval/{model_name}/{dataset}/{dataset}/'
            filename = f'{model_name}_trained_on_{dataset}_evaluated_on_{dataset}_fold_{i+1}.csv' 
            os.makedirs(filepath, exist_ok=True)
            fullpath = filepath + filename
            with open(fullpath, "w", encoding="utf-8", newline='') as csvfile:
                csvwriter = csv.writer(csvfile)
                csvwriter.writerow(["index", "tweet", "label", "prediction"])
                for pred in all_preds:
                    csvwriter.writerow([pred[0], pred[1], pred[2], pred[3]])
                
            print(f"Check {filepath} for all preds")

        else:
            f1, acc, prec, rec = result
        f1_score += f1
            
    print(f"Average F1 score: {f1_score/folds}")


def cross_validate(dataset, model_name, trained_on, load_from, fold_test_dataset, return_all_preds=True, folds=5):
    print(f"Evaluating {model_name} trained on {trained_on} on {dataset}")
    transformer_model = map_model_name(model_name)
    tokenizer = AutoTokenizer.from_pretrained(transformer_model)
    #iskoristi save_to

    loader = TransformerLoader(dataset)
    # if fold_test_dataset:
    loader.load_crossval_dataset(tokenizer, remove_hashtags=True, k=folds)
    # else:
    #     loader.load_test_dataset(tokenizer, remove_hashtags=True)
    
    batch_size = 16
    
    f1_score = 0
    
    for i in range(folds):
        # Create DataLoaders
        print("\nFold: ", i)
        #f"../params/{load_from}.pt"
        model = load_model(transformer_model, f"{load_from}/{model_name}_{trained_on}_fold_{i+1}")
        # if fold_test_dataset:
        valid_dataloader = DataLoader(loader.valid_datasets[i], batch_size=128, shuffle=False)
        dataset_texts=loader.test_texts[i]
        # else:
        #     valid_dataloader = DataLoader(loader.test_dataset, batch_size=128, shuffle=False)
        #     dataset_texts=loader.test_texts
        result = evaluate.evaluate_transformer(model, valid_dataloader, model_name=model_name, trained_on=trained_on, eval_on=dataset, return_all_preds=return_all_preds, dataset_texts=dataset_texts)
        
        if return_all_preds:
            all_preds, (f1, acc, prec, rec) = result
            filepath = f'../preds/crossval/{model_name}/{trained_on}/{dataset}/'
            filename = f'{model_name}_trained_on_{trained_on}_evaluated_on_{dataset}_fold_{i+1}.csv' 
            os.makedirs(filepath, exist_ok=True)
            fullpath = filepath + filename
            with open(fullpath, "w", encoding="utf-8", newline='') as csvfile:
                csvwriter = csv.writer(csvfile)
                csvwriter.writerow(["index", "tweet", "label", "prediction"])
                for pred in all_preds:
                    csvwriter.writerow([pred[0], pred[1], pred[2], pred[3]])
                
            print(f"Check {filepath} for all preds")

        else:
            f1, acc, prec, rec = result
        f1_score += f1
            
    print(f"Average F1 score: {f1_score/folds}")
    
        


    
if __name__ == "__main__":
    args = parse_args()
    if args.load_from and args.save_to:
        raise ValueError("Choose either save path or load path, not both.")
    train_and_evaluate(args.ds, args.model, args.load_from, args.save_to, args.eval_on, True, True, args.cart)
    
    
    
    