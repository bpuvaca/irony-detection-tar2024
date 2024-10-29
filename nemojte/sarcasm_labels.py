from transformer_main import train_and_evaluate

datasets = ['isarcasm_sarc']
models = ['roberta', 'bertweet', 'bert']
params_suffix = ''

for model in models:
    for dataset in datasets:
        train_and_evaluate(dataset, model, None, f"{model}_{dataset}_{params_suffix}", None, False, False)