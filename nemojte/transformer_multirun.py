from transformer_main import train_and_evaluate

datasets = ['irony', 'sarcasm', 'mix']
models = ['roberta', 'bertweet', 'bert']
params_suffix = ''

for i in range(2,6):
    print("======================================= RUN", i, " =======================================")
    for model in models:
        for dataset in datasets:
            train_and_evaluate(dataset, model, None, f"{model}_{dataset}_{i}{params_suffix}", eval_on='irony sarcasm mix')