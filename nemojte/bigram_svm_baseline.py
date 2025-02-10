import numpy as np
import Loader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
from sklearn.metrics import classification_report

path_dict = Loader.file_path_dict

def create_feature_vectors(vectorizer, corpus, fit=False):
    if fit:
        # Fit and transform for training set
        feature_vectors = vectorizer.fit_transform(corpus)
    else:
        # Only transform for validation and test sets
        feature_vectors = vectorizer.transform(corpus)
    return feature_vectors

if __name__ == '__main__':
    
    datasets = ["irony", "sarcasm", "semeval_mix", "isarcasm_mix", "mix"]
    datasets = ["train_" + dataset for dataset in datasets]

    for train_set in datasets:
        for test_set in datasets:
            f1s = []
            train_file_path = path_dict[train_set]
            test_file_path = path_dict[test_set]

            corpus, labels = Loader.parse_dataset(train_file_path, balance=False, remove_hashtags=True)
            test_corpus, test_labels = Loader.parse_dataset(test_file_path, balance=False, remove_hashtags=True)

            train_size = len(corpus)
            test_size = len(test_corpus)
            
            for i in range(5):
                test_start = i * (test_size // 5)
                test_end = (i + 1) * (test_size // 5)
                train_start = i * (train_size // 5)
                train_end = (i + 1) * (train_size // 5)
                
                val_corpus = test_corpus[test_start:test_end]
                val_labels = test_labels[test_start:test_end]
                
                train_corpus = corpus[:train_start] + corpus[train_end:]
                train_labels = labels[:train_start] + labels[train_end:]
                
                vectorizer = TfidfVectorizer(ngram_range=(2, 2))

                train_feature_vectors = create_feature_vectors(vectorizer, train_corpus, fit=True)
                val_feature_vectors = create_feature_vectors(vectorizer, val_corpus)

                classifier = svm.SVC(kernel='linear')
                classifier.fit(train_feature_vectors, train_labels)

                val_predictions = classifier.predict(val_feature_vectors)
                report = classification_report(val_labels, val_predictions, output_dict=True)
                f1_score = round(report['weighted avg']['f1-score'], 3)
                f1s.append(f1_score)
            
            print(f"Train Set: {train_set}, Test Set: {test_set}, Test Set avg F1 Score: {np.mean(f1s)}, Std Dev: {np.std(f1s)}")


    
    
    