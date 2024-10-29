import Loader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
from sklearn.metrics import classification_report

def create_feature_vectors(vectorizer, corpus, fit=False):
    if fit:
        # Fit and transform for training set
        feature_vectors = vectorizer.fit_transform(corpus)
    else:
        # Only transform for validation and test sets
        feature_vectors = vectorizer.transform(corpus)
    return feature_vectors

if __name__ == '__main__':
    irony_train_file_path = '../datasets/irony/irony_train.csv'
    irony_test_file_path = '../datasets/irony/irony_test.csv'

    sarcasm_train_file_path = '../datasets/sarcasm/sarcasm_train.csv'
    sarcasm_test_file_path = '../datasets/sarcasm/sarcasm_test.csv'

    mix_train_file_path = '../datasets/mix/mix_train.csv'
    mix_test_file_path = '../datasets/mix/mix_test.csv'

    train_paths = {"irony": irony_train_file_path, "sarcasm": sarcasm_train_file_path, "mix": mix_train_file_path}
    test_paths = {"irony": irony_test_file_path, "sarcasm": sarcasm_test_file_path, "mix": mix_test_file_path}

    keys = list(train_paths.keys())

    results = {"irony": {"irony": [], "sarcasm": [], "mix": []},
               "sarcasm": {"irony": [], "sarcasm": [], "mix": []},
                "mix": {"irony": [], "sarcasm": [], "mix": []}}

    for _ in range(5):
        for train_set in keys:
            for test_set in keys:

                train_file_path = train_paths[train_set]
                test_file_path = test_paths[test_set]

                corpus, labels = Loader.parse_dataset(train_file_path, balance=True, remove_hashtags=True)
                test_corpus, test_labels = Loader.parse_dataset(test_file_path, balance=False, remove_hashtags=True)

                vectorizer = TfidfVectorizer(ngram_range=(2, 2))

                train_feature_vectors = create_feature_vectors(vectorizer, corpus, fit=True)
                test_feature_vectors = create_feature_vectors(vectorizer, test_corpus)

                classifier = svm.SVC(kernel='linear')
                classifier.fit(train_feature_vectors, labels)

                test_predictions = classifier.predict(test_feature_vectors)
                report = classification_report(test_labels, test_predictions, output_dict=True)
                f1_score = round(report['weighted avg']['f1-score'], 3)
                print(f"Train Set: {train_set}, Test Set: {test_set}, Test Set F1 Score: {f1_score}")

                results[train_set][test_set].append(f1_score)

    for train_set in keys:
        for test_set in keys:
            scores = results[train_set][test_set]
            avg_score = sum(scores) / len(scores)
            std_dev = (sum((x - avg_score) ** 2 for x in scores) / len(scores)) ** 0.5
            print(f"Train Set: {train_set}, Test Set: {test_set}, Average F1 Score: {avg_score:.3f}, Standard Deviation: {std_dev:.3f}")


    
    
    