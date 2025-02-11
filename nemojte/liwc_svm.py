import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score

df = pd.read_csv("../datasets/LIWC/liwc_sarcasm_parsed_filtered.csv")
df = df.drop(columns=["tweet"])

irony = pd.read_csv('../results/LIWC/datasets/liwc_irony_parsed.csv', sep=',').drop(columns=['Analytic','Clout','Authentic','Tone'])
sarcasm = pd.read_csv('../results/LIWC/datasets/liwc_sarcasm_parsed.csv', sep=',').drop(columns=['Analytic','Clout','Authentic','Tone'])

for name, df in zip(['irony', 'sarcasm'], [irony, sarcasm]):

    X = df.iloc[:, 1:].values
    y = df.iloc[:, 0].values 

    imputer = SimpleImputer(strategy="mean")
    X = imputer.fit_transform(X)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    svm = SVC(kernel="rbf", C=1.0, gamma="scale")
    svm.fit(X_train, y_train)

    accuracy = svm.score(X_test, y_test)
    print(f"{name} SVM")
    print(f"Test Accuracy: {accuracy:.2f}")
    y_pred = svm.predict(X_test)
    f1 = f1_score(y_test, y_pred, average="weighted")
    print(f"F1 Score: {f1:.2f}")
