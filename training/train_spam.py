import pandas as pd
import joblib

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# Project folder
BASE_DIR = Path(__file__).resolve().parents[1]

# Load dataset
data = pd.read_csv(BASE_DIR / "data" / "spam.csv")

print("Dataset loaded successfully!")
print("Total messages:", len(data))
print("Columns:", list(data.columns))


# Input and output
X = data["text"]
y = data["label"]


# Convert labels
y = y.map({
    "ham": 0,
    "spam": 1
})


# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# Convert text to TF-IDF
vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english",
    max_features=5000
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

print("TF-IDF conversion completed!")


# ML models
models = {
    "Naive Bayes": MultinomialNB(),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "SVM": LinearSVC()
}


# Train and evaluate
results = {}

for name, model in models.items():

    print("\nTraining:", name)

    model.fit(X_train_tfidf, y_train)

    predictions = model.predict(X_test_tfidf)

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)

    results[name] = f1

    print("Accuracy :", round(accuracy, 4))
    print("Precision:", round(precision, 4))
    print("Recall   :", round(recall, 4))
    print("F1 Score :", round(f1, 4))


# Find best model
best_model_name = max(results, key=results.get)
best_model = models[best_model_name]

print("\n==============================")
print("BEST MODEL:", best_model_name)
print("==============================")


# Save model and vectorizer
joblib.dump(
    best_model,
    BASE_DIR / "models" / "spam_model.pkl"
)

joblib.dump(
    vectorizer,
    BASE_DIR / "models" / "spam_vectorizer.pkl"
)

print("Spam model saved successfully!")