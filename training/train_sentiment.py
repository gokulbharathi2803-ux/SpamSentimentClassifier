import pandas as pd
import joblib

from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)


# ==========================================
# PROJECT FOLDER
# ==========================================

BASE_DIR = Path(__file__).resolve().parents[1]


# ==========================================
# LOAD DATASETS
# ==========================================

train_data = pd.read_csv(
    BASE_DIR / "data" / "train_all.csv"
)

val_data = pd.read_csv(
    BASE_DIR / "data" / "val_all.csv"
)

test_data = pd.read_csv(
    BASE_DIR / "data" / "test_all.csv"
)


print("Sentiment datasets loaded successfully!")

print("Training rows   :", len(train_data))
print("Validation rows :", len(val_data))
print("Testing rows    :", len(test_data))


# ==========================================
# INPUT AND OUTPUT
# ==========================================

X_train = train_data["sentence"]
y_train = train_data["label"]

X_val = val_data["sentence"]
y_val = val_data["label"]

X_test = test_data["sentence"]
y_test = test_data["label"]


# ==========================================
# WORD TF-IDF
# ==========================================

word_vectorizer = TfidfVectorizer(
    lowercase=True,
    ngram_range=(1, 2),
    min_df=2,
    max_features=20000,
    sublinear_tf=True
)


# ==========================================
# CHARACTER TF-IDF
# ==========================================

char_vectorizer = TfidfVectorizer(
    analyzer="char_wb",
    ngram_range=(3, 5),
    min_df=2,
    max_features=10000,
    sublinear_tf=True
)


# ==========================================
# COMBINE WORD + CHARACTER FEATURES
# ==========================================

vectorizer = FeatureUnion([
    ("word", word_vectorizer),
    ("char", char_vectorizer)
])


print("\nCreating TF-IDF features...")


# Training features
X_train_tfidf = vectorizer.fit_transform(
    X_train
)


# Validation features
X_val_tfidf = vectorizer.transform(
    X_val
)


# Test features
X_test_tfidf = vectorizer.transform(
    X_test
)


print("TF-IDF conversion completed!")

print(
    "Number of features:",
    X_train_tfidf.shape[1]
)


# ==========================================
# MACHINE LEARNING MODELS
# ==========================================

models = {

    "Naive Bayes": MultinomialNB(),

    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        C=2.0
    ),

    "SVM": LinearSVC(
        class_weight="balanced",
        C=2.0
    )
}


# ==========================================
# VALIDATION
# ==========================================

results = {}


for name, model in models.items():

    print("\n================================")
    print("Training:", name)
    print("================================")


    # Train model
    model.fit(
        X_train_tfidf,
        y_train
    )


    # Predict validation data
    predictions = model.predict(
        X_val_tfidf
    )


    # Calculate metrics
    accuracy = accuracy_score(
        y_val,
        predictions
    )


    precision = precision_score(
        y_val,
        predictions,
        average="weighted"
    )


    recall = recall_score(
        y_val,
        predictions,
        average="weighted"
    )


    f1 = f1_score(
        y_val,
        predictions,
        average="weighted"
    )


    # Store F1 score
    results[name] = f1


    # Display results
    print("Accuracy :", round(accuracy, 4))
    print("Precision:", round(precision, 4))
    print("Recall   :", round(recall, 4))
    print("F1 Score :", round(f1, 4))


# ==========================================
# FIND BEST MODEL
# ==========================================

best_model_name = max(
    results,
    key=results.get
)


print("\n================================")
print("BEST MODEL:", best_model_name)
print("================================")


# ==========================================
# COMBINE TRAINING + VALIDATION DATA
# ==========================================

X_train_full = pd.concat(
    [X_train, X_val],
    ignore_index=True
)


y_train_full = pd.concat(
    [y_train, y_val],
    ignore_index=True
)


print("\nTraining data combined.")

print(
    "Total training samples:",
    len(X_train_full)
)


# ==========================================
# CREATE FINAL WORD TF-IDF
# ==========================================

final_word_vectorizer = TfidfVectorizer(
    lowercase=True,
    ngram_range=(1, 2),
    min_df=2,
    max_features=20000,
    sublinear_tf=True
)


# ==========================================
# CREATE FINAL CHARACTER TF-IDF
# ==========================================

final_char_vectorizer = TfidfVectorizer(
    analyzer="char_wb",
    ngram_range=(3, 5),
    min_df=2,
    max_features=10000,
    sublinear_tf=True
)


# ==========================================
# FINAL COMBINED VECTORIZER
# ==========================================

final_vectorizer = FeatureUnion([
    ("word", final_word_vectorizer),
    ("char", final_char_vectorizer)
])


print("\nCreating final TF-IDF features...")


# Fit on complete training data
X_train_full_tfidf = final_vectorizer.fit_transform(
    X_train_full
)


# Transform test data
X_test_final_tfidf = final_vectorizer.transform(
    X_test
)


print("Final TF-IDF conversion completed!")


# ==========================================
# CREATE BEST MODEL
# ==========================================

if best_model_name == "Naive Bayes":

    best_model = MultinomialNB()


elif best_model_name == "Logistic Regression":

    best_model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        C=2.0
    )


else:

    best_model = LinearSVC(
        class_weight="balanced",
        C=2.0
    )


# ==========================================
# FINAL MODEL TRAINING
# ==========================================

print("\nTraining final model...")


best_model.fit(
    X_train_full_tfidf,
    y_train_full
)


print("Final model training completed!")


# ==========================================
# FINAL TEST
# ==========================================

test_predictions = best_model.predict(
    X_test_final_tfidf
)


# ==========================================
# FINAL METRICS
# ==========================================

test_accuracy = accuracy_score(
    y_test,
    test_predictions
)


test_precision = precision_score(
    y_test,
    test_predictions,
    average="weighted"
)


test_recall = recall_score(
    y_test,
    test_predictions,
    average="weighted"
)


test_f1 = f1_score(
    y_test,
    test_predictions,
    average="weighted"
)


# ==========================================
# DISPLAY FINAL RESULTS
# ==========================================

print("\n================================")
print("FINAL TEST RESULTS")
print("================================")


print(
    "Accuracy :",
    round(test_accuracy, 4)
)


print(
    "Precision:",
    round(test_precision, 4)
)


print(
    "Recall   :",
    round(test_recall, 4)
)


print(
    "F1 Score :",
    round(test_f1, 4)
)


# ==========================================
# CLASSIFICATION REPORT
# ==========================================

print("\n================================")
print("CLASSIFICATION REPORT")
print("================================")


print(
    classification_report(
        y_test,
        test_predictions
    )
)


# ==========================================
# SAVE MODEL
# ==========================================

joblib.dump(
    best_model,
    BASE_DIR / "models" / "sentiment_model.pkl"
)


# ==========================================
# SAVE VECTORIZER
# ==========================================

joblib.dump(
    final_vectorizer,
    BASE_DIR / "models" / "sentiment_vectorizer.pkl"
)


# ==========================================
# FINISHED
# ==========================================

print("\n================================")
print("MODEL SAVED SUCCESSFULLY!")
print("================================")


print(
    "sentiment_model.pkl"
)


print(
    "sentiment_vectorizer.pkl"
)


print("\nSentiment training completed successfully!")