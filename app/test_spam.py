import joblib
from pathlib import Path

# Project folder
BASE_DIR = Path(__file__).resolve().parents[1]

# Load trained model and TF-IDF vectorizer
model = joblib.load(BASE_DIR / "models" / "spam_model.pkl")
vectorizer = joblib.load(BASE_DIR / "models" / "spam_vectorizer.pkl")

print("Spam classifier loaded successfully!")
print("Type 'exit' to stop.\n")

while True:
    message = input("Enter a message: ")

    if message.lower() == "exit":
        print("Program ended.")
        break

    message_tfidf = vectorizer.transform([message])
    prediction = model.predict(message_tfidf)[0]

    if prediction == 1:
        print("Result: SPAM\n")
    else:
        print("Result: NOT SPAM\n")