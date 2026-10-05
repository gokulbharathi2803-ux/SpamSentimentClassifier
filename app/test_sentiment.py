import joblib
from pathlib import Path


# Project folder
BASE_DIR = Path(__file__).resolve().parents[1]


# Load trained sentiment model
model = joblib.load(
    BASE_DIR / "models" / "sentiment_model.pkl"
)


# Load TF-IDF vectorizer
vectorizer = joblib.load(
    BASE_DIR / "models" / "sentiment_vectorizer.pkl"
)


print("================================")
print("SENTIMENT CLASSIFIER")
print("================================")

print("Model loaded successfully!")
print("Type 'exit' to stop.\n")


while True:

    text = input("Enter text: ")

    # Stop program
    if text.lower() == "exit":

        print("\nProgram ended.")
        break


    # Check empty input
    if not text.strip():

        print("Please enter some text.\n")
        continue


    # Convert text into TF-IDF features
    features = vectorizer.transform([text])


    # Predict sentiment
    prediction = model.predict(features)[0]


    # Display result
    print("Sentiment:", prediction)
    print()