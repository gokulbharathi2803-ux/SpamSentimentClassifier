from flask import Flask, render_template, request
import joblib
from pathlib import Path


# ==========================================
# FLASK APPLICATION
# ==========================================

app = Flask(__name__)


# ==========================================
# PROJECT FOLDER
# ==========================================

BASE_DIR = Path(__file__).resolve().parents[1]


# ==========================================
# LOAD SPAM MODEL
# ==========================================

spam_model = joblib.load(
    BASE_DIR / "models" / "spam_model.pkl"
)

spam_vectorizer = joblib.load(
    BASE_DIR / "models" / "spam_vectorizer.pkl"
)


# ==========================================
# LOAD SENTIMENT MODEL
# ==========================================

sentiment_model = joblib.load(
    BASE_DIR / "models" / "sentiment_model.pkl"
)

sentiment_vectorizer = joblib.load(
    BASE_DIR / "models" / "sentiment_vectorizer.pkl"
)


print("All ML models loaded successfully!")


# ==========================================
# HOME PAGE
# ==========================================

@app.route("/", methods=["GET", "POST"])
def home():

    result = None
    task = None
    text = ""

    if request.method == "POST":

        text = request.form.get("text", "")

        task = request.form.get("task", "")


        # ==================================
        # SPAM DETECTION
        # ==================================

        if task == "spam":

            if text.strip():

                features = spam_vectorizer.transform(
                    [text]
                )

                prediction = spam_model.predict(
                    features
                )[0]


                if prediction == 1:

                    result = "SPAM"

                else:

                    result = "NOT SPAM"


        # ==================================
        # SENTIMENT DETECTION
        # ==================================

        elif task == "sentiment":

            if text.strip():

                features = sentiment_vectorizer.transform(
                    [text]
                )

                result = sentiment_model.predict(
                    features
                )[0]


    return render_template(
        "index.html",
        result=result,
        task=task,
        text=text
    )


# ==========================================
# RUN APPLICATION
# ==========================================

if __name__ == "__main__":

    app.run(debug=True)