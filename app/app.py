from flask import Flask, render_template, request
import joblib
import numpy as np
import re
from scipy.sparse import hstack


# App Initialization

app = Flask(__name__)


# Load Models
BASE_MODEL_PATH = "models"

clf = joblib.load(f"{BASE_MODEL_PATH}/classifier.pkl")
reg = joblib.load(f"{BASE_MODEL_PATH}/regressor.pkl")
tfidf = joblib.load(f"{BASE_MODEL_PATH}/tfidf_vectorizer.pkl")
scaler = joblib.load(f"{BASE_MODEL_PATH}/numeric_scaler.pkl")




def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9+\-*/=%<>\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()



def extract_features(text):
    features = []

    features.append(len(text)) 
    features.append(sum(c in "+-*/=%<>" for c in text))  

    keywords = [
        "dp", "dynamic programming",
        "graph", "tree",
        "recursion", "greedy",
        "binary", "dfs", "bfs"
    ]

    for kw in keywords:
        features.append(text.count(kw))

    return np.array(features).reshape(1, -1)




@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    desc = request.form.get("description", "").strip()
    inp = request.form.get("input_description", "").strip()
    out = request.form.get("output_description", "").strip()

    # If ALL inputs are empty
    if not desc and not inp and not out:
        return render_template(
            "result.html",
            error="No input provided"
        )

    full_text = f"{desc} {inp} {out}"
    clean = clean_text(full_text)

    X_tfidf = tfidf.transform([clean])
    X_extra = extract_features(clean)
    X_extra_scaled = scaler.transform(X_extra)

    X = hstack([X_tfidf, X_extra_scaled])

    difficulty = clf.predict(X)[0]
    score = reg.predict(X)[0]

    return render_template(
        "result.html",
        difficulty=difficulty,
        score=round(float(score), 2)
    )



# Run App

if __name__ == "__main__":
    app.run(debug=True)
