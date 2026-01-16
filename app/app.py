from flask import Flask, render_template, request
import joblib
import numpy as np
import re
from scipy.sparse import hstack
from feature_config import NUMERIC_FEATURE_NAMES

# -----------------------------
# Load models & preprocessors
# -----------------------------
BASE_MODEL_PATH = "/models"

classifier = joblib.load(f"{BASE_MODEL_PATH}/classifier.pkl")
regressor = joblib.load(f"{BASE_MODEL_PATH}/regressor.pkl")
tfidf = joblib.load(f"{BASE_MODEL_PATH}/tfidf_vectorizer.pkl")
scaler = joblib.load(f"{BASE_MODEL_PATH}/numeric_scaler.pkl")

# -----------------------------
# Flask app
# -----------------------------
app = Flask(__name__)

# -----------------------------
# Text cleaning
# -----------------------------


def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9+\-*/=%<>\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# -----------------------------
# Feature engineering
# -----------------------------


def extract_features(full_text, clean_txt):
    features = {}

    # -----------------------
    # Base numeric features
    # -----------------------
    features["text_length"] = len(clean_txt)
    features["word_count"] = len(clean_txt.split())
    features["digit_count"] = sum(c.isdigit() for c in full_text)
    features["line_count"] = full_text.count("\n") + 1
    features["math_symbol_count"] = sum(c in "+-*/=%<>" for c in full_text)

    features["constraint_density"] = features["digit_count"] / \
        (features["word_count"] + 1)

    operators = "+-*/=%<>^"
    features["operator_diversity"] = len(
        set(c for c in full_text if c in operators))

    features["big_o_count"] = len(re.findall(r"O\s*\(", full_text))
    features["large_number_count"] = len(re.findall(r"\b10\^\d+\b", full_text))

    words = clean_txt.split()
    features["unique_word_ratio"] = len(set(words)) / (len(words) + 1)

    features["conditional_count"] = (
        clean_txt.count("if") +
        clean_txt.count("else") +
        clean_txt.count("while")
    )

    features["multi_input"] = int(len(words) > 50)
    features["array_matrix_flag"] = int(
        "array" in clean_txt or "matrix" in clean_txt or "[" in full_text)

    # -----------------------
    # Keyword counts
    # -----------------------
    features["kw_dp"] = clean_txt.count("dp")
    features["kw_dynamic_programming"] = clean_txt.count("dynamic programming")
    features["kw_graph"] = clean_txt.count("graph")
    features["kw_tree"] = clean_txt.count("tree")
    features["kw_recursion"] = clean_txt.count("recursion")
    features["kw_greedy"] = clean_txt.count("greedy")
    features["kw_binary"] = clean_txt.count("binary")
    features["kw_dfs"] = clean_txt.count("dfs")
    features["kw_bfs"] = clean_txt.count("bfs")

    # -----------------------
    # Algorithm flags
    # -----------------------
    features["algo_dp"] = int(
        "dp" in clean_txt or "dynamic programming" in clean_txt)
    features["algo_graph"] = int("graph" in clean_txt)
    features["algo_tree"] = int("tree" in clean_txt)
    features["algo_greedy"] = int("greedy" in clean_txt)
    features["algo_math"] = int("mod" in clean_txt or "gcd" in clean_txt)
    features["algo_bit"] = int("bit" in clean_txt or "xor" in clean_txt)
    features["algo_string"] = int("string" in clean_txt)

    # -----------------------
    # FINAL: strict ordering
    # -----------------------
    numeric_array = np.array(
        [features[name] for name in NUMERIC_FEATURE_NAMES],
        dtype=float
    ).reshape(1, -1)

    return numeric_array


# -----------------------------
# Routes
# -----------------------------
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    description = request.form.get("description", "")
    input_desc = request.form.get("input_description", "")
    output_desc = request.form.get("output_description", "")

    if not description and not input_desc and not output_desc:
        return render_template(
            "result.html",
            error="No Input Provided"
        )
    # Combine text
    full_text = f"{description} {input_desc} {output_desc}"
    full_text = str(full_text)

    clean_txt = clean_text(full_text)

    # TF-IDF
    text_vec = tfidf.transform([clean_txt])

    # Numeric features
    numeric_feats = extract_features(full_text, clean_txt)
    numeric_feats_scaled = scaler.transform(numeric_feats)

    # Final feature vector
    X = hstack([text_vec, numeric_feats_scaled])

    # Predictions
    difficulty_class = classifier.predict(X)[0]
    difficulty_score = regressor.predict(X)[0]

    return render_template(
        "result.html",
        predicted_class=difficulty_class,
        predicted_score=round(difficulty_score, 1)
    )


# -----------------------------
# Run app
# -----------------------------
if __name__ == "__main__":
    app.run(debug=True)
