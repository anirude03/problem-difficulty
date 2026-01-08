# Predicting Programming Problem Difficulty

An intelligent NLP + Machine Learning system that automatically predicts the difficulty class and difficulty score of programming problems using only textual descriptions.

This project mimics how platforms like Codeforces, CodeChef, and Kattis assess problem difficulty — but without human intervention.

# Repository Structure
```
problem_difficulty_prediction/
│
├── data/
│   ├── problem_data.csv
│   ├── preprocessed_data.csv
│   ├── inputs.pkl
│   ├── y_class.pkl
│   ├── y_score.pkl
│       
│
├── notebooks/
│   ├── data_preprocessing.ipynb
│   ├── feature_engineering.ipynb
│   ├── classification.ipynb
│   └── regression.ipynb
│
├── models/
│   ├── classifier.pkl
│   ├── regressor.pkl
│   ├── tfidf_vectorizer.pkl
│   └── numeric_scaler.pkl
│
├── app/
│   ├──app.py
|   ├── templates
│   ├── models
│   |     ├── tfidf_vectorizer.pkl
│   |     ├── numeric_scaler.pkl
│   |     ├── classifier.pkl
|   |     ├── regressor.pkl
|   ├── requirements.txt
|
└── README.md  

```
# Setup and Insatllation
```
Clone the Repository
git clone https://github.com/<your-username>/problem-difficulty-prediction.git
cd problem-difficulty-prediction

# Download the  models directory from the Google drive
        https://drive.google.com/drive/folders/1ytCWJNTxS5la3e_qygF6Aq2SHvAPSasJ?usp=drive_link
   
Add the downloaded models folder  to this directory 
 and Also add the models to the app directory

 Now it is ready to run .

Install Dependencies
pip install -r requirements.txt

If requirements.txt is not available, install manually:
pip install numpy pandas scikit-learn scipy matplotlib seaborn flask joblib nltk


Prepare the Dataset

Place the dataset file inside the data/ directory.

Ensure the dataset contains the following columns:
title
description
input_description
output_description
problem_class
problem_score


Run the Notebooks (In Order)

notebooks/data_preprocessing.ipynb
notebooks/feature_engineering.ipynb
notebooks/classification.ipynb
notebooks/regression.ipynb


Verify Saved Artifacts

models/
├── classifier.pkl
├── regressor.pkl
├── tfidf_vectorizer.pkl
└── numeric_scaler.pkl

data/
├── problem_data.csv
├── preprocessed_data.csv
├── inputs.pkl
├── y_class.pkl
├── y_score.pkl


Run the Web Application
python app/app.py

If it does not redirect automatically,
open your browser and navigate to:
http://127.0.0.1:5000
```

# Project Overview

Online coding platforms usually assign:

Difficulty Class → Easy / Medium / Hard

Difficulty Score → Numerical rating

These labels are often based on human judgment and user feedback.
This project builds an automated ML-based solution that predicts both using only text data, such as:

Problem description

Input description

Output description

# Objectives

By the end of this project, the system can:

Predict difficulty class (classification)

Predict difficulty score (regression)

Use only textual information

Serve predictions via a simple web interface


# Dataset Description

| Column               | Description                |
| -------------------- | -------------------------- |
| `title`              | Problem title              |
| `description`        | Problem statement          |
| `input_description`  | Input format               |
| `output_description` | Output format              |
| `problem_class`      | Easy / Medium / Hard       |
| `problem_score`      | Numerical difficulty score |

Methodology
🔹 1. Text Preprocessing

Combine all text fields into a single document

Handle missing values

Lowercasing and normalization

Preserve mathematical symbols (+ - * / % < >) for complexity signal

🔹 2. Feature Engineering
  TF-IDF Features

Unigrams, bigrams, trigrams

Sublinear TF scaling

Vocabulary pruning

Structural & Linguistic Features

Word count

Line count

Digit count

Constraint density

Operator diversity

Big-O notation count

Algorithm family indicators (DP, Graph, Tree, Greedy, etc.)

Conditional logic density

These features significantly improve performance, especially for Medium vs Hard separation.

3. Feature Scaling

Numeric features scaled using StandardScaler(with_mean=False)

TF-IDF features left untouched (already normalized)

Models Used
🟦 Classification

Linear Support Vector Machine (LinearSVC)
Logistics regression
Random Forest classifier

Class weights enabled to handle imbalance

Metrics

Accuracy

Precision, Recall, F1-score

Confusion Matrix

🟩 Regression

Gradient Boosting Regressor
Random Foresr Regressor

Metrics

Mean Absolute Error (MAE)

Root Mean Squared Error (RMSE)

# Web Interface

A simple Flask web application for users to:

Paste a new problem description

Provide input & output format

Instantly receive:

Predicted difficulty class

Predicted difficulty score

```
Author

Anirudh Kumar Verma
 Machine Learning | Natural Language Processing  | Data Science

License

This project is for academic and research purposes.
```





