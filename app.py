import os
import nltk
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pickle
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string

# Ensure NLTK data is downloaded on startup
nltk.download('punkt')
nltk.download('stopwords')

app = Flask(__name__)
CORS(app)

# Load all models and vectorizers
with open("models/corona_model.pkl", "rb") as f:
    corona_model = pickle.load(f)
with open("models/corona_vectorizer.pkl", "rb") as f:
    corona_vectorizer = pickle.load(f)

with open("models/sms_model.pkl", "rb") as f:
    sms_model = pickle.load(f)
with open("models/sms_vectorizer.pkl", "rb") as f:
    sms_vectorizer = pickle.load(f)

with open("models/fake_model.pkl", "rb") as f:
    fake_model = pickle.load(f)
with open("models/fake_vectorizer.pkl", "rb") as f:
    fake_vectorizer = pickle.load(f)

with open("models/amazon_model.pkl", "rb") as f:
    amazon_model = pickle.load(f)
with open("models/amazon_vectorizer.pkl", "rb") as f:
    amazon_vectorizer = pickle.load(f)

with open("models/news_model.pkl", "rb") as f:
    news_model = pickle.load(f)
with open("models/news_vectorizer.pkl", "rb") as f:
    news_vectorizer = pickle.load(f)

# Preprocessing function
def preprocess(text):
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in string.punctuation]
    tokens = [t for t in tokens if t not in stopwords.words("english")]
    return " ".join(tokens)

# Task-based prediction
def predict_task(text, task):
    cleaned = preprocess(text)

    if task == "corona":
        vec = corona_vectorizer.transform([cleaned])
        return corona_model.predict(vec)[0]

    elif task == "sms":
        vec = sms_vectorizer.transform([cleaned])
        return sms_model.predict(vec)[0]

    elif task == "fake":
        vec = fake_vectorizer.transform([cleaned])
        return fake_model.predict(vec)[0]

    elif task == "amazon":
        vec = amazon_vectorizer.transform([cleaned])
        return amazon_model.predict(vec)[0]

    elif task == "news":
        vec = news_vectorizer.transform([cleaned])
        return news_model.predict(vec)[0]

    else:
        return "Invalid task."

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        text = data.get("text")
        task = data.get("task")

        if not text or not task:
            return jsonify({"error": "Missing text or task"}), 400

        result = predict_task(text, task)
        return jsonify({"result": result})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
