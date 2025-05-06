
from flask import Flask, render_template, request, jsonify
import joblib
import os
import re
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

app = Flask(__name__)

# Load models and vectorizers
models = {
    'amazon_sentiment': joblib.load('models/amazon_model.pkl'),
    'corona_sentiment': joblib.load('models/corona_model.pkl'),
    'fake_news': joblib.load('models/fake_model.pkl'),
    'news_category': joblib.load('models/news_model.pkl'),
    'sms_spam': joblib.load('models/sms_model.pkl'),
}

vectorizers = {
    'amazon_sentiment': joblib.load('models/amazon_vectorizer.pkl'),
    'corona_sentiment': joblib.load('models/corona_vectorizer.pkl'),
    'fake_news': joblib.load('models/fake_vectorizer.pkl'),
    'news_category': joblib.load('models/news_vectorizer.pkl'),
    'sms_spam': joblib.load('models/sms_vectorizer.pkl'),
}

stop_words = set(stopwords.words('english'))

def preprocess(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"\d+", "", text)
    tokens = word_tokenize(text)
    tokens = [w for w in tokens if w.isalpha()]
    tokens = [w for w in tokens if w not in stop_words]
    return ' '.join(tokens)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    task = data.get("task")
    text = data.get("text")

    if not task or not text:
        return jsonify({"error": "Missing task or text"}), 400

    if task not in models or task not in vectorizers:
        return jsonify({"error": "Invalid task"}), 400

    processed = preprocess(text)
    vec = vectorizers[task].transform([processed])
    pred = models[task].predict(vec)[0]

    label_map = {
        'amazon_sentiment': {'1': 'Negative', '2': 'Positive'},
        'sms_spam': {0: 'HAM (Not Spam)', 1: 'SPAM'},
        'fake_news': {0: 'REAL', 1: 'FAKE'},
        'news_category': {
            1: 'Social Issues / Politics',
            2: 'Sports',
            3: 'Finance / Economy',
            4: 'Science & Technology'
        }
    }

    if task in label_map:
        pred = label_map[task].get(pred, str(pred))

    return jsonify({"result": pred})

if __name__ == "__main__":
    app.run(debug=True)
