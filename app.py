
from flask import Flask, request, render_template
from flask_cors import CORS
import re
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Ensure NLTK stopwords are available
nltk.data.path.append('nltk_data')
stop_words = set(stopwords.words('english'))

# Preprocessing functions
def preprocess(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", '', text)
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word.isalpha()]
    return ' '.join([word for word in tokens if word not in stop_words])

# Load models and vectorizers
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

# Mapping dictionaries
amazon_label_map = {'1': 'Negative', '2': 'Positive'}
news_class_map = {
    1: 'Social Issues / Law',
    2: 'Sports',
    3: 'Finance / Economy',
    4: 'Science & Technology'
}

# Prediction routing logic
def predict_task(text, task):
    cleaned = preprocess(text)

    if task == "corona_sentiment":
        vec = corona_vectorizer.transform([cleaned])
        pred = corona_model.predict(vec)[0]
        return f"[Corona Sentiment] → {pred}"

    elif task == "sms_spam":
        vec = sms_vectorizer.transform([cleaned])
        pred = sms_model.predict(vec)[0]
        return "[Spam Detection] → SPAM" if pred == 1 else "[Spam Detection] → HAM (Not Spam)"

    elif task == "fake_news":
        vec = fake_vectorizer.transform([cleaned])
        pred = fake_model.predict(vec)[0]
        return "[Fake News Detection] → FAKE" if pred == 1 else "[Fake News Detection] → REAL"

    elif task == "amazon_sentiment":
        vec = amazon_vectorizer.transform([cleaned])
        pred = amazon_model.predict(vec)[0]
        return f"[Amazon Review Sentiment] → {amazon_label_map.get(pred, 'Unknown')}"

    elif task == "news_category":
        vec = news_vectorizer.transform([cleaned])
        pred = news_model.predict(vec)[0]
        return f"[News Classification] → {news_class_map.get(pred, 'Unknown')}"

    return "Invalid task."

# Flask app setup
app = Flask(__name__)
CORS(app)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    text = request.form.get('text')
    task = request.form.get('task')
    result = predict_task(text, task)
    return render_template('index.html', prediction=result, input_text=text)

if __name__ == "__main__":
    app.run(debug=False)
