import pickle
import string
import numpy as np

from flask import Flask, render_template, request, jsonify

import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer


app = Flask(__name__)

cv = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

ps = PorterStemmer()


def process_text(text):
    text = text.lower()
    tokens = nltk.word_tokenize(text)
    tokens = [token for token in tokens if token.isalnum()]

    stopwords = set(nltk.corpus.stopwords.words('english'))
    tokens = [token for token in tokens if token not in stopwords and token not in string.punctuation]

    tokens = [ps.stem(token) for token in tokens]

    return ' '.join(tokens)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/classify', methods=['POST'])
def classify():
    text = request.form['text']
    text = process_text(text)
    vectorized_text = cv.transform([text])
    prediction = model.predict(vectorized_text)[0]

    result = int(prediction)
    print(result)
    return jsonify(result=result)

if __name__ == '__main__':
    app.run()
