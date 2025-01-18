from flask import Flask, request, render_template
import joblib
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import string

app = Flask(__name__)

vectorizer = joblib.load('vectorizer.joblib')
clf = joblib.load('model.joblib')

# Load model and vectorizer globally
# @app.before_first_request
# def load_model():
#     global vectorizer, clf


# Preprocessing function
def preprocess_text(text):
    stemmer = PorterStemmer()
    stopwords_set = set(stopwords.words('english'))
    # Convert text to lowercase
    text = text.lower()
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Tokenize and remove stopwords
    text = [stemmer.stem(word) for word in text.split() if word not in stopwords_set]
    return ' '.join(text)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    email_to_classify = request.form['email']
    email_to_classify = preprocess_text(email_to_classify)
    email_corpus = [email_to_classify]
    X_email = vectorizer.transform(email_corpus)
    prediction = clf.predict(X_email)
    result = "Spam" if prediction[0] == 1 else "Ham"
    return render_template('result.html', prediction=result)

if __name__ == '__main__':
    app.run(debug=True)