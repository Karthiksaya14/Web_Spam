import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
import string

# Load the dataset
df = pd.read_csv("spam_ham_dataset.csv")
df = df.rename(columns={'v1': 'label_num', 'v2': 'text'})
df = df[['text', 'label_num']]

# Encode labels
df['label_num'] = df['label_num'].map({'ham': 0, 'spam': 1})

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

# Apply preprocessing
corpus = df['text'].apply(preprocess_text)

# Vectorize the text
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(corpus).toarray()
y = df['label_num']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the classifier
clf = RandomForestClassifier(n_jobs=-1, random_state=42)
clf.fit(X_train, y_train)

# Evaluate the model
accuracy = clf.score(X_test, y_test)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Save the model and vectorizer
joblib.dump(vectorizer, 'vectorizer.joblib')
joblib.dump(clf, 'model.joblib')