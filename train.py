import pandas as pd
import numpy as np
import pickle
import re
import pandas as pd
import re
import string
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Download stopwords if not available
nltk.download("stopwords")

# Load dataset
df = pd.read_csv("custom_spam_dataset.csv", encoding="latin-1")

# Keep only relevant columns
df = df[['label', 'text']]
df.columns = ['label', 'text']
df['label'] = df['label'].map({'ham': 0, 'spam': 1})  # Convert labels to binary

# Text preprocessing function
stemmer = PorterStemmer()
stop_words = set(stopwords.words("english"))

def clean_text(text):
    text = text.lower()  # Lowercase
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    text = ' '.join([stemmer.stem(word) for word in text.split() if word not in stop_words])  # Stemming & stopword removal
    return text

df['clean_text'] = df['text'].apply(clean_text)

# Feature extraction (TF-IDF)
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))  # Unigrams & bigrams
X = vectorizer.fit_transform(df['clean_text'])
y = df['label']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training (Optimized with Random Forest)
model = RandomForestClassifier(n_estimators=300, max_depth=50, random_state=42)
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Model Accuracy: {accuracy:.4f}")

# Save model & vectorizer
pickle.dump(model, open("spam_model.pkl", "wb"))
pickle.dump(vectorizer, open("tfidf_vectorizer.pkl", "wb"))

print("🎯 Model and vectorizer saved successfully!")
import pandas as pd
import re
import string
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Download stopwords if not available
nltk.download("stopwords")

# Load dataset
df = pd.read_csv("custom_spam_dataset.csv", encoding="latin-1")

# Keep only relevant columns
df = df[['label', 'text']]
df.columns = ['label', 'text']
df['label'] = df['label'].map({'ham': 0, 'spam': 1})  # Convert labels to binary

# Text preprocessing function
stemmer = PorterStemmer()
stop_words = set(stopwords.words("english"))

def clean_text(text):
    text = text.lower()  # Lowercase
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    text = ' '.join([stemmer.stem(word) for word in text.split() if word not in stop_words])  # Stemming & stopword removal
    return text

df['clean_text'] = df['text'].apply(clean_text)

# Feature extraction (TF-IDF)
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))  # Unigrams & bigrams
X = vectorizer.fit_transform(df['clean_text'])
y = df['label']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training (Optimized with Random Forest)
model = RandomForestClassifier(n_estimators=300, max_depth=50, random_state=42)
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Model Accuracy: {accuracy:.4f}")

# Save model & vectorizer
pickle.dump(model, open("spam_model.pkl", "wb"))
pickle.dump(vectorizer, open("tfidf_vectorizer.pkl", "wb"))

print("🎯 Model and vectorizer saved successfully!")
