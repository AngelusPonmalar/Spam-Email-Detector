import pandas as pd
import nltk
import pickle
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
df = pd.read_csv("custom_spam_dataset.csv")
df["label_num"] = df["label"].map({"ham": 0, "spam": 1})  # Convert labels to numeric

print("Columns in dataset:", df.columns)
print("Unique values in 'label':", df["label"].unique())

# Preprocessing function
nltk.download('stopwords')
nltk.download('punkt')

def preprocess_text(text):
    words = word_tokenize(text.lower())  # Convert to lowercase & tokenize
    words = [word for word in words if word.isalnum()]  # Remove special characters
    words = [word for word in words if word not in stopwords.words("english")]  # Remove stopwords
    return " ".join(words)

# Apply preprocessing
df["cleaned_text"] = df["text"].apply(preprocess_text)

# Feature extraction using TF-IDF
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df["cleaned_text"])  # Features
y = df["label_num"]  # Target labels

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = MultinomialNB()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluate Model Performance
accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Accuracy: {accuracy:.4f}")
print("🔍 Classification Report:\n", classification_report(y_test, y_pred))

# Save the trained model and vectorizer
pickle.dump(model, open("spam_model.pkl", "wb"))
pickle.dump(vectorizer, open("tfidf_vectorizer.pkl", "wb"))

print("✅ Model and vectorizer saved successfully as 'spam_model.pkl' and 'tfidf_vectorizer.pkl'!")
