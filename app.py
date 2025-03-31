from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Load trained model and vectorizer
model = pickle.load(open("spam_model.pkl", "rb"))
vectorizer = pickle.load(open("tfidf_vectorizer.pkl", "rb"))

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get email text from request
        email_text = request.json.get("email", "")
        if not email_text.strip():
            return jsonify({"error": "No email text provided"})

        # Vectorize the email using the previously trained vectorizer
        email_vectorized = vectorizer.transform([email_text])

        # Get probability predictions for spam and ham
        prediction_proba = model.predict_proba(email_vectorized)[0]
        
        spam_prob = round(prediction_proba[1] * 100, 2)  # Spam probability in percentage
        ham_prob = round(prediction_proba[0] * 100, 2)   # Ham probability in percentage

        # Assign prediction label (Spam or Ham)
        prediction_label = "Spam" if spam_prob > ham_prob else "Ham"

        # Prepare response with predictions
        response = {
            "spam": spam_prob,
            "ham": ham_prob,
            "prediction": prediction_label
        }
        
        # Return response as JSON
        return jsonify(response)
    
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
