# app.py
from flask import Flask, request, jsonify
from newspaper import Article
import joblib
import re
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import spacy
import numpy as np
import nltk
from urllib.parse import urlparse

# Setup NLP
nltk.download("stopwords")
nltk.download("wordnet")
nlp = spacy.load("en_core_web_sm")
lemma = WordNetLemmatizer()
spacy_stop = nlp.Defaults.stop_words

# NLTK stopwords (convert list to set)
nltk_stop = set(stopwords.words("english"))

# Combine them
stop_words = spacy_stop | nltk_stop

# Load trained model
loaded_model = joblib.load("best_svm_model.pkl")


# Preprocessing function (identical from the notebook)
def clean_text(text):
    text = text.lower()
    # Simple contractions
    text = re.sub(r"i'm", "i am", text)
    text = re.sub(r"he's", "he is", text)
    text = re.sub(r"she's", "she is", text)
    text = re.sub(r"that's", "that is", text)
    text = re.sub(r"what's", "what is", text)
    text = re.sub(r"where's", "where is", text)
    text = re.sub(r"\'ll", " will", text)
    text = re.sub(r"\'ve", " have", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"\'d", " would", text)
    text = re.sub(r"won't", "will not", text)
    text = re.sub(r"can't", "cannot", text)
    # Remove special characters
    text = re.sub(r"[-()\"#!@$%^&*{}?.,:;']", " ", text)
    text = re.sub(r"\s+", " ", text)
    # Lemmatize and remove stopwords
    cleaned = " ".join(
        [lemma.lemmatize(w) for w in text.split() if w not in stop_words]
    )
    return cleaned


# Flask app
app = Flask(__name__)


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    url = data.get("url") if data else None

    # Check for empty input
    if not url or not url.strip():
        return jsonify({"error": "Please provide a URL"}), 400

    # Check if it's a valid URL format
    parsed = urlparse(url)
    if not parsed.scheme or not parsed.netloc:
        return jsonify({"error": "Invalid URL format"}), 400

    try:
        # Now, we're using a Article scrapper using Article library
        # We will first initialize it
        article = Article(
            url,
            browser_user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/141.0.0.0 Safari/537.36",
        )

        # we use this browser_user_agent like this since many news sites
        # (e.g., ABC, CNN, or paywalled outlets) block “bot” requests or return an empty page
        # Therefore, setting it means we're impersonating Chrome on Windows,
        # which helps ensure the request looks legitimate and the article downloads successfully.

        article.download()  # download raw HTML from page's url
        article.parse()  # identify title and article body
        # then remove unnecessary tags like ads, navigation menu or unrealted text
        full_text = article.title + " " + article.text  # combine title and text

        # Preprocess
        cleaned_text = clean_text(full_text)

        # Predict label 1 or 0
        pred_class = loaded_model.predict([cleaned_text])[0]

        # Since LinearSVC does not return probability, we will return confidence proxy
        decision_score = loaded_model.decision_function([cleaned_text])[0]
        prob_fake = 1 / (1 + np.exp(-decision_score))
        prob_real = 1 - prob_fake

        # Adjust label if class 0 = REAL
        predicted_label = "REAL" if pred_class == 0 else "FAKE"

        return jsonify(
            {
                "prediction": predicted_label,  # Label name: real or fake
                "confidence": {
                    "REAL": round(float(prob_real), 3),  # confidence of real and fake
                    "FAKE": round(float(prob_fake), 3),
                },
                "article_text_snippet": cleaned_text[
                    :500
                ],  # show the first 500 words of that article
            }
        )

    except Exception as e:
        return (
            jsonify({"error": str(e)}),
            500,
        )  # return error message when the link provided is not a real link


if __name__ == "__main__":
    app.run(debug=True)
