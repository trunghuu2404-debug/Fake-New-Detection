# Fake-New-Detection

## Project structure
```
Fake-News-Detection/
├─ dataset                    # contain our fake and real news dataset
├─ api.py                     # Flask API that handles URL prediction
├─ interface.py               # Streamlit interface for user interaction
├─ logistic_regression.ipynb  # Training notebook of LR model
├─ model_svm.nb               # Training notebook of SVM model
├─ model_nb.ipynb             # Training notebook of Naive Bayes model
├─ best_svm_model.pkl         # Saved best SVM model
├─ best_logreg_model.pkl      # Saved best Logistic regression model
├─ best_nb_model.pkl          # Saved best Naive Bayes model
├─ README.md
```
This project is a **fake news detection system** that takes a URL of a news article, scrapes its content, preprocesses it, and predicts whether the news is REAL or FAKE using a best trained machine learning model (SVM).  

It includes:  
- A **Flask API** (`app.py`) that handles URL input, scrapes the article using `newspaper`, cleans the text, and returns predictions.  
- A **Streamlit interface** (`interface.py`) to input URLs and display predictions with probabilities and an article snippet.  

---

## Features
- Scrapes article text from any accessible URL using the `newspaper3k` library.  
- Text preprocessing: lowercasing, removing special characters, lemmatization, removing stopwords (from both SpaCy and NLTK).  
- Uses a trained **SVM model** for classification.  
- Returns **pseudo-probabilities** for REAL/FAKE when the model does not support `predict_proba`.  
- Displays predictions via a **simple web interface** with Streamlit.

---

## How to run

1. Navigate to api.py and run **python api.py** on terminal.
2. Then onto another terminal, run **streamlit run interface.py** to run the interface.

When on the interface:
* Prediction (REAL or FAKE)
* Probabilities for each class
* First 500 characters of the cleaned article
