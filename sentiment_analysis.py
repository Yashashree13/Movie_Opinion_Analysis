import re
import joblib
import nltk
from nltk.corpus import stopwords

# Download stopwords
nltk.download("stopwords")
stop_words = set(stopwords.words("english"))

# --------- Clean Function ---------
def clean_text(text):
    text = str(text)
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"[^a-zA-Z]", " ", text)
    text = text.lower()
    text = " ".join([w for w in text.split() if w not in stop_words])
    return text

# --------- Load Model & Vectorizer ---------
model = joblib.load("sentiment_model.pkl")
tfidf = joblib.load("tfidf_vectorizer.pkl")

# --------- Predict Function ---------
def predict_review(review):
    review_clean = clean_text(review)
    review_vec = tfidf.transform([review_clean])
    pred = model.predict(review_vec)[0]
    
    # Optional: Map to Positive/Neutral/Negative with 5-star rating
    prob = model.predict_proba(review_vec)[0] if hasattr(model, "predict_proba") else None
    if pred == 1:
        return "Positive 😀 (4-5 Stars)"
    elif pred == 0:
        return "Negative 😡 (1-2 Stars)"
    else:
        return "Neutral 😐 (3 Stars)"

# --------- Interactive Loop ---------
if __name__ == "__main__":
    print("Type your review (multi-line allowed). Type 'exit' to quit.")
    while True:
        review = input("\nEnter review: ")
        if review.lower() == "exit":
            break
        print("Predicted Sentiment:", predict_review(review))