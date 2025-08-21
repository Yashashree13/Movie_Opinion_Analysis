import joblib
import re
import nltk
from nltk.corpus import stopwords

# Download stopwords
nltk.download("stopwords")
stop_words = set(stopwords.words("english"))

# ---------- Load Trained Model ----------
MODEL_PATH = "sentiment_model.pkl"
VECTORIZER_PATH = "tfidf_vectorizer.pkl"

model = joblib.load(MODEL_PATH)
tfidf = joblib.load(VECTORIZER_PATH)

# ---------- Text Cleaning Function ----------
def clean_text(text):
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"[^a-zA-Z]", " ", text)
    text = text.lower()
    text = " ".join([w for w in text.split() if w not in stop_words])
    return text

# ---------- Prediction Function ----------
def predict_review(review):
    review_clean = clean_text(review)
    review_vec = tfidf.transform([review_clean])

    # If model supports predict_proba
    try:
        prob = model.predict_proba(review_vec)[0][1]
    except AttributeError:
        # For models without predict_proba (like LinearSVC), use decision_function
        prob = 0.5  # fallback, treat as neutral

    pred = model.predict(review_vec)[0]

    # Assign sentiment and stars
    if 0.4 <= prob <= 0.6:
        sentiment = "Neutral 😐"
        stars = 3
    elif pred == 1:
        sentiment = "Positive 😀"
        stars = 4 if prob < 0.9 else 5
    else:
        sentiment = "Negative 😡"
        stars = 2 if prob > 0.1 else 1

    return sentiment, stars

# ---------- Interactive Loop ----------
print("\n=== Movie Review Sentiment Predictor ===")
while True:
    movie_name = input("\nEnter movie name (or type 'exit' to quit): ")
    if movie_name.lower() == "exit":
        break

    review = input("Write your review: ")
    sentiment, stars = predict_review(review)

    print(f"\nMovie: {movie_name}")
    print(f"Predicted Sentiment: {sentiment}")
    print(f"Star Rating: {stars}/5")