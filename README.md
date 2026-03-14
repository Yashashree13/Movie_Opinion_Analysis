# 🎬 Movie Opinion Analysis
Movie Sentiment Analysis project that predicts whether a movie review is **Positive, Neutral, or Negative** using a trained **Machine Learning model**.
The system analyzes movie reviews and classifies their sentiment using Natural Language Processing (NLP) techniques and a model trained on the **IMDB movie review dataset**.
---
## 🚀 Features
- Analyze **single-line or multi-line movie reviews**
- Predicts **Positive, Neutral, or Negative sentiment**
- Displays sentiment with **emoji for better clarity**
- Model trained on **IMDB dataset**
- **Auto-saves trained model** so it can be reused without retraining
- Supports **CSV and text input reviews**
---
## 📂 Project Structure
Movie_Opinion_Analysis
│
├── IMDB Dataset.csv
├── imdb-dataset-of-50k-movie-reviews.zip
│
├── main.py
├── train_model.py
├── sentiment_analysis.py
├── predict_review.py
│
├── test_reviews.csv
├── test_reviews.txt
├── predicted_output.csv
│
├── requirements.txt
└── README.md

## 🧠 How It Works
1. **Data Collection**
   - Uses IMDB movie reviews dataset.

2. **Text Preprocessing**
   - Lowercasing
   - Removing punctuation
   - Removing stopwords
   - Tokenization

3. **Feature Extraction**
   - Converts text into numerical features using **TF-IDF Vectorization**.

4. **Model Training**
   - A machine learning model is trained on labeled movie reviews.

5. **Prediction**
   - New reviews are analyzed and classified into **Positive, Neutral, or Negative sentiment**.
---

## Installation
Clone the repository:
git clone https://github.com/YOUR_USERNAME/Movie_Opinion_Analysis.git
cd Movie_Opinion_Analysis
---

## ▶️ Usage
Run the prediction script:
python predict_review.py
Enter a movie review, and the system will output the predicted sentiment along with an emoji.
---
## Example
Input:
The movie was fantastic and the acting was brilliant!
Output:
Sentiment: Positive 😊
---

## 📊 Output
Prediction results can also be saved in:
predicted_output.csv
---

## 👩‍💻 Author
**Yashashree Mishra**  
B.Tech CSE
