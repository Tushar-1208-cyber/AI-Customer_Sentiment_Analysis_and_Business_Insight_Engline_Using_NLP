# ============================================
# SENTIMENT ANALYSIS (BINARY CLASSIFICATION)
# ============================================

# This script:
# 1. Loads processed data
# 2. Removes neutral class
# 3. Converts text to TF-IDF vectors
# 4. Trains Logistic Regression model
# 5. Evaluates performance

# ============================================
# IMPORT LIBRARIES
# ============================================
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report


# ============================================
# LOAD DATA
# ============================================
df = pd.read_csv("../data/processed_reviews.csv")

# Defensive check: Remove any NaN reviews that might have slipped through
df = df.dropna(subset=["processed_text"])


# ============================================
# FEATURES AND LABELS
# ============================================
# Note: Neutral reviews and empty rows are now filtered in preprocessing.py
X = df["processed_text"]
y = df["sentiment"]


# ============================================
# TRAIN-TEST SPLIT
# ============================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# ============================================
# TF-IDF VECTORIZATION (IMPROVED)
# ============================================
vectorizer = TfidfVectorizer(
    max_features=5000,
    stop_words="english",
    ngram_range=(1, 2),
    min_df=5
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)


# ============================================
# TRAIN MODEL (BINARY LOGISTIC REGRESSION)
# ============================================
model = LogisticRegression(class_weight="balanced")
model.fit(X_train_vec, y_train)


# ============================================
# PREDICTIONS
# ============================================
y_pred = model.predict(X_test_vec)


# ============================================
# EVALUATION
# ============================================
if __name__ == "__main__":
    print("\n===== BINARY MODEL PERFORMANCE =====\n")
    print(classification_report(y_test, y_pred))


# ============================================
# PREDICTION FUNCTION
# ============================================
def predict_logit(text):
    """
    Predicts sentiment using the trained Logistic Regression model.
    """
    # Transform text to same TF-IDF space
    vec = vectorizer.transform([text])
    return model.predict(vec)[0]