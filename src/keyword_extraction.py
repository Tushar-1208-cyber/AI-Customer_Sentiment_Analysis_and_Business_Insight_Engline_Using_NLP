# ============================================
# KEYWORD EXTRACTION USING TF-IDF
# ============================================

# This script:
# 1. Loads processed review data
# 2. Uses TF-IDF to find important words
# 3. Extracts top keywords overall
# 4. Extracts keywords per sentiment (positive/negative)

# ============================================
# IMPORT LIBRARIES
# ============================================
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


# ============================================
# LOAD DATA
# ============================================
df = pd.read_csv("../data/processed_reviews.csv")

# Remove any empty reviews (NaN values)
df = df.dropna(subset=["processed_text"])

# Use processed text (important)
texts = df["processed_text"].astype(str)


# ============================================
# APPLY TF-IDF VECTORIZATION
# ============================================
# Upgraded with bigrams, custom stopwords, and frequency filtering
custom_stopwords = ["product", "item", "use", "work", "read", "book", "author", "story", "good", "great", "just", "quality", "excellent", "perfect"]
from sklearn.feature_extraction import text
final_stopwords = list(text.ENGLISH_STOP_WORDS.union(custom_stopwords))

vectorizer = TfidfVectorizer(
    max_features=5000,
    stop_words=final_stopwords,
    ngram_range=(1, 2),  # include bigrams
    min_df=5             # ignore rare words
)
X = vectorizer.fit_transform(texts)

# Get feature names (words)
feature_names = vectorizer.get_feature_names_out()


# ============================================
# FUNCTION: GET TOP KEYWORDS
# ============================================
def get_top_keywords(tfidf_matrix, feature_names, top_n=10):
    """
    Returns top N keywords based on average TF-IDF score
    """
    import numpy as np

    # Calculate mean TF-IDF score for each word
    scores = tfidf_matrix.mean(axis=0).A1

    # Get top indices
    top_indices = scores.argsort()[::-1][:top_n]

    # Return top words
    return [feature_names[i] for i in top_indices]


# ============================================
# OVERALL KEYWORDS
# ============================================
print("\n===== TOP KEYWORDS (OVERALL) =====\n")
top_keywords = get_top_keywords(X, feature_names, top_n=15)
print(top_keywords)


# ============================================
# KEYWORDS BY SENTIMENT
# ============================================

def get_keywords_by_sentiment(sentiment):
    """
    Extract keywords for a specific sentiment
    """
    subset = df[df["sentiment"] == sentiment]
    texts = subset["processed_text"].astype(str)

    vec = TfidfVectorizer(
        max_features=3000,
        stop_words=final_stopwords,
        ngram_range=(1, 2),
        min_df=5
    )
    X_subset = vec.fit_transform(texts)

    return get_top_keywords(X_subset, vec.get_feature_names_out(), top_n=10)


print("\n===== POSITIVE KEYWORDS =====\n")
print(get_keywords_by_sentiment("positive"))

print("\n===== NEGATIVE KEYWORDS =====\n")
print(get_keywords_by_sentiment("negative"))