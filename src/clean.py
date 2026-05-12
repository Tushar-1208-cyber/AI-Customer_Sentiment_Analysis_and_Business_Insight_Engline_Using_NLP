import pandas as pd
import json
import re

# ================================
# STEP 1: LOAD DATA
# ================================
data = []

with open("Electronics.json", "r", encoding="utf-8") as f:
    for i, line in enumerate(f):
        if i >= 50000:   # limit (increase later if needed)
            break
        
        review = json.loads(line)
        
        data.append({
            "review_text": review.get("reviewText"),
            "rating": review.get("overall"),
            "date": review.get("reviewTime"),
            "summary": review.get("summary")
        })

df = pd.DataFrame(data)

# ================================
# STEP 2: REMOVE NULL VALUES
# ================================
df.dropna(subset=["review_text", "rating"], inplace=True)

# ================================
# STEP 3: CREATE SENTIMENT LABEL
# ================================
def get_sentiment(rating):
    if rating >= 4:
        return "positive"
    elif rating == 3:
        return "neutral"
    else:
        return "negative"

df["sentiment"] = df["rating"].apply(get_sentiment)

# ================================
# STEP 4: CLEAN TEXT
# ================================
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", "", text)  # remove symbols
    text = re.sub(r"\s+", " ", text)         # remove extra spaces
    return text.strip()

df["clean_text"] = df["review_text"].apply(clean_text)

# ================================
# STEP 5: FINAL DATA
# ================================
# We keep review_text (original) for NER/Rules and clean_text for ML models
df = df[["review_text", "clean_text", "rating", "sentiment", "date"]]

# ================================
# STEP 6: SAVE FILE
# ================================
df.to_csv("../data/cleaned_reviews.csv", index=False)

# ================================
# STEP 7: OUTPUT
# ================================
print("CLEANING DONE!")
print(df.head())
print("\nTotal rows:", len(df))