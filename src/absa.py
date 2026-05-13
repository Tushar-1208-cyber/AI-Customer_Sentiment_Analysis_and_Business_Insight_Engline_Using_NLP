# ============================================
# ASPECT-BASED SENTIMENT ANALYSIS (ABSA)
# ============================================

import pandas as pd
import ast

# ============================================
# LOAD DATA
# ============================================
try:
    df = pd.read_csv("../data/processed_reviews.csv")
    df['pos_tags'] = df['pos_tags'].apply(ast.literal_eval)
except FileNotFoundError:
    print("Error: processed_reviews.csv not found. Please run preprocessing.py first.")
    exit()

# ============================================
# STRICT LEXICON & FILTERS
# ============================================
positive_words = {"good", "great", "amazing", "excellent", "love"}
negative_words = {"bad", "poor", "worst", "blurry", "hate"}

invalid_words = {
    "is", "was", "are", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did",
    "can", "could", "should", "would",
    "just", "only", "also", "very", "so", "too",
    "write", "read", "got", "make", "part", "know",
    "right", "main", "all", "of", "and", "but", "this", "that",
    "i", "me", "my", "you", "your", "he", "she", "it", "we", "they",
    "who", "what", "which", "when", "where", "how", "why"
}

# ============================================
# ABSA CORE FUNCTION (STRICT)
# ============================================
def absa_from_pos(pos_tags, window=3, debug=False):
    """
    STRICT ABSA: Filters noisy aspects and only uses exact sentiment matches.
    """
    results = {}
    
    for i, (word, tag) in enumerate(pos_tags):
        # 1. CLEAN TOKEN & ASPECT FILTERING
        clean_word = word.strip(".,!?;:\"'()[]").lower()
        
        # Only allow NOUNs that are alphabetic, > 2 chars, and not in invalid_words
        if tag == "NOUN" and len(clean_word) > 2 and clean_word.isalpha():
            if clean_word in invalid_words:
                continue
            
            aspect = clean_word
            
            # 2. CONTEXT WINDOW SEARCH
            start = max(0, i - window)
            end = min(len(pos_tags), i + window + 1)
            
            best_sentiment = None
            matched_word = None
            min_dist = float('inf')
            
            for j in range(start, end):
                if i == j: continue
                
                s_word, s_tag = pos_tags[j]
                s_word_clean = s_word.strip(".,!?;:\"'()[]").lower()
                
                # 3. STRICT SENTIMENT MATCHING
                is_pos = s_word_clean in positive_words
                is_neg = s_word_clean in negative_words
                
                if is_pos or is_neg:
                    base_sentiment = "positive" if is_pos else "negative"
                    
                    # 4. NEGATION HANDLING
                    negated = False
                    for k in range(max(0, j-2), j):
                        prev_word = pos_tags[k][0].strip(".,!?;:\"'()[]").lower()
                        if prev_word in ["not", "no", "never", "n't"] or pos_tags[k][1] == "NEGATION":
                            negated = True
                            break
                    
                    final_sentiment = base_sentiment
                    if negated:
                        final_sentiment = "negative" if base_sentiment == "positive" else "positive"
                    
                    # Track closest sentiment
                    dist = abs(j - i)
                    if dist < min_dist:
                        min_dist = dist
                        best_sentiment = final_sentiment
                        matched_word = s_word_clean
            
            # 5. OUTPUT ASSIGNMENT
            if best_sentiment:
                results[aspect] = best_sentiment
                if debug:
                    print(f"  [DEBUG] Aspect: '{aspect}' | Sentiment Word: '{matched_word}' | Mapping: {best_sentiment}")
                    
    return results

# ============================================
# APPLY ABSA TO DATAFRAME
# ============================================
print("Performing Strict Aspect-Based Sentiment Analysis...")
df["aspect_sentiment"] = df["pos_tags"].apply(absa_from_pos)

# ============================================
# DEBUGGING SAMPLE OUTPUTS
# ============================================
print("\n===== DEBUGGING SAMPLE OUTPUTS =====\n")
sample_reviews = [
    "The camera is good but the battery life is poor.",
    "This laptop is amazing and not expensive.",
    "I love this phone but the screen is bad."
]

# Note: These are simulated to show the debug print in action
from preprocessing import tokenize, pos_tagger
for text in sample_reviews:
    print(f"Text: '{text}'")
    tokens = tokenize(text)
    tags = pos_tagger(tokens)
    res = absa_from_pos(tags, debug=True)
    print(f"Final Output: {res}\n" + "-"*40)

# ============================================
# REAL DATASET SAMPLES (CLEAN)
# ============================================
print("\n===== CLEAN RESULTS FROM DATASET =====\n")
# Filter for rows that have at least one valid aspect-sentiment pair
clean_df = df[df["aspect_sentiment"].apply(lambda x: len(x) > 0)].head(10)

for _, row in clean_df.iterrows():
    print(f"Review Snippet: {row['review_text'][:80]}...")
    print(f"Clean ABSA: {row['aspect_sentiment']}")
    print("-" * 50)

# ============================================
# GENERIC TEST CASE
# ============================================

test_text = "The camera is blurry but the screen is amazing."
sim_pos = [
    ('The', 'DETERMINER'), ('camera', 'NOUN'), ('is', 'VERB'), ('blurry', 'ADJ'), 
    ('but', 'CONJUNCTION'), ('the', 'DETERMINER'), ('screen', 'NOUN'), ('is', 'VERB'), ('amazing', 'ADJ')
]
# Note: since we are being STRICT, "blurry" and "amazing" must be in the lexicon
# Adding them temporarily to the lexicon for the generic test if not present
print(f"Input: '{test_text}'")
print(f"Output: {absa_from_pos(sim_pos)}")
