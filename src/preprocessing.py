# import pandas as pd
# import spacy
# import os
# from tqdm import tqdm

# # ================================
# # LOAD SPACY MODEL
# # ================================
# try:
#     nlp = spacy.load("en_core_web_sm")
# except OSError:
#     print("SpaCy model 'en_core_web_sm' not found. Downloading...")
#     import subprocess
#     import sys
#     subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
#     nlp = spacy.load("en_core_web_sm")


# # ================================
# # LOAD CLEANED CSV DATA
# # ================================
# # Ensuring the path works whether run from root or src directory
# script_dir = os.path.dirname(os.path.abspath(__file__))
# data_path = os.path.join(os.path.dirname(script_dir), "data", "cleaned_reviews.csv")

# if not os.path.exists(data_path):
#     # Fallback to current directory for flexibility
#     data_path = "data/cleaned_reviews.csv"

# print(f"Loading data from: {data_path}")
# df = pd.read_csv(data_path)


# # ================================
# # BATCH NLP PROCESSING
# # ================================
# # We combine Tokenization, Stopword Removal, Lemmatization, POS Tagging, and NER 
# # into a single efficient pass using nlp.pipe for much faster execution.

# processed_texts = []
# pos_tags_list = []
# entities_list = []

# print(f"Processing {len(df)} reviews using spaCy...")

# # nlp.pipe is the optimized way to process large amounts of text in batches
# for doc in tqdm(nlp.pipe(df["clean_text"].astype(str), batch_size=50), total=len(df)):
    
#     # 1. TOKENIZATION + STOPWORD REMOVAL + LEMMATIZATION
#     # spaCy splits sentence into individual words (tokens)
#     # Convert word to base form (running → run) and skip common words (is, the, etc.)
#     tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
#     processed_texts.append(" ".join(tokens))
    
#     # 2. POS TAGGING (Each token mapped to its grammatical role)
#     pos_tags_list.append([(token.text, token.pos_) for token in doc])
    
#     # 3. NAMED ENTITY RECOGNITION (NER) (Extract names, places, dates, etc.)
#     entities_list.append([(ent.text, ent.label_) for ent in doc.ents])

# # Add the results back to the dataframe
# df["processed_text"] = processed_texts
# df["pos_tags"] = pos_tags_list
# df["entities"] = entities_list


# # ================================
# # DISPLAY OUTPUT
# # ================================
# print("\n===== SAMPLE OUTPUT =====\n")

# print(df[[
#     "clean_text",
#     "processed_text",
#     "pos_tags",
#     "entities"
# ]].head())


# # ================================
# # SAVE PROCESSED DATA (FOR ML MODEL)
# # ================================
# output_path = os.path.join(os.path.dirname(script_dir), "data", "processed_reviews.csv")
# print(f"\nSaving processed data to: {output_path}")
# df.to_csv(output_path, index=False)
# print("Saved successfully! ✅")




# ============================================
# CUSTOM NLP PIPELINE (WITHOUT SPACY)
# ============================================

import pandas as pd
import re

# ============================================
# LOAD DATA
# ============================================
df = pd.read_csv("../data/cleaned_reviews.csv")

# ============================================
# STEP 1: TEXT CLEANING
# ============================================
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# ============================================
# STEP 2: TOKENIZATION
# ============================================
def tokenize(text):
    return text.split()

# ============================================
# STEP 3: STOPWORD REMOVAL
# ============================================
stopwords = set([
    'the', 'is', 'and', 'in', 'to', 'it', 'of', 'for', 'on', 'this',
    'that', 'was', 'with', 'as', 'but', 'are', 'i', 'you', 'he', 'she', 
    'they', 'we', 'have', 'has', 'had', 'do', 'does', 'did', 'a', 'an', 
    'be', 'been', 'being'
])

def remove_stopwords(tokens):
    return [word for word in tokens if word not in stopwords]

# ============================================
# STEP 4: LEMMATIZATION (IMPROVED RULE-BASED)
# ============================================
def lemmatize(tokens):
    lemmas = []
    for word in tokens:
        if word.endswith("ies") and len(word) > 4:
            word = word[:-3] + "y"
        elif word.endswith("ing") and len(word) > 4:
            word = word[:-3]
        elif word.endswith("ed") and len(word) > 3:
            word = word[:-2]
        elif word.endswith("s") and len(word) > 3:
            word = word[:-1]
        lemmas.append(word)
    return lemmas

# ============================================
# STEP 5: RULE-BASED POS TAGGING (NEW)
# ============================================
def pos_tagger(tokens):
    """
    Improved Rule-Based POS Tagger.
    Assigns tags based on expanded dictionaries, suffixes, and patterns.
    """
    tagged_output = []
    
    # Expanded Rule Dictionaries
    verbs = {
        "is", "am", "are", "was", "were", "be", "been", "being", 
        "have", "has", "had", "do", "does", "did",
        "eat", "read", "write", "run", "go", "going", "gone", "buy", "bought", "get", "got", "make", "made"
    }
    determiners = {"the", "a", "an", "this", "that", "these", "those", "my", "your", "his", "her", "its", "our", "their"}
    conjunctions = {"and", "or", "but", "so", "yet", "nor", "for"}
    prepositions = {"in", "on", "at", "to", "for", "with", "by", "about", "as", "into", "through", "after", "before", "under", "over"}
    pronouns = {"i", "you", "he", "she", "it", "we", "they", "me", "him", "her", "us", "them", "who", "whom", "which", "what"}
    negations = {"not", "no", "never", "none", "neither", "nor", "n't"}
    
    # Suffix rules
    adj_suffixes = ("ous", "ful", "able", "ive", "al", "ic", "ish", "less")
    noun_suffixes = ("tion", "ment", "ness", "ity", "ism", "ship", "ance", "ence", "ist")
    adv_suffixes = ("ly", "ward", "wise")
    
    for i, word in enumerate(tokens):
        # Strip common punctuation for cleaner suffix matching
        word_clean = word.lower().strip(".,!?;:\"'()[]")
        prev_word = tokens[i-1].lower() if i > 0 else ""
        
        # Rule 1: Specific Lookups (Highest Priority)
        if word_clean in verbs:
            tag = "VERB"
        elif word_clean in pronouns:
            tag = "PRONOUN"
        elif word_clean in determiners:
            tag = "DETERMINER"
        elif word_clean in prepositions:
            tag = "PREPOSITION"
        elif word_clean in negations:
            tag = "NEGATION"
        elif word_clean in conjunctions:
            tag = "CONJUNCTION"
            
        # Rule 2: Numbers
        elif word_clean.isdigit():
            tag = "NUM"
            
        # Rule 3: Contextual Rule - "to" + word -> VERB (Infinitive)
        elif prev_word == "to":
            tag = "VERB"

        # Rule 4: Proper Nouns (Capitalized and not first word)
        elif i > 0 and word and word[0].isupper() and word_clean not in verbs:
            tag = "PROPER_NOUN"
            
        # Rule 5: Suffix - "ing", "ed", "es", "s" (if common verb endings)
        elif word_clean.endswith(("ing", "ed", "es")):
            tag = "VERB"
            
        # Rule 6: Suffix - Adverbs
        elif word_clean.endswith(adv_suffixes):
            tag = "ADVERB"
            
        # Rule 7: Suffix - Adjectives
        elif word_clean.endswith(adj_suffixes):
            tag = "ADJ"
            
        # Rule 8: Suffix - Nouns (including possessives like banana's)
        elif word_clean.endswith(noun_suffixes) or word_clean.endswith("'s"):
            tag = "NOUN"
            
        # Rule 9: Default
        else:
            # Check if it starts with a capital letter (even if first word) for Proper Noun fallback
            if word and word[0].isupper() and i == 0:
                tag = "PROPER_NOUN"
            else:
                tag = "NOUN"
            
        tagged_output.append((word, tag))
        
    return tagged_output

# ============================================
# STEP 6: RULE-BASED NER TAGGING (NEW)
# ============================================
def ner_tagger(tokens):
    """
    Identifies entities like PERSON, LOCATION, DATE, and NUMBER.
    """
    entities = []
    
    # Predefined lists
    locations = {"India", "USA", "London", "UK", "New York", "Paris", "Germany"}
    months = {"January", "February", "March", "April", "May", "June", 
              "July", "August", "September", "October", "November", "December"}
    titles = {"Mr", "Mr.", "Dr", "Dr.", "Mrs", "Mrs.", "Ms", "Ms."}
    pronouns = {"I", "You", "He", "She", "It", "We", "They", "Me", "Him", "Her", "Us", "Them"}
    
    # Regex patterns
    date_pattern = r"\d{1,2}/\d{1,2}/\d{2,4}"
    
    skip_next = False
    for i, word in enumerate(tokens):
        if skip_next:
            skip_next = False
            continue
            
        word_clean = word.strip(".,!?;:\"'()[]")
        
        # Rule 1: Specific Titles followed by Capitalized Word -> PERSON
        if word_clean in titles and i + 1 < len(tokens):
            next_word = tokens[i+1].strip(".,!?;:\"'()[]")
            if next_word and next_word[0].isupper():
                entities.append((f"{word} {tokens[i+1]}", "PERSON"))
                skip_next = True
                continue 
        
        # Rule 2: Date Patterns (Regex or Months)
        if re.match(date_pattern, word_clean) or word_clean in months:
            entities.append((word, "DATE"))
            
        # Rule 3: Specific Location Lookup
        elif word_clean in locations:
            entities.append((word, "LOCATION"))
            
        # Rule 4: Numbers
        elif word_clean.isdigit():
            entities.append((word, "NUMBER"))
            
        # Rule 5: Capitalized Word (not first word) -> PERSON
        elif i > 0 and word_clean and word_clean[0].isupper():
            # Exclude locations, months, and pronouns
            if word_clean.lower() not in locations and word_clean not in months and word_clean not in pronouns:
                entities.append((word, "PERSON"))
                
    return entities

# ============================================
# APPLY PIPELINE
# ============================================
def process_text(text, return_pos=False, return_ner=False):
    # Note: text is already cleaned in cleaned_reviews.csv
    # We use original text for NER/POS to keep capitalization
    tokens = tokenize(str(text))
    
    # Apply POS tagging and NER tagging
    pos_tags = pos_tagger(tokens)
    entities = ner_tagger(tokens)
    
    # Continue with stopword removal and lemmatization for the clean text
    clean_tokens = remove_stopwords(tokens)
    clean_tokens = lemmatize(clean_tokens)
    
    result = [" ".join(clean_tokens)]
    if return_pos: result.append(pos_tags)
    if return_ner: result.append(entities)
    
    return tuple(result) if len(result) > 1 else result[0]

# Applying the pipeline
# 1. Processed text still uses 'clean_text' (lowercased)
df["processed_text"] = df["clean_text"].apply(lambda x: process_text(x))

# 2. POS tags and Entities now use 'review_text' (keeps capitalization for accuracy)
df["pos_tags"] = df.apply(lambda row: process_text(row["review_text"], return_pos=True)[1], axis=1)
df["entities"] = df.apply(lambda row: process_text(row["review_text"], return_ner=True)[1], axis=1)

df = df[df["sentiment"] != "neutral"]
df = df.dropna(subset=["processed_text"])
df = df[df["processed_text"].str.strip() != ""] # Remove empty strings that become NaNs on reload

# ============================================
# OUTPUT & EXAMPLE USAGE
# ============================================
print("\n===== PROCESSED DATA WITH POS & NER =====\n")
print(df[["clean_text", "processed_text", "pos_tags", "entities"]].head())

# Manual Example Usage
print("\n===== MANUAL EXAMPLE =====\n")
sample_text = "Mr. Smith visited London and India in January 2024. He saw 5 birds on 12/10/2023."
tokens = tokenize(sample_text)
tags = pos_tagger(tokens)
entities = ner_tagger(tokens)

print(f"Input Text: {sample_text}")
print(f"POS Tags: {tags[:10]}...") # truncate for brevity
print(f"Entities: {entities}")

# ============================================
# SAVE PROCESSED DATA
# ============================================
df.to_csv("../data/processed_reviews.csv", index=False)
print("\nSaved processed data to: ../data/processed_reviews.csv")