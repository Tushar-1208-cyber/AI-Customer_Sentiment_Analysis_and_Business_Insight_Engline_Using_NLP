# ============================================
# LLM-BASED ABSA USING OLLAMA (ROBUST)
# ============================================

import ollama
import json
import re

# ============================================
# FUNCTION: LLM ABSA
# ============================================
def absa_llm(review, model_name="llama3:latest", retry=True):
    """
    Performs Aspect-Based Sentiment Analysis using Ollama LLM.
    Uses Llama3 by default for better instruction following.
    Includes robust JSON extraction and retry logic.
    """

    system_prompt = (
        "You are a strict NLP analyzer. Your ONLY output must be a FLAT JSON dictionary. "
        "Example Format: {\"aspect\": \"sentiment\", \"aspect2\": \"sentiment\"} "
        "Do NOT nest dictionaries. Do NOT use lists. Do NOT include any introduction. "
        "STRICT RULES: "
        "1. Extract ONLY the 3-5 most important aspects explicitly mentioned. "
        "2. Sentiments must ONLY be 'positive' or 'negative'. NEVER use 'neutral'. "
        "3. Output aspect names in lowercase with spaces (no underscores)."
    )
    
    user_prompt = f"""
Return a flat JSON dictionary for this review:
"{review}"
"""

    import time
    
    # 2-Attempt Loop for Network Stability
    for attempt in range(2):
        try:
            response = ollama.chat(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                options={"temperature": 0} # Maximum consistency
            )

            # Get raw response and clean common markdown artifacts
            raw_output = response["message"]["content"].strip()
            
            # Remove markdown code blocks if present (```json ... ```)
            clean_output = re.sub(r'```json\s*|\s*```', '', raw_output)

            # ROBUST JSON EXTRACTION: Find the first { and last }
            json_match = re.search(r'\{.*\}', clean_output, re.DOTALL)
            
            if json_match:
                json_str = json_match.group(0)
                try:
                    data = json.loads(json_str)
                    
                    # POST-PROCESSING: Standardize aspect names (lowercase, spaces)
                    final_data = {}
                    for aspect, sentiment in data.items():
                        clean_aspect = str(aspect).replace("_", " ").lower().strip()
                        # Final check: Skip if model still returned 'neutral'
                        if str(sentiment).lower() != "neutral":
                            final_data[clean_aspect] = str(sentiment).lower()
                    
                    return final_data
                except:
                    pass
            
            # If parsing failed but connection was okay, break loop and handle below
            break

        except Exception as e:
            if attempt == 0:
                print(f"🔄 Network glitch (500), retrying in 2 seconds...")
                time.sleep(2)
                continue
            else:
                print(f"❌ Connection Error after retry: {e}")
                return {}

    # Final fallback if parsing failed
    print(f"⚠️ Warning: Model {model_name} returned invalid output.")
    return {}


# ============================================
# TEST FUNCTION
# ============================================
def run_tests():
    print("\n===== ROBUST LLM ABSA TESTS (LLAMA3) =====\n")

    test_reviews = [
        "The camera is blurry but the screen is amazing",
        "This book has great characters but weak plot",
        "Battery life is poor but design is excellent",
        "The story is good but writing is bad"
    ]

    for review in test_reviews:
        print("Review:", review)
        result = absa_llm(review) # Uses llama3 by default
        print("Output:", result)
        print("-" * 50)


# ============================================
# APPLY TO DATASET (OPTIONAL)
# ============================================
def run_on_dataset():
    import pandas as pd

    print("\n===== RUNNING ON DATASET (MIXED SAMPLE) =====\n")

    try:
        df = pd.read_csv("../data/processed_reviews.csv")
        
        # Take 3 Positive and 3 Negative reviews to see the difference
        pos_sample = df[df["sentiment"] == "positive"].head(3)
        neg_sample = df[df["sentiment"] == "negative"].head(3)
        
        df_sample = pd.concat([pos_sample, neg_sample]).copy()
        print(f"Processing {len(df_sample)} mixed reviews...")
        
        df_sample["llm_absa"] = df_sample["clean_text"].apply(lambda x: absa_llm(x))
        
        print("\nResults (Positive & Negative Examples):")
        for _, row in df_sample.iterrows():
            print(f"\n[{row['sentiment'].upper()}] Review: {row['clean_text'][:80]}...")
            print(f"ABSA Output: {row['llm_absa']}")
        
    except FileNotFoundError:
        print("Error: processed_reviews.csv not found.")


# ============================================
# FUNCTION: GENERATE BUSINESS STRATEGY
# ============================================
def generate_business_strategy(absa_results, model_name="llama3:latest"):
    """
    Takes structured ABSA results and generates high-level business strategy.
    """
    if not absa_results:
        return "No data available to generate strategy."

    # Format the results for the prompt
    data_summary = json.dumps(absa_results)
    
    prompt = (
        f"Based on these results: {data_summary}, give 3 very short, simple tips for the business owner. "
        "Use easy language (no big business words). "
        "Example: 'People love the camera, keep it up!' or 'Battery is bad, fix it fast.' "
        "Just the 3 tips, very short, one per line."
    )

    try:
        # Using options to speed up generation by limiting tokens
        response = ollama.generate(
            model=model_name, 
            prompt=prompt,
            options={"num_predict": 100, "temperature": 0.5}
        )
        return response['response']
    except Exception as e:
        return f"Error generating strategy: {e}"

# ============================================
# FUNCTION: GENERATE BRAND REPORT
# ============================================
def generate_brand_report(top_positive, top_negative, model_name="llama3:latest"):
    """
    Generates a strategic brand perception report based on global dataset trends.
    """
    pos_str = ", ".join(top_positive)
    neg_str = ", ".join(top_negative)
    
    prompt = (
        f"Analyze these market trends. Customers love: {pos_str}. Customers hate: {neg_str}. "
        "Write a short 'Brand Health Report' (max 4 lines). "
        "Tell me: 1. Overall Brand Perception. 2. Main Satisfaction Driver. 3. Biggest Risk to Brand. "
        "Use simple, professional language."
    )

    try:
        response = ollama.generate(
            model=model_name, 
            prompt=prompt,
            options={"num_predict": 150}
        )
        return response['response']
    except Exception as e:
        return f"Error generating report: {e}"

# ============================================
# MAIN
# ============================================
if __name__ == "__main__":
    # STEP 1: Run basic tests
    run_tests()

    # STEP 2: Run on dataset sample
    run_on_dataset()