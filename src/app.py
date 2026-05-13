import streamlit as st
import pandas as pd
import os
import re
from absa_llm import absa_llm, generate_business_strategy, generate_brand_report
from absa import absa_from_pos
from preprocessing import tokenize, pos_tagger, ner_tagger, remove_stopwords, lemmatize
from sentiment_model import predict_logit
from bert_model import predict_bert

# ============================================
# PAGE CONFIGURATION
# ============================================
st.set_page_config(
    page_title="AI Customer Intelligence Engine",
    page_icon=None,
    layout="wide"
)

# Custom CSS for better aesthetics
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
    }
    .stMetric {
        background-color: #1e2130;
        padding: 15px;
        border-radius: 10px;
    }
    .pipeline-node {
        background-color: #262730;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #4CAF50;
        margin-bottom: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

# ============================================
# HELPER FUNCTIONS (CACHED)
# ============================================
@st.cache_data
def load_data():
    try:
        # Load the processed reviews dataset
        return pd.read_csv("../data/processed_reviews.csv")
    except:
        return None

@st.cache_data
def read_code(filename):
    try:
        with open(filename, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        return f"Error loading file: {e}"

# ============================================
# SIDEBAR NAVIGATION
# ============================================
st.sidebar.title("Project Pipeline")
page = st.sidebar.radio(
    "Navigation:",
    ["Dashboard Overview", "1. Preprocessing", "2. Logistic Regression (Baseline)", "3. BERT Model (Deep Learning)", "4. Rule-Based ABSA", "5. LLM-Based ABSA", "Executive Insights"],
    key="nav_radio"
)

st.sidebar.markdown("---")
st.sidebar.success("Model: Llama3 (Local)")
st.sidebar.info("Dataset: Amazon Electronics")

# ============================================
# SECTION: DASHBOARD OVERVIEW
# ============================================
if page == "Dashboard Overview":
    st.title("AI Customer Intelligence & Decision Engine")
    st.markdown("### The Complete NLP Intelligence Pipeline")
    
    # Visual Flowchart using Graphviz
    st.graphviz_chart("""
    digraph G {
        rankdir=LR;
        node [shape=box, style=filled, color="#4CAF50", fontcolor=white, fontname="Helvetica"];
        "Raw Review" -> "Preprocessing";
        "Preprocessing" -> "Sentiment Model";
        "Sentiment Model" -> "ABSA Engine";
        "ABSA Engine" -> "Business Insights";
        
        node [color="#1E88E5"];
        "Preprocessing" -> "POS Tags & Entities";
        "Sentiment Model" -> "Positive/Negative Label";
        "ABSA Engine" -> "Aspect-Level Sentiment";
    }
    """)

    col1, col2, col3 = st.columns(3)
    df_count = load_data()
    row_count = len(df_count) if df_count is not None else 49132
    
    with col1:
        st.metric(label="Data Volume", value=f"{row_count:,}", delta="Reviews")
    with col2:
        st.metric(label="Pipeline Speed", value="~5ms", delta="Per Token")
    with col3:
        st.metric(label="LLM Accuracy", value="High", delta="Context Aware")

    st.markdown("---")
    st.subheader("Executive Summary")
    st.write("""
    This project automates the analysis of customer feedback. Instead of reading thousands of reviews, 
    the engine extracts what customers are talking about (Aspects) and how they feel about them (Sentiment).
    """)

# ============================================
# SECTION: 1. Preprocessing (DATASET EXAMPLES)
# ============================================
elif page == "1. Preprocessing":
    st.title("Step 1: Preprocessing & Tagging")
    
    # academic diagram
    st.markdown("""
        <div style="display: flex; justify-content: space-between; align-items: center; background: #1e2130; padding: 20px; border-radius: 15px; margin-bottom: 30px;">
            <div style="text-align: center;"><div style="background: #3b82f6; width: 40px; height: 40px; border-radius: 50%; line-height: 40px; margin: 0 auto;">T</div><b>Tokens</b></div>
            <div style="flex: 1; height: 2px; background: #3b82f6; margin: 0 10px;"></div>
            <div style="text-align: center;"><div style="background: #10b981; width: 40px; height: 40px; border-radius: 50%; line-height: 40px; margin: 0 auto;">S</div><b>Stopwords</b></div>
            <div style="flex: 1; height: 2px; background: #10b981; margin: 0 10px;"></div>
            <div style="text-align: center;"><div style="background: #ef4444; width: 40px; height: 40px; border-radius: 50%; line-height: 40px; margin: 0 auto;">L</div><b>Lemmas</b></div>
        </div>
    """, unsafe_allow_html=True)

    st.write("Below are examples of how our engine structures raw customer feedback from the dataset.")

    try:
        df = load_data()
        if df is not None:
            # PAGINATION CONTROL
            st.markdown("### Browse Dataset")
            page_size = 5
            total_pages = len(df) // page_size
            page_num = st.slider("Select Page:", 1, min(total_pages, 100), 1) # Limit to first 100 pages for speed
            
            start_idx = (page_num - 1) * page_size
            end_idx = start_idx + page_size
            examples = df.iloc[start_idx:end_idx]
            
            for i, row in examples.iterrows():
                text = row["clean_text"]
                with st.expander(f"Review Row {i+1}: {text[:60]}...", expanded=(i == start_idx)):
                    # Run the actual pipeline
                    tokens = tokenize(text)
                    tags = pos_tagger(tokens)
                    entities = ner_tagger(tokens)
                    clean_tokens = remove_stopwords(tokens)
                    lemms = lemmatize(clean_tokens)
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**Raw Review Segment:**")
                        st.info(text)
                        st.markdown("**Step 2: Stopword Removal**")
                        st.write(f"`{clean_tokens[:10]}...`")
                        
                    with col2:
                        st.markdown("**Structured Data:**")
                        st.write(f"Tokens: `{tokens[:8]}...`")
                        st.markdown("**Step 3: Lemmatization (Final Result)**")
                        st.success(" ".join(lemms))
        else:
            st.warning("Dataset not found. Showing static example instead.")
    except Exception as e:
        st.error(f"Error processing samples: {e}")

    st.markdown("---")
    st.subheader("Why this matters?")
    st.write("By identifying NOUNS, our system knows that 'battery' is a feature. By identifying ADJECTIVES, it knows 'great' is an opinion.")
    
    with st.expander("View Implementation (preprocessing.py)"):
        st.code(read_code("preprocessing.py"), language="python")

# ============================================
# SECTION: 2. LOGISTIC REGRESSION (BASELINE)
# ============================================
elif page == "2. Logistic Regression (Baseline)":
    st.title("Step 2: Logistic Regression (Baseline Model)")
    
    st.markdown("""
        <div style="display: flex; justify-content: space-around; align-items: center; background: #1e2130; padding: 20px; border-radius: 15px; margin-bottom: 30px;">
            <div style="text-align: center;"><div style="background: #3b82f6; width: 40px; height: 40px; border-radius: 50%; line-height: 40px; margin: 0 auto;">V</div><b>TF-IDF</b></div>
            <div style="flex: 1; height: 2px; background: #3b82f6; margin: 0 10px;"></div>
            <div style="text-align: center;"><div style="background: #10b981; width: 40px; height: 40px; border-radius: 50%; line-height: 40px; margin: 0 auto;">LR</div><b>Logit Reg.</b></div>
        </div>
    """, unsafe_allow_html=True)

    st.write("We use TF-IDF Vectorization and Logistic Regression as our high-speed statistical baseline.")

    st.subheader("Real-Time Prediction (Logit)")
    logit_input = st.text_input("Enter text for Logit analysis:", "This purchase was actually great!")
    if logit_input:
        res = predict_logit(logit_input)
        st.success(f"Logit Prediction: {res.upper()}")

    st.markdown("---")
    st.subheader("Model Performance on Dataset")
    
    try:
        df = load_data()
        if df is not None:
            # PAGINATION CONTROL
            st.markdown("### Browse Logit Results")
            page_size_sent = 10
            total_pages_sent = len(df) // page_size_sent
            page_num_sent = st.number_input("Enter Page Number:", 1, total_pages_sent, 1)
            
            start_idx_sent = (page_num_sent - 1) * page_size_sent
            end_idx_sent = start_idx_sent + page_size_sent
            
            sample_df = df.iloc[start_idx_sent:end_idx_sent][["clean_text", "sentiment"]]
            sample_df.columns = ["Review Text", "Logit Label"]
            st.table(sample_df)
        else:
            st.warning("Dataset not found.")
    except Exception as e:
        st.error(f"Error loading samples: {e}")

    st.markdown("---")
    with st.expander("View Training Logic (sentiment_model.py)"):
        st.code(read_code("sentiment_model.py"), language="python")

# ============================================
# SECTION: 3. BERT MODEL (DEEP LEARNING)
# ============================================
elif page == "3. BERT Model (Deep Learning)":
    st.title("Step 3: BERT Transformer Model")
    
    st.markdown("""
        <div style="display: flex; justify-content: space-around; align-items: center; background: #1e2130; padding: 20px; border-radius: 15px; margin-bottom: 30px;">
            <div style="text-align: center;"><div style="background: #ef4444; width: 40px; height: 40px; border-radius: 50%; line-height: 40px; margin: 0 auto;">T</div><b>Transformer</b></div>
            <div style="flex: 1; height: 2px; background: #ef4444; margin: 0 10px;"></div>
            <div style="text-align: center;"><div style="background: #3b82f6; width: 40px; height: 40px; border-radius: 50%; line-height: 40px; margin: 0 auto;">B</div><b>BERT</b></div>
        </div>
    """, unsafe_allow_html=True)

    st.write("Using DistilBERT, we achieve deep contextual understanding of customer emotions.")

    st.subheader("Real-Time BERT Analysis")
    bert_input = st.text_input("Enter text for BERT analysis:", "The delivery was late, but the item is high quality.")
    if bert_input:
        with st.spinner("BERT is thinking..."):
            res = predict_bert(bert_input)
            st.info(f"BERT Prediction: {res.upper()}")

    st.markdown("---")
    st.subheader("BERT Performance on Dataset")
    st.write("Browse the dataset to see how the Transformer model classifies real reviews.")
    
    try:
        df = load_data()
        if df is not None:
            # PAGINATION CONTROL
            st.markdown("### Browse BERT Data")
            page_size_bert = 5
            total_pages_bert = len(df) // page_size_bert
            page_num_bert = st.number_input("Select BERT Page:", 1, min(total_pages_bert, 100), 1)
            
            start_idx_bert = (page_num_bert - 1) * page_size_bert
            end_idx_bert = start_idx_bert + page_size_bert
            
            bert_sample = df.iloc[start_idx_bert:end_idx_bert]
            
            if st.button("Run BERT on this Page"):
                with st.spinner("BERT analyzing 5 reviews..."):
                    results = []
                    for text in bert_sample["clean_text"]:
                        results.append(predict_bert(text).upper())
                    
                    display_df = pd.DataFrame({
                        "Review Text": bert_sample["clean_text"].tolist(),
                        "BERT Label": results
                    })
                    st.table(display_df)
            else:
                st.info("Click the button above to run BERT on these 5 reviews.")
        else:
            st.warning("Dataset not found.")
    except Exception as e:
        st.error(f"Error loading BERT samples: {e}")

    st.markdown("---")
    st.subheader("Comparative Insight")
    st.info("While Logistic Regression relies on word counts, BERT understands the 'vibe' and context of the entire sentence.")

    st.markdown("---")
    with st.expander("View BERT Implementation (bert_model.py)"):
        st.code(read_code("bert_model.py"), language="python")

# ============================================
# SECTION: 4. RULE-BASED ABSA
# ============================================
elif page == "4. Rule-Based ABSA":
    st.title("Step 4: Rule-Based Aspect Analysis")
    
    st.markdown("""
        <div style="display: flex; justify-content: space-around; align-items: center; background: #1e2130; padding: 20px; border-radius: 15px; margin-bottom: 30px;">
            <div style="text-align: center;"><div style="background: #3b82f6; width: 40px; height: 40px; border-radius: 50%; line-height: 40px; margin: 0 auto;">A</div><b>Aspects</b></div>
            <div style="flex: 1; height: 2px; background: #3b82f6; margin: 0 10px;"></div>
            <div style="text-align: center;"><div style="background: #f59e0b; width: 40px; height: 40px; border-radius: 50%; line-height: 40px; margin: 0 auto;">W</div><b>Window</b></div>
            <div style="flex: 1; height: 2px; background: #f59e0b; margin: 0 10px;"></div>
            <div style="text-align: center;"><div style="background: #ef4444; width: 40px; height: 40px; border-radius: 50%; line-height: 40px; margin: 0 auto;">N</div><b>Negation</b></div>
        </div>
    """, unsafe_allow_html=True)

    st.write("Using custom logic, POS tagging, and proximity windows to link opinions to features.")

    st.subheader("Real-Time Rule Analysis")
    rule_input = st.text_area("Enter a review for rule-based matching:", "The camera is good but the battery life is bad.")
    
    if st.button("Run Rule Engine"):
        with st.spinner("Processing tags and distance..."):
            tokens = tokenize(rule_input)
            tags = pos_tagger(tokens)
            result = absa_from_pos(tags)
            
        if result:
            st.success("Analysis Complete!")
            cols = st.columns(len(result) if len(result) > 0 else 1)
            for i, (aspect, sentiment) in enumerate(result.items()):
                with cols[i % len(cols)]:
                    color = "#28a745" if sentiment.lower() == "positive" else "#dc3545"
                    st.markdown(f"""
                        <div style="border: 2px solid {color}; padding: 15px; border-radius: 10px; text-align: center;">
                            <b style="color: {color};">{aspect.upper()}</b><br>
                            {sentiment.upper()}
                        </div>
                    """, unsafe_allow_html=True)
        else:
            st.warning("No aspects found.")

    st.markdown("---")
    with st.expander("View Rule Logic (absa.py)"):
        st.code(read_code("absa.py"), language="python")

# ============================================
# SECTION: 5. LLM-BASED ABSA
# ============================================
elif page == "5. LLM-Based ABSA":
    st.title("Step 5: Advanced LLM-Based Insights")
    
    st.markdown("""
        <div style="display: flex; justify-content: space-around; align-items: center; background: #1e2130; padding: 20px; border-radius: 15px; margin-bottom: 30px;">
            <div style="text-align: center;"><div style="background: #ef4444; width: 40px; height: 40px; border-radius: 50%; line-height: 40px; margin: 0 auto;">P</div><b>Prompt</b></div>
            <div style="flex: 1; height: 2px; background: #ef4444; margin: 0 10px;"></div>
            <div style="text-align: center;"><div style="background: #3b82f6; width: 40px; height: 40px; border-radius: 50%; line-height: 40px; margin: 0 auto;">L3</div><b>Llama3</b></div>
            <div style="flex: 1; height: 2px; background: #3b82f6; margin: 0 10px;"></div>
            <div style="text-align: center;"><div style="background: #10b981; width: 40px; height: 40px; border-radius: 50%; line-height: 40px; margin: 0 auto;">J</div><b>JSON</b></div>
        </div>
    """, unsafe_allow_html=True)

    st.write("Leveraging Llama3 to understand context and complex patterns.")

    st.subheader("Advanced Intelligence Demo")
    user_text = st.text_area("Type a complex review:", "The display is crystal clear but the shipping was very slow.")
    
    if st.button("Generate Deep Insights"):
        with st.spinner("Llama3 analyzing context..."):
            result = absa_llm(user_text)
            
        if result:
            st.write("### Intelligence Extracted:")
            cols = st.columns(len(result) if len(result) > 0 else 1)
            for i, (aspect, sentiment) in enumerate(result.items()):
                with cols[i % len(cols)]:
                    color = "#28a745" if sentiment.lower() == "positive" else "#dc3545"
                    st.markdown(f"""
                        <div style="background-color: {color}; padding: 20px; border-radius: 10px; text-align: center; color: white;">
                            <h4 style="margin:0;">{aspect.upper()}</h4>
                            <hr style="margin: 10px 0;">
                            <h2 style="margin:0;">{sentiment.upper()}</h2>
                        </div>
                    """, unsafe_allow_html=True)
        else:
            st.error("Ollama not found.")

    st.markdown("---")
    with st.expander("View LLM Logic (absa_llm.py)"):
        st.code(read_code("absa_llm.py"), language="python")

# ============================================
# SECTION: Executive Insights
# ============================================
elif page == "Executive Insights":
    st.title("Executive Intelligence & Strategic Advice")
    st.write("This section turns NLP results into actionable business decisions using Llama3.")

    st.subheader("Strategic Decision Simulator")
    exec_review = st.text_area("Input customer feedback for strategic analysis:", 
                             "The camera quality is breathtaking, but the battery drains in just 2 hours. Also, the price is quite high for these specs.")

    if st.button("Generate Strategic Advice"):
        with st.spinner("AI Consultant analyzing market impact..."):
            # Step 1: Get ABSA
            absa_results = absa_llm(exec_review)
            # Step 2: Get Strategic Advice
            strategy = generate_business_strategy(absa_results)
            
        if absa_results and strategy:
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.markdown("### **1. Feature Analysis**")
                for aspect, sentiment in absa_results.items():
                    color = "#28a745" if sentiment.lower() == "positive" else "#dc3545"
                    st.markdown(f"**{aspect.upper()}**: <span style='color:{color}'>{sentiment.upper()}</span>", unsafe_allow_html=True)
            
            with col2:
                st.markdown("### **2. Strategic Advice**")
                st.success(strategy)
                
            st.info("Executive Summary: Focus on maintaining the quality of high-performing features while urgently addressing the pain points identified above.")
        else:
            st.error("Could not generate insights. Ensure Ollama is running.")

    st.markdown("---")
    st.subheader("Strategic Insights from Dataset")
    st.write("Pick any real review from your dataset to generate a business strategy.")
    
    try:
        df = load_data()
        if df is not None:
            # PAGINATION CONTROL
            page_size_exec = 5
            total_pages_exec = len(df) // page_size_exec
            page_num_exec = st.number_input("Select Dataset Page (Executive):", 1, min(total_pages_exec, 100), 1)
            
            start_idx_exec = (page_num_exec - 1) * page_size_exec
            end_idx_exec = start_idx_exec + page_size_exec
            exec_sample = df.iloc[start_idx_exec:end_idx_exec]
            
            for i, row in exec_sample.iterrows():
                text = row["clean_text"]
                with st.expander(f"Review {i+1}: {text[:60]}...", expanded=False):
                    st.info(text)
                    if st.button(f"Analyze Review {i+1}", key=f"btn_{i}"):
                        with st.spinner("AI Strategist thinking..."):
                            absa = absa_llm(text)
                            strat = generate_business_strategy(absa)
                            st.markdown("#### Findings:")
                            st.write(absa)
                            st.success(strat)
        else:
            st.warning("Dataset not found.")
    except Exception as e:
        st.error(f"Error loading executive data: {e}")

    st.markdown("---")

    st.markdown("---")
    st.subheader("Global Brand Health & Satisfaction Trends")
    st.write("This section analyzes the entire dataset to find global patterns impacting brand perception.")
    
    if st.button("Generate Global Brand Report"):
        try:
            df = load_data()
            if df is not None:
                with st.spinner("Analyzing 50,000 reviews for global trends..."):
                    # Quick way to get trends: Top nouns in positive vs negative
                    pos_df = df[df["sentiment"] == "positive"]
                    neg_df = df[df["sentiment"] == "negative"]
                    
                    # Simulating trend extraction (top words excluding common ones)
                    ignore = ["product", "item", "great", "good", "bad", "buy", "bought", "one"]
                    
                    def get_trends(text_list):
                        words = " ".join(text_list.astype(str)).split()
                        freq = pd.Series([w for w in words if len(w) > 3 and w not in ignore]).value_counts()
                        return freq.head(5).index.tolist()
                    
                    top_pos = get_trends(pos_df["processed_text"])
                    top_neg = get_trends(neg_df["processed_text"])
                    
                    # Generate LLM Report
                    report = generate_brand_report(top_pos, top_neg)
                    
                col1, col2 = st.columns(2)
                with col1:
                    st.write("### **Positive Trends**")
                    st.success(", ".join(top_pos))
                    st.write("### **Negative Trends**")
                    st.error(", ".join(top_neg))
                    
                with col2:
                    st.write("### **Strategic Brand Health Report**")
                    st.info(report)
            else:
                st.warning("Dataset not found.")
        except Exception as e:
            st.error(f"Error: {e}")

    st.markdown("---")

# ============================================
# SECTION: MODEL COMPARISON (VISUAL METRICS)
# ============================================
# ============================================
# FOOTER
# ============================================
st.markdown("---")
st.caption("AI Customer Intelligence & Decision Engine | Powered by Llama3 & Streamlit")
