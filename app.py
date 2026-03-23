import streamlit as st
import spacy
import pandas as pd
from textblob import TextBlob
import base64

# ================= LOAD CSS =================
def load_css():
    with open("index.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()

# ================= BACKGROUND =================
def set_bg():
    with open("bg.jpeg", "rb") as img:
        b64 = base64.b64encode(img.read()).decode()

    st.markdown(f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpeg;base64,{b64}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}
        </style>
    """, unsafe_allow_html=True)

set_bg()

# ================= LOAD MODEL =================
nlp = spacy.load("en_core_web_sm")

# ================= TITLE =================
st.markdown('<h1 class="main-title">🧠 NLP Text Analyzer</h1>', unsafe_allow_html=True)

# ================= CARD START =================
st.markdown('<div class="app-card">', unsafe_allow_html=True)

# ================= INPUT =================
st.markdown('<p class="stylish-heading">Enter Text</p>', unsafe_allow_html=True)
sentence = st.text_area("", placeholder="Type your sentence here...")

# ================= BUTTON =================
st.markdown('<div class="run-container">', unsafe_allow_html=True)
analyze = st.button("Analyze")
st.markdown('</div>', unsafe_allow_html=True)

# ================= PROCESS =================
if analyze:

    lower_case = sentence.lower()
    upper_case = sentence.upper()

    doc = nlp(sentence)

    tokens = [token.text for token in doc if not token.is_punct]

    pos_tags = [(token.text, token.pos_) for token in doc if not token.is_punct]

    lemma_data = [(token.text, token.pos_, token.lemma_) for token in doc if not token.is_punct]

   # ================= ENTITY FIX (FINAL 🔥) =================
    entities = [(ent.text, ent.label_) for ent in doc.ents]

# ALWAYS check tokens also (not only when empty)
    extra_entities = []

    for token in doc:
    # detect proper nouns (Hyderabad, India, etc.)
        if token.pos_ == "PROPN":
            extra_entities.append((token.text, "GPE (detected)"))

# merge both
    entities = list(set(entities + extra_entities))

    # ================= SENTIMENT =================
    blob = TextBlob(lower_case)
    polarity = blob.sentiment.polarity

    if polarity > 0.1:
        sentiment = "Positive 😊"
    elif polarity < -0.1:
        sentiment = "Negative 😡"
    else:
        sentiment = "Neutral 😐"

    # ================= OUTPUT =================
    st.markdown('<h3>Case Conversion</h3>', unsafe_allow_html=True)
    st.write("Lowercase:", lower_case)
    st.write("Uppercase:", upper_case)

    # TOKENS
    st.markdown('<h3>Tokens</h3>', unsafe_allow_html=True)
    st.markdown(" • ".join(tokens))

    # POS TAGS
    st.markdown('<h3>POS Tags</h3>', unsafe_allow_html=True)
    st.table(pd.DataFrame(pos_tags, columns=["Token", "POS Tag"]))

    # LEMMATIZATION
    st.markdown('<h3>Lemmatization</h3>', unsafe_allow_html=True)
    st.table(pd.DataFrame(lemma_data, columns=["Token", "POS", "Lemma"]))

    # ENTITIES
    st.markdown('<h3>Named Entities</h3>', unsafe_allow_html=True)
    if entities:
        st.table(pd.DataFrame(entities, columns=["Entity", "Label"]))
    else:
        st.write("No entities found.")

    # SENTIMENT
    st.markdown('<h3>Sentiment</h3>', unsafe_allow_html=True)
    st.success(sentiment)

# ================= CARD END =================
st.markdown('</div>', unsafe_allow_html=True)