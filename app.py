import streamlit as st
from transformers import pipeline

st.set_page_config(page_title="Fake News Detector", page_icon="📰")

@st.cache_resource
def load_models():
    try:
        summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
        detector = pipeline("text-classification", model="mrm8488/bert-tiny-finetuned-fake-news-detection")
        return summarizer, detector
    except Exception as e:
        st.error("⚠️ Model loading failed. Please try again later.")
        st.write(e)
        return None, None

summarizer, detector = load_models()

st.title("📰 Fake News Detector for Students")
st.markdown("Analyze articles, assess credibility, and summarize them to prevent misinformation.")

article = st.text_area("Paste a news article or paragraph:", height=200)

if st.button("Analyze Article"):
    if article.strip() == "":
        st.warning("Please enter some text.")
    elif summarizer is None or detector is None:
        st.error("Model not loaded. Please refresh the page and try again.")
    else:
        with st.spinner("Analyzing..."):
            # Summarize
            summary = summarizer(article, max_length=120, min_length=30, do_sample=False)[0]['summary_text']
            # Detect
            result = detector(article)[0]
            label = result['label']
            score = result['score']

        st.subheader("🧾 Summary:")
        st.write(summary)

        st.subheader("🔍 Credibility Check:")
        if label.lower() == "fake":
            st.error(f"⚠️ This article appears **FAKE** with confidence {score:.2f}")
        else:
            st.success(f"✅ This article appears **REAL** with confidence {score:.2f}")
