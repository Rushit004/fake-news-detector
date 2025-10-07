import streamlit as st
from transformers import pipeline

# Cache model loading for faster runs
@st.cache_resource
def load_models():
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    detector = pipeline("text-classification", model="mohameddhiaa/fake-news-classifier")
    return summarizer, detector

summarizer, detector = load_models()

st.title("📰 Fake News Detector for Students")
st.markdown("Paste a news article and instantly check its credibility.")

article = st.text_area("Enter or paste your article:", height=250)

if st.button("Analyze Article"):
    if len(article.strip()) < 50:
        st.warning("Please provide a longer text for accurate analysis.")
    else:
        with st.spinner("Analyzing..."):
            summary = summarizer(article, max_length=120, min_length=40, do_sample=False)[0]['summary_text']
            result = detector(article)[0]
            label = result['label']
            score = round(result['score'], 3)

            if label.lower() == "fake":
                advice = "⚠️ This article may be unreliable. Cross-check with official sources."
            else:
                advice = "✅ Appears credible — but always double-check facts."

        st.subheader("🧩 Summary")
        st.write(summary)
        st.subheader("🔍 Credibility Result")
        st.write(f"**{label} ({score})**")
        st.subheader("📘 Advice")
        st.success(advice)
