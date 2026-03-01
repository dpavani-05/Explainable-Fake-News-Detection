
import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
st.set_page_config(page_title="Explainable Fake News Detector", layout="centered")
st.title("📰 Explainable Fake News Detection")
st.markdown("Enter a full news article (minimum 20 words) to classify it.")
model = joblib.load("src/model.pkl")
vectorizer = joblib.load("src/vectorizer.pkl")

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    words = text.split()
    words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
    return " ".join(words)

st.title("Explainable Fake News Detector")

user_input = st.text_area("Enter News Text")

if st.button("Predict"):
    cleaned = clean_text(user_input)
    if len(user_input.split()) < 20:
        st.warning("Please enter a full news article (at least 20 words).")
        st.stop()
    vect = vectorizer.transform([cleaned])
    prediction = model.predict(vect)[0]
    prob = model.predict_proba(vect).max()
    feature_names = vectorizer.get_feature_names_out()
    coefs = model.coef_[0]
    # Calculate word contribution
    word_contributions = vect.toarray()[0] * coefs
    # Get top positive and negative words
    top_fake_indices = word_contributions.argsort()[-5:]
    top_real_indices = word_contributions.argsort()[:5]
    st.subheader("🔍 Explainability")
    st.write("### Words pushing towards FAKE:")
    for idx in reversed(top_fake_indices):
        if word_contributions[idx] != 0:
            st.write(f"{feature_names[idx]} ({round(word_contributions[idx], 4)})")
    st.write("### Words pushing towards REAL:")
    for idx in top_real_indices:
        if word_contributions[idx] != 0:
            st.write(f"{feature_names[idx]} ({round(word_contributions[idx], 4)})")
    if prediction == 1:
        st.error("Fake News")
    else:
        st.success("Real News")
    st.write("Confidence:", round(prob * 100, 2), "%")