import streamlit as st
import pandas as pd
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
nltk.download('stopwords')
nltk.download('wordnet')
from IMDB_train import text_purify

model = joblib.load('IMDB.pkl')
vectorizer = joblib.load('vectorizer.pkl')

st.title("IMDB Movie Review Sentiment Analysis")

st.write("Enter a movie review to predict its sentiment (positive or negative):")

review_input = st.text_area("Review Input", height=200)

if st.button("Predict Sentiment"):
    if review_input:
        # Preprocess the input review
        cleaned_review = text_purify(review_input)
        # Convert the cleaned review to a format suitable for the model
        review_vector = vectorizer.transform([cleaned_review])
        # Predict sentiment
        prediction = model.predict(review_vector)
        sentiment = "Positive" if prediction[0] == 1 else "Negative"
        st.success(f"The predicted sentiment is: {sentiment}")
    else:
        st.error("Please enter a review to analyze.")
else:
    st.write("Please enter a review to get the sentiment prediction.")
    
