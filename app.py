import streamlit as st
import pandas as pd
import joblib
from IMDB_train import text_purify
import base64

def get_base64(file_path):
    with open(file_path, "rb") as f:
        return base64.b64encode(f.read()).decode()

img_base64 = get_base64("bg2.png")

st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url("data:image/png;base64,{img_base64}");
        background-size: cover;
        background-position: center;
    }}
    </style>
    """,
    unsafe_allow_html=True
)


model = joblib.load('IMDB.pkl')
vectorizer = joblib.load('vectorizer.pkl')

st.title("IMDB Movie Review Analysis")

tab1, tab2 = st.tabs(["Single Review", "Multiple Reviews"])

with tab1:
    st.write("Enter a movie review to predict its sentiment (positive or negative):")

    st.write("For example: This movie was fantastic! I loved the acting and the plot was engaging.")

    review_input = st.text_area("Review Input", height=100)

    if st.button("Predict Sentiment"):
        if review_input.strip():
            # Preprocess the input review
            cleaned_review = text_purify(review_input)
            # Convert the cleaned review to a format suitable for the model
            review_vector = vectorizer.transform([cleaned_review])
            # Predict sentiment
            prediction = model.predict(review_vector)
            probs = model.predict_proba(review_vector)[0]
            sentiment = "Positive" if prediction[0] == 1 else "Negative"
            st.success(f"The predicted sentiment is: {sentiment}")
            
            st.write(f"**Positive Probability:** {probs[1]*100:.2f}%")
            st.write(f"**Negative Probability:** {probs[0]*100:.2f}%")
        else:
            st.error("Please enter a review to analyze.")
    else:
        st.write("Please enter a review to get the sentiment prediction.")
with tab2:
    st.write("Upload a CSV file containing multiple movie reviews to predict their sentiments.")

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"], key="upload_csv")
    text_block = st.text_area("Or paste multiple reviews (one per line):", height=200, key="multi_input")

    reviews = []
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        if 'review' in df.columns:
            reviews = df['review'].dropna().tolist()
        else:
            st.error("CSV must contain a 'review' column.")

    elif text_block.strip():
        reviews = [line.strip() for line in text_block.split("\n") if line.strip()]

    if st.button("Predict All", key="predict_multi"):
        if reviews:
            cleaned_reviews = [text_purify(r) for r in reviews]
            review_vectors = vectorizer.transform(cleaned_reviews)
            predictions = model.predict(review_vectors)
            probs = model.predict_proba(review_vectors)
            results_df = pd.DataFrame({
                "Review": reviews,
                "Sentiment": ["Positive" if p == 1 else "Negative" for p in predictions],
                "Positive %": (probs[:,1]*100).round(2),
                "Negative %": (probs[:,0]*100).round(2)
            })
            st.dataframe(results_df)
            st.download_button(
                "Download Results as CSV",
                data=results_df.to_csv(index=False).encode("utf-8"),
                file_name="sentiment_results.csv",
                mime="text/csv"
            )
        else:
            st.error("Please provide reviews.")
