import streamlit as st
import pandas as pd
import string

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# Load dataset
df = pd.read_csv("customer_support_tickets.csv")
st.write(df.columns)

# Clean text
def clean_text(text):
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text

df["clean_text"] = df["Ticket Description"].apply(clean_text)

# Train model inside app
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df["clean_text"])
y = df["Ticket Type"]

model = MultinomialNB()
model.fit(X, y)

# UI
st.title("🎫 Support Ticket Classifier")

ticket = st.text_area("Enter your issue:")

if st.button("Predict"):
    clean = clean_text(ticket)
    vec = vectorizer.transform([clean])
    prediction = model.predict(vec)

    st.success(f"Category: {prediction[0]}")
