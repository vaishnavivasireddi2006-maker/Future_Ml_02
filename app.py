import streamlit as st
import pandas as pd
import string
import re

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return text

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# Load dataset
df = pd.read_csv("customer_support_tickets.csv")
st.write(df["Ticket Type"].value_counts())   # 👈 check imbalance

# Clean text
def clean_text(text):
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text

df["clean_text"] = df["Ticket Description"].apply(clean_text)

# Train model inside app
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X = vectorizer.fit_transform(df["clean_text"])

y = df["Ticket Type"]

model = LogisticRegression(max_iter=1000)
model.fit(X, y)
# UI
st.title("🎫 Support Ticket Classifier")

ticket = st.text_area("Enter your issue:")

if st.button("Predict"):
    clean = clean_text(ticket)
    vec = vectorizer.transform([clean])
    prediction = model.predict(vec)

    st.success(f"Category: {prediction[0]}")
