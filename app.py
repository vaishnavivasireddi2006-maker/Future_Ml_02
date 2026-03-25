import streamlit as st
import pickle
import string

# Page configuration
st.set_page_config(
    page_title="AI Support Ticket Classifier",
    page_icon="🎫",
    layout="centered"
)

# Custom CSS for colors
st.markdown("""
<style>
.main {
    background-color: #f5f7fa;
}

.title {
    text-align: center;
    color: #2c3e50;
}

.result-box {
    background-color: #d4edda;
    padding: 20px;
    border-radius: 10px;
    font-size: 20px;
    color: #155724;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

# Load ML model
model = pickle.load(open("ticket_model.pkl","rb"))
vectorizer = pickle.load(open("vectorizer.pkl","rb"))

# Header
st.markdown("<h1 class='title'>🎫 AI Support Ticket Classification</h1>", unsafe_allow_html=True)

st.write(
"""
This AI system automatically categorizes customer support tickets using
**Machine Learning and NLP**.
"""
)

# Ticket input
st.subheader("✍️ Enter Ticket Description")

ticket = st.text_area(
    "",
    placeholder="Example: I cannot login to my account and password reset is not working"
)

# Example tickets
st.subheader("💡 Example Issues")

col1, col2 = st.columns(2)

with col1:
    st.info("💳 Billing Issue\n\nMy credit card was charged twice")

with col2:
    st.info("🖥 Technical Issue\n\nThe application crashes when I open it")

# Text cleaning
def clean_text(text):
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text

# Prediction
if st.button("🔍 Predict Ticket Category"):

    if ticket.strip() == "":
        st.warning("⚠️ Please enter a ticket description")

    else:
        clean_ticket = clean_text(ticket)

        ticket_vector = vectorizer.transform([clean_ticket])

        prediction = model.predict(ticket_vector)

        st.markdown(
            f"<div class='result-box'>✅ Predicted Category: <b>{prediction[0]}</b></div>",
            unsafe_allow_html=True
        )

# Footer
st.write("---")
st.caption("Built with Python • Scikit-learn • Streamlit")
