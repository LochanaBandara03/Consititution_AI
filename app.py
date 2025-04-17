# app.py

import streamlit as st
from main import get_answer  # Importing the backend function from main.py

# Streamlit UI
st.set_page_config(page_title="Sri Lanka Legal AI Assistant", layout="wide")
st.title("📘 Sri Lanka Constitution Legal AI Assistant")

# User input for the legal question
question = st.text_area("Enter your legal question:", height=100)

if st.button("Get Answer") and question.strip():
    with st.spinner("Searching the Constitution..."):
        # Get the answer and legal context from the AI model
        result, legal_text = get_answer(question)

    # Display the AI's answer
    st.subheader("🔎 Answer:")
    st.markdown(result)

    # Optionally, show the retrieved legal context in an expandable section
    with st.expander("📄 View Retrieved Legal Context"):
        st.write(legal_text)
