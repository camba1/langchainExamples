import streamlit as st
from langchain.llms import OpenAI

st.title("Simple Chatter")

OPENAI_API_KEY = st.sidebar.text_input("OpenAI API Key", type="password")


def generate_response(prompt):
    llm = OpenAI(temperature=0.6, openai_api_key=OPENAI_API_KEY)
    st.info(llm(prompt))


with st.form("chat_form"):
    message = st.text_area("Enter your text", "What are three qualities of good developers")
    submit_button = st.form_submit_button("Submit")
    if not OPENAI_API_KEY.startswith("sk-"):
        st.warning("Please enter a valid OpenAI API key", icon="⚠️")
    if submit_button and OPENAI_API_KEY.startswith("sk-"):
        generate_response(message)
