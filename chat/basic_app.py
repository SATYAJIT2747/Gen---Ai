import streamlit as st
st.title("Paper Summarizer")
query = st.text_input("Enter your query:" , "enter prompt here")
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
load_dotenv()
chat_model = ChatGoogleGenerativeAI(model = "gemini-2.5-flash" , temperature = 1)
if st.button("Summarize"):
    result = chat_model.invoke(query)
    st.success(result.content)

    