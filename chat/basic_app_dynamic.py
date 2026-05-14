import streamlit as st

# query = st.text_input("Enter your query:" , "enter prompt here")
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
load_dotenv()
chat_model = ChatGoogleGenerativeAI(model = "gemini-2.5-flash" , temperature = 1)
st.title("Paper Summarizer")

paper = st.selectbox("Select the paper you want to summarize:", ["word2vec", "BERT", "GPT-3" , "T5", "RoBERTa"])
summary_type = st.selectbox("Select the type of summary you want:", ["Short summary", "Detailed summary", "Key points"])
style = st.selectbox("select the style of summary you want:", ["Formal", "Informal", "Technical", "Layman"] )
length = st.selectbox("Select the length of the summary:", ["100 words", "200 words", "300 words", "500 words"])
template  = PromptTemplate.from_template(
    "Summarize the paper {paper} in a {summary_type} with a {style} style and a length of {length}."
) 
# template obj is not string and we want str type query 

if st.button("Summarize"):
    query = template.format(paper=paper, summary_type=summary_type, style=style, length=length)
    result = chat_model.invoke(query)
    st.success(result.content)

    