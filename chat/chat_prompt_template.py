from langchain_core.prompts import ChatPromptTemplate
chat_templ = ChatPromptTemplate([
    ("system", "You are a helpful assistant that summarizes research papers."),
    ("user", "Summarize the paper {paper} in a {summary_type} with a {style} style and a length of {length}.")
])
prompt = chat_templ.invoke({'paper': "BERT", 'summary_type': "Short summary", 'style': "Formal", 'length': "200 words"})
print(prompt)