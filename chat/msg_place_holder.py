from langchain_core.prompts import ChatPromptTemplate , MessagesPlaceholder
chat_templ = ChatPromptTemplate([
    MessagesPlaceholder(variable_name="chat_history") , 
    ('human' , '{query}')
])
chat_history = []
with open("history.txt" , "r") as f:
    chat_history.extend(f.readlines())
prompt = chat_templ.invoke({'chat_history': chat_history , 'query': "What is the capital of France?"})
print(prompt)
