from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
load_dotenv()
chat_model = ChatGoogleGenerativeAI(model = "gemini-2.5-flash" , temperature = 0.7)
chat_history = []
while True:
    query = input("You:")
    chat_history.append({"role":"user" , "content": query})
    if(query.lower() != "exit"):
        result = chat_model.invoke(chat_history)
        chat_history.append({"role":"assistant" , "content": result.content})   
        print("AI:" , result.content)
    
        
    