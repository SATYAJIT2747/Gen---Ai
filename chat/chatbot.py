from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
load_dotenv()
chat_model = ChatGoogleGenerativeAI(model = "gemini-2.5-flash" , temperature = 0.7)
while True:
    query = input("You:")
    if(query.lower() != "exit"):
        result = chat_model.invoke(query)
        print("AI:" , result.content)
        
    