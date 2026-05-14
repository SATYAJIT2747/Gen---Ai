from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
load_dotenv()
chat_model = ChatGoogleGenerativeAI(model= "gemini-2.5-flash" , temperature= 1 , max_tokens= 256)

result = chat_model.invoke("what is the reason of india pakistan conflict? Answer in 150 words.")
print(result.content)