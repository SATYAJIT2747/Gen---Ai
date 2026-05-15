# from langchain_huggingface import HuggingFaceEndpoint , ChatHuggingFace
from langchain_google_genai import  ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
load_dotenv()
parser = StrOutputParser()
# llm  = HuggingFaceEndpoint(repo_id="deepseek-ai/DeepSeek-V4-Pro", task="text-generation")
# model = ChatHuggingFace(llm=llm)
chat_model = ChatGoogleGenerativeAI(model = "gemini-2.5-flash" , temperature = 0.7)
template1 = PromptTemplate(template = "write a detailed report on {topic} in 500 words" , input_variables= ["topic"])
template2 = PromptTemplate(template = "summarize the following text in 100 words: {text}" , input_variables= ["text"])
chain = template1 | chat_model | parser | template2 | chat_model | parser
result = chain.invoke({'topic' :"black holes"})
print(result)