from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
parser = StrOutputParser()
load_dotenv()
model = ChatGoogleGenerativeAI(model = "gemini-2.5-flash" , temperature = 0.7)
prompt1 = PromptTemplate(template= 'write a detailed report on {topic} in 500 words' , input_variables=['topic'])
prompt2 = PromptTemplate(template= 'give me 5 interesting facts based on {text}' , input_variables=['text'])
output_parser = StrOutputParser()
chain = prompt1|model|parser|prompt2|model|parser
result = chain.invoke({'topic':"Recent advancements in AI"})
print(result)