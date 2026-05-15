from langchain_google_genai import  ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from dotenv import load_dotenv
load_dotenv()
parser = JsonOutputParser()

chat_model = ChatGoogleGenerativeAI(model = "gemini-2.5-flash" , temperature = 0.7)
template = PromptTemplate(template = 'give me  a name of an fictional char and his age , city {format_instructions}' , input_variables= [] , partial_variables= { 'format_instructions' : parser.get_format_instructions()})
chain = template | chat_model | parser
result = chain.invoke({})
print(result)