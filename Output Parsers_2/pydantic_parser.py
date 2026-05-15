from langchain_google_genai import  ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser 
from pydantic import BaseModel , Field
from dotenv import load_dotenv
load_dotenv()
class Person(BaseModel):
    name :str = Field(description = "name of the person")
    age :int = Field(description = "age of the person" , ge = 0 , le = 120)
    city : str = Field(description = "city where the person lives ")
parser = PydanticOutputParser(pydantic_object=Person)
chat_model = ChatGoogleGenerativeAI(model = "gemini-2.5-flash" , temperature = 0.7)
template = PromptTemplate(template = "give me a name of a fictonal character and his age and {city} {format_instructions}" , input_variables= ['city'] , partial_variables= { 'format_instructions' : parser.get_format_instructions()})
chain = template |chat_model | parser
result = chain.invoke({'city' : 'new york'})
print(result)