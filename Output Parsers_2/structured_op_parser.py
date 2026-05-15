from langchain_google_genai import  ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain.output_parsers import StructuredOutputParser , ResponseSchema
from dotenv import load_dotenv
load_dotenv()
schema = [
    ResponseSchema(name = 'fact1' , description = 'the first fact about black hole'),
    ResponseSchema(name = 'fact2' , description = 'the second fact about black hole '),
    ResponseSchema(name = 'fact3' , description = 'the third fact about black hole'),
    ResponseSchema(name = 'fact4' , description = 'the fourth fact about black hole'),
    ResponseSchema(name = 'fact5' , description = 'the fifth fact about black hole')
]
parser = StructuredOutputParser.from_response_schemas(schema)

chat_model = ChatGoogleGenerativeAI(model = "gemini-2.5-flash" , temperature = 0.7)
template = PromptTemplate(template = 'give me  5 factes about {topic} {format_instructions}' , input_variables= ['topic'] , partial_variables= { 'format_instructions' : parser.get_format_instructions()})
chain = template | chat_model | parser
result = chain.invoke({'topic': 'black hole'})
print(result)