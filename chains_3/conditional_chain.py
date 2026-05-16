from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser , PydanticOutputParser
from langchain.schema.runnable import RunnableParallel , RunnableLambda , RunnableBranch
from pydantic import BaseModel , Field
from dotenv import load_dotenv
from typing import Literal
load_dotenv()

# Parser
parser = StrOutputParser()
class Feedback(BaseModel):
    sentiment : Literal['positive' , 'negative'] = Field(description = "sentiment of the feedback either positive or negative")

# Pydantic parser
parser2 = PydanticOutputParser(pydantic_object= Feedback)

# Model
model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.7
)
# Prompt 1
prompt1 = PromptTemplate(
    template = 'classify the sentiment of the following feedback as positive , negative  : {feedback} {format_instructions}' , input_variables= ['feedback'] , partial_variables= { 'format_instructions' : parser2.get_format_instructions()}
    
)

prompt2 = PromptTemplate(template= 'write an appreciation message for the following positive feedback : {feedback}' , input_variables= ['feedback'])
prompt3 = PromptTemplate(template= 'write a response to the following negative feedback : {feedback}' , input_variables= ['feedback'])
classifier_chain = prompt1 |model | parser2
branch_chain = RunnableBranch((lambda x : x.sentiment == 'positive' , prompt2 |model | parser)
                              ,
                              (lambda x : x.sentiment == 'negative' , prompt3 |model | parser)
                              ,RunnableLambda(lambda x : "neutral feedback received , no response needed")
                              )
chain = classifier_chain | branch_chain

result = chain.invoke({'feedback' : 'The application has a clean and attractive interface, and some features work really well, but the overall performance is inconsistent. Sometimes the system responds quickly, while at other times it becomes very slow and unresponsive. I also noticed a few bugs during usage, especially when switching between pages. Customer support was helpful and resolved some of my issues, but the experience could still be improved. Overall, the product has good potential, but it requires better optimization and stability updates to provide a smoother user experience.'
})
print(result)