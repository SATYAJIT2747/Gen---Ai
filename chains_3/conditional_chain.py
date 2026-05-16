# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain_core.prompts import PromptTemplate
# from langchain_core.output_parsers import StrOutputParser , PydanticOutputParser
# from langchain.schema.runnable import RunnableParallel , RunnableLambda , RunnableBranch
# from pydantic import BaseModel , Field
# from dotenv import load_dotenv
# from typing import Literal
# load_dotenv()

# # Parser
# parser = StrOutputParser()
# class Feedback(BaseModel):
#     sentiment : Literal['positive' , 'negative'] = Field(description = "sentiment of the feedback either positive or negative")

# # Pydantic parser
# parser2 = PydanticOutputParser(pydantic_object= Feedback)

# # Model
# model = ChatGoogleGenerativeAI(
#     model="gemini-3.1-flash-lite",
#     temperature=0.7
# )
# # Prompt 1
# prompt1 = PromptTemplate(
#     template = 'classify the sentiment of the following feedback as positive , negative  : {feedback} {format_instructions}' , input_variables= ['feedback'] , partial_variables= { 'format_instructions' : parser2.get_format_instructions()}
    
# )

# prompt2 = PromptTemplate(template= 'write an appreciation message for the following positive feedback 2 lines  : {feedback}' , input_variables= ['feedback'])
# prompt3 = PromptTemplate(template= 'write a response to the following negative feedback  2 lines: {feedback}' , input_variables= ['feedback'])
# classifier_chain = prompt1 |model | parser2
# branch_chain = RunnableBranch((lambda x : x.sentiment == 'positive' , prompt2 |model | parser)
#                               ,
#                               (lambda x : x.sentiment == 'negative' , prompt3 |model | parser)
#                               ,RunnableLambda(lambda x : "neutral feedback received , no response needed")
#                               ) // branch chain is only getting sentiment not the feedback
# chain = classifier_chain | branch_chain

# result = chain.invoke({'feedback' : 'The application has a clean and attractive interface, and some features work really well, but the overall performance is inconsistent. Sometimes the system responds quickly, while at other times it becomes very slow and unresponsive. I also noticed a few bugs during usage, especially when switching between pages. Customer support was helpful and resolved some of my issues, but the experience could still be improved. Overall, the product has good potential, but it requires better optimization and stability updates to provide a smoother user experience.'
# })
# print(result



# earlier here feedback → classifier_chain → branch_chain .from classifier_chain we are getting sentiment  only but we also want to keep the original feedback for later use in the appreciation or response message. So we can use RunnableParallel to keep both sentiment and feedback and pass them to the branch chain. In the branch chain we can access both sentiment and feedback and generate appropriate response based on the sentiment.
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from langchain.schema.runnable import RunnableParallel, RunnableLambda, RunnableBranch
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from typing import Literal

load_dotenv()

# Output parser
parser = StrOutputParser()

# Pydantic model
class Feedback(BaseModel):
    sentiment: Literal['positive', 'negative'] = Field(
        description="sentiment of the feedback either positive or negative"
    )

# Pydantic parser
parser2 = PydanticOutputParser(pydantic_object=Feedback)

# Model
model = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite",
    temperature=0.7
)

# Prompt 1
prompt1 = PromptTemplate(
    template='''
classify the sentiment of the following feedback
as positive or negative:

{feedback}

{format_instructions}
''',
    input_variables=['feedback'],
    partial_variables={
        'format_instructions': parser2.get_format_instructions()
    }
)

# Response prompts
prompt2 = PromptTemplate(
    template='write an appreciation message for the following positive feedback in 2 lines: {feedback}',
    input_variables=['feedback']
)

prompt3 = PromptTemplate(
    template='write a response to the following negative feedback in 2 lines: {feedback}',
    input_variables=['feedback']
)

# Classifier chain
classifier_chain = prompt1 | model | parser2

# Keep BOTH sentiment and feedback
full_chain = RunnableParallel(
    sentiment=classifier_chain,
    feedback=RunnableLambda(lambda x: x["feedback"])
) # this will keep both sentiment and feedback in the output of full_chain

'''{ 
runnable parallel will give us output in the form of a dictionary 
   "sentiment": Feedback(sentiment='positive'),
   "feedback": "The app is amazing"
}'''
# Branching
branch_chain = RunnableBranch(
    (
        lambda x: x["sentiment"].sentiment == "positive",
        prompt2 | model | parser
    ),
    (
        lambda x: x["sentiment"].sentiment == "negative",
        prompt3 | model | parser
    ),
    RunnableLambda(lambda x: "neutral feedback received")
)

# Final chain
chain = full_chain | branch_chain

# Invoke
result = chain.invoke({
    "feedback": "The application has bugs and performance issues."
})

print(result)