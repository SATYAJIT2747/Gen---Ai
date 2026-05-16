from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableParallel
from dotenv import load_dotenv

load_dotenv()

# Parser
parser = StrOutputParser()

# Model
model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.7
)

# Prompt 1
prompt1 = PromptTemplate(
    template='Write short notes on topic: {topic}',
    input_variables=['topic']
)

# Prompt 2
prompt2 = PromptTemplate(
    template='Make 5 quiz questions based on topic: {topic}',
    input_variables=['topic']
)

# Prompt 3
prompt3 = PromptTemplate(
    template='''
Merge the following notes and quiz questions into a single formatted text.

Notes:
{notes}

Quiz:
{quiz}
''',
    input_variables=['notes', 'quiz']
)

# Parallel chain
parallel_chain = RunnableParallel(
    {
        'notes': prompt1 | model | parser,
        'quiz': prompt2 | model | parser
    }
)

# Final chain
chain = parallel_chain | prompt3 | model | parser

# Run
result = chain.invoke({
    'topic': "Logistic Regression"
})

print(result)
chain.get_graph().print_ascii()