from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001"
)

query = "What is machine learning?"

query_embedding = embeddings.embed_query(query)

print(len(query_embedding))
print(query_embedding[:5])   # first 5 values