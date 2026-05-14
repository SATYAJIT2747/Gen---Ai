from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
load_dotenv()
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001")
# Documents
documents = [
    "Machine learning is a subset of artificial intelligence.",
    "Cricket is a popular sport in India.",
    "Deep learning uses neural networks."
]

# User Query
query = "What is AI and machine learning?"

query_embed = embeddings.embed_query(query)
doc_embed = embeddings.embed_documents(documents)
scores = cosine_similarity([query_embed], doc_embed)[0]
doc_score_pairs = list(zip(documents, scores))
doc_score_pairs.sort(key = lambda x :x[1], reverse=True)
for doc, score in doc_score_pairs:
    print(f"Document: {doc}\nSimilarity Score: {score}\n")