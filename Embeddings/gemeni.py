import google.generativeai as genai
from dotenv import load_dotenv
import os

load_dotenv()

genai.configure(
    api_key=os.getenv("Google_api_key")
)

result = genai.embed_content(
    model="models/gemini-embedding-001",
    content="What is machine learning?"
)

embedding = result["embedding"]

print(len(embedding))