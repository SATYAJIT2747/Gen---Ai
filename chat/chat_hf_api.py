from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-V4-Pro",
    task="text-generation"
   
)

model = ChatHuggingFace(llm=llm)

result = model.invoke(
    "what is the reason of india pakistan conflict? Answer in 150 words."
)

print(result.content)