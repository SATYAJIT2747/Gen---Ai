from langchain_google_genai import ChatGoogleGenerativeAI
from typing import Annotated , Optional
from pydantic import BaseModel
from dotenv import load_dotenv
load_dotenv()
class Review(BaseModel):
    keythemes: Annotated[list[str], "The key themes of the movie"]
    summary: Annotated[str, "A brief summary of the review"]
    sentiment: Annotated[str, "The sentiment of the review (positive/negative/neutral)"]
    pros : Optional[Annotated[list[str], "The positive aspects of the movie"]]
    cons: Optional[Annotated[list[str], "The negative aspects of the movie"]]
model = ChatGoogleGenerativeAI(model = "gemini-2.5-flash" , temperature = 0.7)
str_model = model.with_structured_output(Review)
review_text = """
The Shawshank Redemption is one of the most emotional and inspiring movies ever made. 
The story is beautifully written and keeps the audience deeply connected throughout the film. 
Tim Robbins and Morgan Freeman delivered outstanding performances, and their acting felt natural and powerful. 
The friendship between the characters was heartwarming and memorable.

The movie also delivers strong life lessons about hope, patience, and freedom. 
The background music perfectly matched the emotional scenes and enhanced the overall experience.

However, the movie feels slightly long in some parts, and a few scenes could have been shorter. 
The cinematography is decent but not visually extraordinary compared to modern films.

Overall, it is a masterpiece and definitely a must-watch movie for every cinema lover.
"""
output = str_model.invoke(review_text)
print(output)
#gemini-2.5-flash is not supporting structured output yet. so i will use pydantic to parse the output. 
