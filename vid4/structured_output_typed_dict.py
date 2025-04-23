from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace
from typing import TypedDict, Annotated
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFacePipeline.from_model_id(
    model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm, temperature=0.1)

class Review(TypedDict): # Inherits from Typed dict
    summary : Annotated[str, "Provide a brief summary of the review"]
    sentiment : Annotated[str, "Provide sentiment of the review, possitive, negative, neutral"]

#model = OpenAI()

result = model.with_structured_output(Review)

print(result['sentiment'])