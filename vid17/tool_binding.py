from langchain_huggingface import HuggingFacePipeline,ChatHuggingFace
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFacePipeline.from_model_id(
        model_id="meta-llama/Llama-3.2-3B-Instruct",
        task="text-generation"
)

model = ChatHuggingFace(llm = llm)
result = llm.invoke("What is the capital of France?")

print(result)