from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline





llm = HuggingFacePipeline.from_model_id(
    model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation"
)

# Wrap in LangChain's LLM wrapper
model = ChatHuggingFace(llm=llm, temperature=0.1)

response = model.invoke("What is the capital of India?")

print(response.content)