from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv


load_dotenv()


llm= HuggingFaceEndpoint(
    repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation",
    model_kwargs={"tokenizer.pad_token_id": " "}
)

model = ChatHuggingFace(llm=llm, temperature=0.1)
response = model.invoke("What is the capital of France?")  

print(response)
