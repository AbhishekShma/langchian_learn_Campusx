from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace
from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

model_id = "meta-llama/Llama-3.2-3B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_id)
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

model = AutoModelForCausalLM.from_pretrained(model_id)
model.resize_token_embeddings(len(tokenizer))  # only needed if fine-tuning

text_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device=-1,
    pad_token_id=tokenizer.pad_token_id
)

llm = HuggingFacePipeline(pipeline=text_pipeline)
model = ChatHuggingFace(llm=llm, temperature=0.1)
load_dotenv()


chat = ChatPromptTemplate(
[
    ("system", "You are a helpful {domain} assistant"),
    ("human", "{input}")
]

)



prompt = chat.invoke({"domain": "general", "input": "What is the capital of France?"})


while True:
    user_input = input("You:")
    if user_input.lower() == "exit":
        break
    
    response = model.invoke(user_input)
    print(f"Assistant: {response.content}") 
