from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage 
from langchain.memory.buffer_window import ConversationBufferWindowMemory
from dotenv import load_dotenv
from typing import TypedDict, Annotated
import asyncio
import re

load_dotenv()

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


chat_history = ConversationBufferWindowMemory(
    memory_key="chat_history",
    k=5,
    return_messages=True,
    max_token_limit=4096,  # Adjust this value as needed
)

chat_template = ChatPromptTemplate(
    [
        ("system", "You are a helpful {domain} assistant"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}")
    ]
)

class Response(TypedDict): # Inherits from Typed dict
    Response : Annotated[str, "The main assistant reply here."]
    Metadata : Annotated[str, """The metadata such as: 
Cutting Knowledge Date: December 2023
Today Date: 23 Apr 2025 
<|eot_id|><|start_header_id|>  <|end_header_id|>                       
 also, any history                        
                         """]

structured_model = model.with_structured_output(Response)

# For cleaning the response of Llama


def extract_last_assistant_response(raw_output):
    pattern = r'<\|start_header_id\|>assistant<\|end_header_id\|>(.*?)(?:<\|eot_id\|>|$)'
    matches = re.findall(pattern, raw_output, re.DOTALL)
    
    if matches:
        last_response = matches[-1].strip()
        return last_response
    else:
        return "No assistant response found."
    

chat_history.chat_memory.add_ai_message("Hello, how can I help you today?")   

messages = chat_history.chat_memory.messages

while True:
    user_input = input("You: ")
    chat_history.chat_memory.add_user_message(user_input)
    prompt = chat_template.invoke({"domain": "general", "input": user_input, "chat_history": messages})   
    if user_input.lower() == "exit":
        break
    response = model.invoke(prompt)
    response = structured_model.invoke(prompt)
    
    print(type(response))
    #chat_history.chat_memory.add_ai_message(response.content)
    #print(f"Assistant: {response.content}")



