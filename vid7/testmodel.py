from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate       
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableBranch, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel,Field
from typing import Literal,Annotated
from langchain_core.messages import HumanMessage,AIMessage,SystemMessage
from langchain.memory.buffer_window import ConversationBufferWindowMemory



#model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
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
    pad_token_id=tokenizer.pad_token_id,
    max_length=1024,
    temperature=0.7,
    truncation=True

)

llm = HuggingFacePipeline(pipeline=text_pipeline)
model = ChatHuggingFace(llm=llm)
chat_history = ConversationBufferWindowMemory(
    memory_key="chat_history",
    k=5,
    return_messages=True,
    max_token_limit=4096,  # Adjust this value as needed
)
chat_template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("human", "{input}")
]
)

messages = chat_history.chat_memory.messages
user_input = "What is the capital of France?"
chat_history.chat_memory.add_user_message(user_input)
prompt = chat_template.format(input= user_input, chat_history= messages) 
response = model.invoke(prompt)

print(response.content)