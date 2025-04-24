from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv


load_dotenv()


from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

tokenizer = AutoTokenizer.from_pretrained(model_id)
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

model = AutoModelForCausalLM.from_pretrained(model_id)
model.resize_token_embeddings(len(tokenizer))  # only needed if fine-tuning

text_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    #device=-1,
    pad_token_id=tokenizer.pad_token_id
)

llm = HuggingFacePipeline(pipeline=text_pipeline)
model = ChatHuggingFace(llm=llm, temperature=0.1)

parser = StrOutputParser()

prompt1 = PromptTemplate(
    template = "Explain in detail about {topic}",
    input_variables= ["topic"]
)

prompt2 = PromptTemplate(
    template = "Summarise in 1 line the following: {text}",
    input_variables= ["text"]
)

chain = prompt1 | model | parser | prompt2 | model | parser

result = chain.invoke({"topic" : "black hole"})

print(result)



