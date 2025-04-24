from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableBranch, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel,Field
from typing import Literal,Annotated

# models
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

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
    #device=-1,
    pad_token_id=tokenizer.pad_token_id
)

llm = HuggingFacePipeline(pipeline=text_pipeline)
model = ChatHuggingFace(llm=llm, temperature=0.1)


#parsers

class Review(BaseModel):
    sentiment :  Annotated[Literal["Positive","Negative"] , Field(description="Assign a positive or negative label to a review")]

parser_pydantic = PydanticOutputParser(pydantic_object=Review)
parser_string = StrOutputParser()

# Prompts
prompt1 = PromptTemplate(

    template="Classify the following feedback into the categories positive or negative.\n {feedback} \n {format_instructions}",
    input_variables=["feedback"],
    partial_variables={"format_instructions" : parser_pydantic.get_format_instructions()} 
)

prompt2 = PromptTemplate(

   prompt = "Write an appropritate response to the following positive feedback: \n {feedback} \n {format_instructions}" ,
   input_variables=["feedback"],
   partial_variables={"format_instructions"}
)
prompt3 = PromptTemplate(

   prompt = "Write an appropritate response to the following negative feedback: \n {feedback} \n {format_instructions}" ,
   input_variables=["feedback"],
   partial_variables={"format_instructions"}
)

#Chains
classifier_chain = prompt1 | model | parser_string


branch_chain = RunnableBranch(
    (lambda x : x =="Positive", prompt2 | model | parser_string),
    (lambda x : x == "Negative",prompt3 | model | parser_string),
    RunnableLambda(lambda x : "No sentiment found")

)


chain = classifier_chain | branch_chain