from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate       
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableBranch, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel,Field
from typing import Literal,Annotated
from langchain_core.messages import HumanMessage,AIMessage,SystemMessage

# models
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

#model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
model_id = "meta-llama/Llama-3.2-3B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_id)


print(tokenizer.chat_template is not None)


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


#parsers

class Review(BaseModel):
    sentiment :  Annotated[Literal["Positive","Negative"] , Field(description="Assign a positive or negative label to a review")]
    feedback : Annotated[str, Field(description="The feedback provided by the user")]


parser_pydantic = PydanticOutputParser(pydantic_object=Review)
parser_string = StrOutputParser()

# Prompts

def format_prompt(user_message):
    return f"<|user|>\n{user_message}\n<|assistant|>\n"


prompt1 = PromptTemplate(

template=
    (
    "<|user|>\n"
    "You are a strict JSON formatter.\n"
    "Do not include any other text in the response.\n"
    "Classify the following feedback as either 'Positive' or 'Negative'.\n"
    "Return the result with both the sentiment and the feedback included.\n"
    "Feedback:\n{feedback}\n"
    "Feedback Ends Here\n"
    "Return the result in the following format:\n"
    "{format_instructions}\n"
    "<|assistant|>\n"
    ),        
    input_variables=["feedback"],
    partial_variables={"format_instructions" : parser_pydantic.get_format_instructions()}   
)

#print(prompt1.format(feedback="I love this product, it works great!"))

prompt2 = PromptTemplate(

template = "<|user|>\nWrite an appropritate response to the following positive feedback: \n {feedback}\n<|assistant|>\n" ,
input_variables=["feedback"]
)
#print(prompt2.format(feedback="I love this product, it works great!"))

prompt3 = PromptTemplate(

template = "<|user|>\nWrite an apropritate response to the following negative feedback: \n {feedback}\n<|assistant|>\n" ,
input_variables=["feedback"]
)

#Clean llm output

def split_output_string(string):

    result_of_split= string.rsplit("<|end_header_id|>",-1)

    return result_of_split[-1]


def format_llm_output(result : str):
    result = split_output_string(result)
    result = result.replace("\n","")
    result.replace("`", "")
    return result

#print(format_llm_output("Hello <|end_header_id|>"))

formatting_runnable = RunnableLambda(format_llm_output)
#Chains
classifier_chain = prompt1 | model | parser_string |formatting_runnable| parser_string

result = classifier_chain.invoke({"feedback" : "I love this product, it works great!"})
#print(result)



# result = split_output_string(result)
# result = result.replace("\n","")
# result.replace("`", "")
print(result)
'''
branch_chain = RunnableBranch(
    (lambda x : x.sentiment =="Positive", prompt2 | model | parser_string),
    (lambda x : x.sentiment == "Negative",prompt3 | model | parser_string),
    RunnableLambda(lambda x : "No sentiment found")

)


chain = classifier_chain | branch_chain

result = chain.invoke({"feedback" : "I love this product, it works great!"})
print(result)
'''





