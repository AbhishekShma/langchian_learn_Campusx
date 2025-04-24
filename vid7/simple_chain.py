from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel,Field


llm = HuggingFacePipeline.from_model_id(
        model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        task="text-generation"
)

model = ChatHuggingFace(llm = llm)


class Person(BaseModel):
    name : str = Field(description="name of the Person")
    age : int = Field(description="age of the Person")

parser = PydanticOutputParser(pydantic_object=Person)

template = PromptTemplate(
    template="Generate details of a fictional person \n {format_instructions}",
    input_variables=[],
    partial_variables={"format_instructions" : parser.get_format_instructions()}
)

#prompt = template.format()

#print(prompt)


#chain = template | model | parser

chain = template | model
result = chain.invoke({})

print(result)
