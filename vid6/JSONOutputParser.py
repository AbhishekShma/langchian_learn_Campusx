from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from langchain_core.output_parsers import JsonOutputParser
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv


load_dotenv()


llm = HuggingFacePipeline.from_model_id(
        model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        task="text-generation"
)

model = ChatHuggingFace(llm = llm)


parser = JsonOutputParser()

prompt_template = PromptTemplate(
    template = "Generate a name, city, place for a fictional person \n {format_instruction} \n Only provide output, no other data",
    input_variables = [],
    partial_variables={"format_instruction" : parser.get_format_instructions()}
)


#prompt = prompt_template.format()

#print(prompt)

def get_after_phrase(phrase,text):
    if phrase in text:
        return text.rsplit(phrase, 1)[-1]
    else:
        return

chain = prompt_template | llm

result = chain.invoke({})

print(result)