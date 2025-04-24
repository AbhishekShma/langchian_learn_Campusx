from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv


load_dotenv()


llm = HuggingFacePipeline.from_model_id(
        model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        task="text-generation"
)

model = ChatHuggingFace(llm = llm)

schema = [
    ResponseSchema(name = "fact_1", description="Fact 1 about the topic"),
    ResponseSchema(name = "fact_2", description="Fact 2 about the topic"),
    ResponseSchema(name = "fact_3", description="Fact 3 about the topic"),

]

parser = StructuredOutputParser(response_schemas=schema)

prompt_template = PromptTemplate(
    template = "Generate 3 facts about {topic} \n {format_instruction}",
    input_variables = ["topic"],
    partial_variables={"format_instruction" : parser.get_format_instructions()}
)
#prompt = prompt_template.format(topic = "India")

#print(prompt)




def get_after_phrase(phrase,text):
    if phrase in text:
        return text.rsplit(phrase, 1)[-1]
    else:
        return

#chain = prompt_template | model | parser
chain = prompt_template | model
topic = "India"
result = chain.invoke(input={"topic" : topic})

print(result)


