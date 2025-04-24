from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableParallel, RunnablePassthrough
from pydantic import BaseModel,Field


llm = HuggingFacePipeline.from_model_id(
        model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        task="text-generation"
)


model = ChatHuggingFace(llm = llm)

parser = StrOutputParser()

template1 = PromptTemplate(

    template = "Generate 5 facts on {topic}\n Only return 5 items. No metadata. No other text.",
    input_variables= ["topic"]
)

template2 = PromptTemplate(
    template= "Generate 5 questions on {topic}\n Only return 5 items. No metadata. No other text.",
    input_variables = ["topic"]
)
template3 = PromptTemplate(
    template= "Merge the following documents: Facts -> {facts} \n questions -> {questions}",
    input_variables= ["topic"]
)

# Now I create the chains


parallel_chain = RunnableParallel(
    {
        "facts" : template1 | model | parser,
        "questions" : template2 |model | parser
     }
)

merge_chain = template3 | model | parser

chain = parallel_chain | merge_chain

result = chain.invoke(input= {"topic" : "Black Hole"})

print(result)

#chain.get_graph().print_ascii()
