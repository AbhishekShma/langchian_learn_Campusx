from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
import streamlit as st

llm = HuggingFacePipeline.from_model_id(
    model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation"
)


model = ChatHuggingFace(llm=llm, temperature=0.1)

st.title("Summarize paper")

st.header("Summarizer")

paper_input = st.selectbox("Select paper name", ["Attention is all you need"])

style_input = st.selectbox("Select style", ["Formal", "Informal"])

length_input = st.selectbox("Select length", ["Short", "Medium", "Long"])

template = PromptTemplate(
    template="""
You are a helpful assistant that summarizes papers. You will be given a paper and you will summarize it in a {style} style. The summary should be {length} and should include the main points of the paper. The paper is: {paper}.
""",
imput_variables=["paper", "style", "length"]
)

prompt = template.invoke(dict(paper=paper_input, style=style_input, length=length_input))
if "summary" not in st.session_state:   
    st.session_state["summary"] = ""
st.button("Summarize", on_click=lambda: st.session_state.update({"summary": model.invoke(prompt).content}))

st.write("Summary:")
st.write(st.session_state["summary"])

load_dotenv()
