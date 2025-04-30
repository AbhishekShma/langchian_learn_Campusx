from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.vectorstores import FAISS
from langchain.schema import Document
from groq import Groq
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace, HuggingFaceEmbeddings


embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-MiniLM-L6-v2"
)


llm_model = "Your model here"


documents = [
    Document(
        page_content="tests, tests, tests, tests, tests, tests, "

    ),
    Document(
        page_content="tests, tests, tests, tests, tests, tests, "

    ),
    Document(
        page_content="tests, tests, tests, tests, tests, tests, "

    ),
    Document(
        page_content="tests, tests, tests, tests, tests, tests, "

    )

]
vectorstore = FAISS.from_documents(
    documents= documents,
    embedding= embedding_model
)


retriever = MultiQueryRetriever.from_llm(llm = llm_model, retriever= vectorstore.as_retriever())

