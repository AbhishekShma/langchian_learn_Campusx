from langchain.vectorstores import Chroma
from langchain.schema import Document
vectorstore = Chroma(
    #embedding_function= put the embedding model here.
    persist_directory="chroma_db",
    collection_name="sample"
)



