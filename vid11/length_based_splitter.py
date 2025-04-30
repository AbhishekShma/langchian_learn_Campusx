from langchain.text_splitter import CharacterTextSplitter


splitter = CharacterTextSplitter(
    separator="\n\n",
    chunk_size = 1000,
    chunk_overlap= 100

)

