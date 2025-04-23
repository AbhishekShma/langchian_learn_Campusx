from langchain_huggingface import HuggingFaceEndpointEmbeddings
from sklearn.metrics.pairwise import cosine_similarity

embedding_model = HuggingFaceEndpointEmbeddings(
    model="sentence-transformers/all-MiniLM-L6-v2"
)

documents = [
    "The capital of India is  Delhi.",
    "The capital of France is Pris.",
    "The capital of Japan is Tkyo.",
]

document_embeddings = embedding_model.embed_documents(documents)

query = "What is the capital of France?"
query_embedding = embedding_model.embed_query(query)

cos_similarity = cosine_similarity([query_embedding], document_embeddings)



result = sorted(list(enumerate(cos_similarity[0])), key=lambda x: x[1],reverse=True)[0]
print(result)
print(query)
print(documents[result[0]])
print(f"Cosine Similarity: {result[1]}")