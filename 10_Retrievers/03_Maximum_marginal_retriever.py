from langchain_huggingface import HuggingFaceEmbeddings,HuggingFaceEndpoint,ChatHuggingFace
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS

# Sample documents
docs = [
    Document(page_content="LangChain makes it easy to work with LLMs."),
    Document(page_content="LangChain is used to build LLM based applications."),
    Document(page_content="Chroma is used to store and search document embeddings."),
    Document(page_content="Embeddings are vector representations of text."),
    Document(page_content="MMR helps you get diverse results when doing similarity search."),
    Document(page_content="LangChain supports Chroma, FAISS, Pinecone, and more."),
]

embeddings = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2")

vectorStore = FAISS.from_documents(
    documents=docs,
    embedding=embeddings
)

retriever = vectorStore.as_retriever(search_type = 'similarity',search_kwargs = {"k" : 2})

query = "What is langchain?"

result = retriever.invoke(query)
print(result)

retriever = vectorStore.as_retriever(search_type = 'mmr',search_kwargs = {"k" : 2})
result = retriever.invoke(query)
print(result)