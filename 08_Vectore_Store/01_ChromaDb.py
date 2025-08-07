from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace
from langchain_chroma import Chroma
from langchain.schema import Document
from langchain_huggingface import HuggingFaceEmbeddings



llm = HuggingFaceEndpoint(
  repo_id="Qwen/Qwen3-Coder-480B-A35B-Instruct",
  task="text-generation",
  temperature=1.8,
  max_new_tokens=10
)

embeddingModel = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2")
llmModel = ChatHuggingFace(llm=llm)

vector_store = Chroma(
    embedding_function=embeddingModel,
    persist_directory="my_chroma_db",
    collection_name="sample"
)

doc1 = Document(
    page_content="Hello I am Makwana Vraj",
    metadata = {"name" : "Hello"}
)

# Add Vectore to Database
# vector_store.add_documents([doc1])


# Fetched Database
print(vector_store.get(include=['embeddings','documents']))