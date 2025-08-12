from langchain_community.document_loaders import TextLoader,WebBaseLoader
from langchain_huggingface import HuggingFaceEmbeddings,HuggingFaceEndpoint,ChatHuggingFace
import bs4

# Embedding Model 
embeddings = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2")

# Text Loader to read .txt file 
# loader = TextLoader("RAG2/speech.txt")
# data = loader.load()
# print(data)

# WebBaseLoader to read -- load,chunk and content of HTML Page

loader = WebBaseLoader(web_path="https://stackoverflow.com/questions")
data = loader.load()
print(data)


