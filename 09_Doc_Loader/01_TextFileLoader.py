from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

loader = TextLoader('/Users/ztlab141/Desktop/Python/Langchain Models/09_Doc_Loader/01_Data.txt',encoding='utf-8')
docs = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size = 20,
    chunk_overlap = 0,
    separators=""
)

SplitData = splitter.split_documents(docs)

print(SplitData)
