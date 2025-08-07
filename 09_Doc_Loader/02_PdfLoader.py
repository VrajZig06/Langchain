from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("09_Doc_Loader/Breeding Feature Scenarios (1).pdf")

docs = loader.load()

print(len(docs)) # 3
print(type(docs)) # List
print(docs[0]) # page_content and metadata
print(type(docs[0])) # --> <class 'langchain_core.documents.base.Document'>