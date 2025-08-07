from langchain_community.document_loaders import DirectoryLoader,PyPDFLoader

lodaer = DirectoryLoader(
    path="09_Doc_Loader/Pdfs",
    glob="*.pdf",
    loader_cls=PyPDFLoader
)

docs = lodaer.lazy_load()

for doc in docs:
    print(doc)