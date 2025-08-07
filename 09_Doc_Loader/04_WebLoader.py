from langchain_community.document_loaders import WebBaseLoader

loader = WebBaseLoader(web_path="https://python.langchain.com/docs/integrations/document_loaders/web_base/")

docs = loader.load()
print(docs[0].page_content)