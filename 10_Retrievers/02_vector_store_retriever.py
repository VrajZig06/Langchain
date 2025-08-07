from langchain_huggingface import HuggingFaceEmbeddings,HuggingFaceEndpoint,ChatHuggingFace
from langchain_core.documents import Document
from langchain_chroma.vectorstores import Chroma

document = [
    Document(page_content="Virat Kohli is a former Indian captain and one of the greatest modern batsmen. He is known for his aggressive style and consistency across all formats. Kohli is the highest run-scorer in IPL and among top in international cricket."),
    Document(page_content="Sachin Tendulkar is known as the God of Cricket and has over 34,000 international runs. He was the first player to score a double hundred in ODIs. Tendulkar represented India for 24 years and won the 2011 World Cup."),
    Document(page_content="MS Dhoni is India’s most successful captain, having led the team to all three major ICC trophies. He is known for calmness and finishing abilities. Dhoni also captained Chennai Super Kings to five IPL titles.")
]

embeddings = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2")

vectorStore = Chroma.from_documents(
    documents=document,
    embedding=embeddings
)

retriever = vectorStore.as_retriever(search_kwargs = {"k" : 1})

query = "Give me details of MS Dhoni"

result = retriever.invoke(query)
print(result)