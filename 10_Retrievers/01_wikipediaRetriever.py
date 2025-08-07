from langchain_community.retrievers import WikipediaRetriever

retriever = WikipediaRetriever(lang='en',top_k_results=2)

query = "Give me details of India and Pakistan Border?"

docs = retriever.invoke(query)

print(docs)