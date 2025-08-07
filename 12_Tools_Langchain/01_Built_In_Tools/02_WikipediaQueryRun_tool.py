from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper

# Create a Wikipedia API wrapper
wiki_api = WikipediaAPIWrapper()

search_engine = WikipediaQueryRun(api_wrapper=wiki_api)

query = input("Enter your Search Query Here :- ")

result = search_engine.invoke(query)

print(result)
