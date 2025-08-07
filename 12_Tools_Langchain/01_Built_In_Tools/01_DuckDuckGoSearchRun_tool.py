from langchain_community.tools import DuckDuckGoSearchRun
import re
search_tool = DuckDuckGoSearchRun()

searchQuery = input("Enter Your SearchQuery Here :- ")

result = search_tool.invoke(searchQuery)

print(result)