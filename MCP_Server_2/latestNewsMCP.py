from fastmcp import FastMCP
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper

MCPServer = FastMCP("Latest News")

api_wrapper = WikipediaAPIWrapper()

@MCPServer.tool()
async def latest_news(topic:str) -> str:
    tool = WikipediaQueryRun(
        name="Latest News Provider",
        description="this is a tool that will give you latest news using wikipedia tool",
        api_wrapper=api_wrapper)

    response = tool.run(topic)
    return response

if __name__ == "__main__":
    MCPServer.run(transport="streamable-http",host="127.0.0.1", port=8000)