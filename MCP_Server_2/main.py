from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace

import asyncio

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen3-Coder-480B-A35B-Instruct",
    task="text-generation",
    max_new_tokens=100)

async def main():
    client = MultiServerMCPClient(
        {
            "math" : {
                "command" : "python3",
                "args" : ['MCP_Server_2/MathMCP.py'],
                "transport" : "stdio"
            },
            # "currency" : {
            #     "url" : "http://127.0.0.1:8000/mcp",
            #     "transport" : "streamable_http"
            # },
            "latestNews" : {
                "url" : "http://127.0.0.1:8000/mcp",
                "transport" : "streamable_http"
            }
        }
    )

    tools = await client.get_tools()
    model = ChatHuggingFace(llm=llm)

    agent = create_react_agent(
        model,tools
    )

    #  -------- Agent 1 ---------
    # response = await agent.ainvoke(
    #     {
    #         "messages" : [{"role" : 'user',"content" :"what's (2+3) * 4?"}],
    #         "tool_calls" : True,
    #         "tool_choice" : "auto",
    #     }
    # )

    # print("Math Response",response['messages'][-1].content)

    #  -------- Agent 2 ---------
    # response = await agent.ainvoke({
    #         "messages" : [{"role" : 'user',"content" :"convert currency 10 USD to INR"}],
    #         "tool_calls" : True,
    #         "tool_choice" : "auto",
    #     })
    
    # print("Currency Response",response['messages'][-1].content)

    #  -------- Agent 3 ---------
    response = await agent.ainvoke(
        {
            "messages" : [{"role" : 'user',"content" :"Artificial Intelligence"}],
            "tool_calls" : True,
            "tool_choice" : "auto",
        }
    )

    print("Wikipedia Response",response['messages'][-1].content)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        print("Error : ",e)
