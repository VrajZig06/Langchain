from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace

import asyncio

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen3-Coder-480B-A35B-Instruct",
    task="text-generation",
    max_new_tokens=100)


def separator(number:int,data):
    print(f"\n-------------- Response {number} --------------\n")
    print(data)
    print(f"\n-------------- ************** --------------\n")

async def main():
    client = MultiServerMCPClient(
        {
            "math" : {
                "command" : "python3",
                "args" : ['MCP_Server_2/MathMCP.py'],
                "transport" : "stdio"
            },
            "currency" : {
                "url" : "http://127.0.0.1:9000/mcp",
                "transport" : "streamable_http"
            },
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
    response = await agent.ainvoke(
        {
            "messages" : [{"role" : 'user',"content" :"what's (2+3) * 4?"}],
            "tool_calls" : True,
            "tool_choice" : "auto",
        }
    )

    data = response['messages'][-1].content
    separator(1,data)

    #  -------- Agent 2 ---------
    response = await agent.ainvoke({
            "messages" : [{"role" : 'user',"content" :"convert currency amount 1 from USD to INR"}],
            "tool_calls" : True,
            "tool_choice" : "auto",
        })
    
    data = response['messages'][-1].content
    separator(2,data)

    #  -------- Agent 3 ---------
    response = await agent.ainvoke(
        {
            "messages" : [{"role" : 'user',"content" :"Tell me About Next future tech in World."}],
            "tool_calls" : True,
            "tool_choice" : "auto",
        }
    )

    data = response['messages'][-1].content
    separator(3,data)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        print("Error : ",e)
