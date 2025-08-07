from langchain.tools import tool
from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from langchain_core.messages import AIMessage,HumanMessage,ToolMessage
# Tool Binding : Tool Binding is the process of registering all tools with LLM Model.
# So that LLM Knows What tools LLM has,LLM can take description from that tool.


# Note : The LLM does not execute Actual Tool but it suggest that tool and make one Schema related to that tool. Langchain or Programmers needs to take responsibilities to run that tool when LLM suggests in "tool_calls" Attributes.

history = []

@tool
def Multiply(a:int,b:int) -> int:
    """ This is Function for Two Numbers Multiplications. """
    return a*b

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen3-Coder-480B-A35B-Instruct",
    task="text-generation",
    temperature=1.8,
    max_new_tokens=10
)

model = ChatHuggingFace(llm=llm)

# Tool is Bind With Model (LLM Model)
llm_with_tools = model.bind_tools([Multiply])

query = input("What is your query? :- ")
history.append(query)

# Tool Suggestion By LLM
result = llm_with_tools.invoke(query)
history.append(result)

if len(result.tool_calls) > 0:
    # Tool Execution : here we use Actual Tool and Args that provides by LLM to execute that tool.
    ans = Multiply.invoke(result.tool_calls[0])
    history.append(ans)

    # Re-Call LLM for this History
    final_result = llm_with_tools.invoke(history)
    print(final_result.content)
else:
    print(result.content)



