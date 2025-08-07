from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import AIMessage,HumanMessage,SystemMessage
from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace

# used in Multiple Turn Conversations

llm = HuggingFaceEndpoint(
  repo_id = "Qwen/Qwen3-Coder-480B-A35B-Instruct",
  task = "text-generation",
  max_new_tokens = 10
)
model = ChatHuggingFace(llm = llm)

chat_templet = ChatPromptTemplate([
  ("system","You are a expert {domain_name}."),
  ("human","Explain in simple term {topic_name}")
])

prompt = chat_templet.invoke({
  "domain_name" : "Doctor",
  "topic_name" : "What is HeartAttack ?"
})

result = model.invoke(prompt)

print(result.content)


