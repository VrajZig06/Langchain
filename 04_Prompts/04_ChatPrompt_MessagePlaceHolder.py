# Message Placeholder is used to create Prompt that includes all previous Conversations

from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace
from langchain.prompts import ChatPromptTemplate,MessagesPlaceholder


llm = HuggingFaceEndpoint(
  repo_id="Qwen/Qwen3-Coder-480B-A35B-Instruct",
  temperature=0,
  task="text-generation",
  max_new_tokens=10
)

model = ChatHuggingFace(llm=llm)

prompt_templet = ChatPromptTemplate([
  ("system" , "You are a helpful customer support agent"),
  MessagesPlaceholder(variable_name="Chat_History"),
  ('human',"{query}")
])

Chat_History = []

# Load all Previous History of User
with open("04_Prompts\chat_history.txt") as f:
  data = f.readlines()
  Chat_History.extend(data)  # Means Enter Multiple Items in Chat_History List

prompt = prompt_templet.invoke({
  "Chat_History": Chat_History,
  "query" : "Where is my Refund?"
})

result = model.invoke(prompt)

print(result.content)

