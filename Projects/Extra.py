from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen3-235B-A22B-Thinking-2507",
    task="text-generation",
    max_new_tokens=100)

model = ChatHuggingFace(llm = llm)

result = model.invoke("What is the Price of Tata Steel Stock Price today?")

print(result.content)