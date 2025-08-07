from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv

load_dotenv()

model = ChatAnthropic(model="gpt-4",temperature=0.1,max_completion_tokens=10)

result = model.invoke("What is AI?")

print(result.content)
