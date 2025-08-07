from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-1.5-pro",temperature=0.1,max_completion_tokens=10)

result = model.invoke("What is AI?")

print(result.content)
