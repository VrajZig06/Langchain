from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import PromptTemplate


llm = HuggingFaceEndpoint(
 repo_id = "Qwen/Qwen3-Coder-480B-A35B-Instruct",
  task="text-generation",
  max_new_tokens=50,
  temperature=0
)

model = ChatHuggingFace(llm=llm)

templet1 = PromptTemplate(
  template="""Describe this topic in 20 lines, Topic Name : {topic}""",
  input_variables=['topic']
) 

templet2 = PromptTemplate(
  template= """Summarize following text into 2 lines.\n {text}""",
  input_variables=['text']
)

parser = StrOutputParser()

chain = templet1 | model | parser | templet2 | model | parser

result = chain.invoke({"topic" : "AI"})

print(result)