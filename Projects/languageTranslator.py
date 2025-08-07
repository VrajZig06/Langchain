from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

llm = HuggingFaceEndpoint(
  repo_id="Qwen/Qwen3-Coder-480B-A35B-Instruct",
  temperature=0,
  task="text-generation",
  max_new_tokens=10
)

model = ChatHuggingFace(llm=llm)

templet = PromptTemplate(
  template="Translate Following {text} to {language}",
  input_variables=['text','language']
)

parser = StrOutputParser()

chain = templet | model | parser

text = input("Enter Text Here : ")
language = input("in Which Language you want to Convert Above Text. \n")

result = chain.invoke({
  "text" : text,
  "language" : language
})

print("\n",result)
