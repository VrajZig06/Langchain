from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

llm = HuggingFaceEndpoint(
  repo_id="Qwen/Qwen3-Coder-480B-A35B-Instruct",
  task="text-generation",
  temperature=1.8,
  max_new_tokens=10
)

model = ChatHuggingFace(llm=llm)

templet = PromptTemplate(
  template="""Generate 5 intresting facts about {topic}.\n""",
  input_variables=['topic']
)

parser = StrOutputParser()


chain = templet | model | parser

result = chain.invoke({
  "topic" : "Cricket"
})

chain.get_graph().print_ascii()

print(result)