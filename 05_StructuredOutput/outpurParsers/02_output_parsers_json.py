from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace
from langchain_core.output_parsers import JsonOutputParser
from langchain.prompts import PromptTemplate


llm = HuggingFaceEndpoint(
 repo_id = "Qwen/Qwen3-Coder-480B-A35B-Instruct",
  task="text-generation",
  max_new_tokens=50,
  temperature=0
)

model = ChatHuggingFace(llm=llm)

# JosnParser 
parser = JsonOutputParser()

templet = PromptTemplate(
  template="Give me name,age,city of frictional person \n {format_instruction}",
  input_variables=[],
  partial_variables={'format_instruction' : parser.get_format_instructions()}
)

# ------ Without Chain ------
# prompt = templet.invoke({})
# result = model.invoke(prompt)
# print(parser.parse(result.content))


# ------ Using Chain --------
chain = templet | model | parser
final_result = chain.invoke({})
print(final_result)
