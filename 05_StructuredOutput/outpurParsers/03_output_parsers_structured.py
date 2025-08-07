from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace
from langchain.output_parsers import ResponseSchema,StructuredOutputParser
from langchain.prompts import PromptTemplate


llm = HuggingFaceEndpoint(
 repo_id = "Qwen/Qwen3-Coder-480B-A35B-Instruct",
  task="text-generation",
  max_new_tokens=50,
  temperature=0
)

model = ChatHuggingFace(llm=llm)

schema = [
  ResponseSchema(name="fact1",description="Fact 1 of topic"),
  ResponseSchema(name="fact2",description="Fact 2 of topic"),
  ResponseSchema(name="fact3",description="Fact 3 of topic"),
]

parser = StructuredOutputParser.from_response_schemas(schema)

templet = PromptTemplate(
  template="""Give me 3 Facts about Topic : {topic} \n {format_instruction}""",
  input_variables=['topic'],
  partial_variables={"format_instruction" : parser.get_format_instructions()}
)

chain = templet | model | parser

result = chain.invoke({
  "topic" : "AI"
})

print(result)

'''
--> Output: 
{'fact1': 'Artificial Intelligence was first coined as a term in 1956 by computer scientist John McCarthy at the Dartmouth Conference, marking the official beginning of AI as a field of study.', 'fact2': 'Machine learning, a subset of AI, enables computers to learn and improve from experience without being explicitly programmed for every task, allowing systems to recognize patterns and make predictions.', 'fact3': 'AI already powers many everyday technologies including voice assistants like Siri and Alexa, recommendation systems on Netflix and Amazon, and search engines like Google, making it an integral part of modern digital life.'}

'''