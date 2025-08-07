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
parser = StrOutputParser()

templet1 = PromptTemplate(
  template= "Give me Full Detailed Description for {topic}",
  input_variables=['topic']
)

templet2 = PromptTemplate(
  template="Give me 5 lines Summary for given following text.\n {text}",
  input_variables=['text']
)

chain = templet1 | model | parser | templet2 | model | parser

result = chain.invoke({
  "topic" : "AI"
})

print(result)

chain.get_graph().print_ascii()

"""
                                        +-------------+       
                                        | PromptInput |
                                        +-------------+
                                                *
                                                *
                                                *
                                        +----------------+
                                        | PromptTemplate |
                                        +----------------+
                                                *
                                                *
                                                *
                                      +-----------------+
                                      | ChatHuggingFace |
                                      +-----------------+
                                                *
                                                *
                                                *
                                      +-----------------+
                                      | StrOutputParser |
                                      +-----------------+
                                                *
                                                *
                                                *
                                    +-----------------------+
                                    | StrOutputParserOutput |
                                    +-----------------------+
                                                *
                                                *
                                                *
                                        +----------------+
                                        | PromptTemplate |
                                        +----------------+
                                                *
                                                *
                                                *
                                      +-----------------+
                                      | ChatHuggingFace |
                                      +-----------------+
                                                *
                                                *
                                                *
                                      +-----------------+
                                      | StrOutputParser |
                                      +-----------------+
                                                *
                                                *
                                                *
                                    +-----------------------+
                                    | StrOutputParserOutput |
                                    +-----------------------+

"""