from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel

textInput = """
The world is divided into seven continents, with Asia being the largest in both area and population.
Earth's surface is about 71% water, primarily in oceans like the Pacific, Atlantic, and Indian.
Mount Everest, located in the Himalayas, is the tallest mountain above sea level.
The Amazon Rainforest, spanning several South American countries, is the largest tropical rainforest on Earth.
The Sahara Desert is the largest hot desert in the world, covering much of North Africa.
There are about 195 countries in the world today, each with its own government and culture.
English, Mandarin Chinese, and Spanish are among the most widely spoken languages globally.
The global population surpassed 8 billion people in 2022, with more than half living in urban areas.
Climate change is a significant global issue, causing rising sea levels and more extreme weather.
The United Nations, founded in 1945, plays a key role in promoting peace and international cooperation.
The world’s economy is interconnected, with global trade impacting local industries.
Technological advancements continue to reshape global communication and economies.
Major world religions include Christianity, Islam, Hinduism, and Buddhism.
Natural disasters such as earthquakes, hurricanes, and wildfires affect many regions worldwide.
Preserving biodiversity is critical to maintaining Earth's ecological balance and sustainability.
"""


llm = HuggingFaceEndpoint(
  repo_id="Qwen/Qwen3-Coder-480B-A35B-Instruct",
  task="text-generation",
  temperature=1.8,
  max_new_tokens=10
)

model = ChatHuggingFace(llm=llm)
parser = StrOutputParser()

templet1 = PromptTemplate(
  template="Generate notes from following texts. \n {text}",
  input_variables=['text']
)

templet2 = PromptTemplate(
  template="Generate Quiz from following texts. \n {text}",
  input_variables=['text']
)

templet3 = PromptTemplate(
  template="Merge Both {notes} and {Quiz} in one Document",
  input_variables=['notes',"Quiz"]
)

parser = StrOutputParser()

parallel_chain  = RunnableParallel({
  "notes" : templet1 | model | parser,
  "Quiz" : templet2 | model | parser
})

merge_chain = templet3 | model | parser

chain = parallel_chain | merge_chain 

result = chain.invoke({
  "text" : textInput
})

print(result)

chain.get_graph().print_ascii()


"""
                          +---------------------------+
                          | Parallel<notes,Quiz>Input |
                          +---------------------------+
                              ***               ***
                          ***                     ***
                        **                           **
            +----------------+                    +----------------+
            | PromptTemplate |                    | PromptTemplate |
            +----------------+                    +----------------+
                      *                                   *
                      *                                   *
                      *                                   *
            +-----------------+                  +-----------------+
            | ChatHuggingFace |                  | ChatHuggingFace |
            +-----------------+                  +-----------------+
                      *                                   *
                      *                                   *
                      *                                   *
            +-----------------+                  +-----------------+
            | StrOutputParser |                  | StrOutputParser |
            +-----------------+                  +-----------------+
                              ***               ***
                                ***         ***
                                    **     **
                        +----------------------------+
                        | Parallel<notes,Quiz>Output |
                        +----------------------------+
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