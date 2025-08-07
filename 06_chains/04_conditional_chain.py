from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser,PydanticOutputParser
from langchain_core.runnables import RunnableParallel,RunnableBranch,RunnableLambda
from pydantic import BaseModel
from typing import Literal

class FeedBack(BaseModel):
  sentiments : Literal['Positive',"Negative"]

llm = HuggingFaceEndpoint(
  repo_id="Qwen/Qwen3-Coder-480B-A35B-Instruct",
  task="text-generation",
  temperature=1.8,
  max_new_tokens=10
)

model = ChatHuggingFace(llm=llm)
parser = StrOutputParser()
parser2 = PydanticOutputParser(pydantic_object=FeedBack)

templet1 = PromptTemplate(
  template="Analysis of floowing feedback is Negative or positivie. \n {feedback}. \n {format_instruction}",
  input_variables=['feedback'],
  partial_variables={"format_instruction" : parser2.get_format_instructions()}
)

templet2 = PromptTemplate(
  template="Give Appropriate Answer for Following Negative Feedback in 2 lines only. \n {feedback}",
  input_variables=['feedback']
)

templet3 = PromptTemplate(
  template="Give Appropriate Answer for Following Positive Feedback in 2 lines only. \n {feedback}",
  input_variables=['feedback']
)

analysis_chain = templet1 | model | parser2

branch_chain = RunnableBranch(
  (lambda x: x.sentiments == "Positive", templet3 | model | parser),
  (lambda x: x.sentiments == "Negative", templet2 | model | parser),
  RunnableLambda(lambda x : "Not Extract Sentiments")
)

chain = analysis_chain | branch_chain

result = chain.invoke({
  "feedback" : "This is very Nice and affordable for Middle Class Family."
})

print(result)