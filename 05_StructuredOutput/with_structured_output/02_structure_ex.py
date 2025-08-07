from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from typing import TypedDict,Annotated

llm = HuggingFaceEndpoint(
  repo_id="Qwen/Qwen3-Coder-480B-A35B-Instruct",
  temperature=0,
  task="text-generation",
  max_new_tokens=10
)

model = ChatHuggingFace(llm=llm)

review = "The Realme 11x 5G offers excellent battery life with its 5000mAh capacity, easily lasting a full day of use, and supports 33W fast charging, which can fully charge the phone in about 80 minutes."

# now convert this review in format of {summary and sentiments}
class Review(TypedDict):
  summary : Annotated[str,"A brief Summary of review"]
  sentiments : Annotated[str,"Return sentiment of Review either negative,positive or neutral"] 

structured_model = model.with_structured_output(Review)

result = structured_model.invoke(review)

print(result)


"""
--> Output: 

{'summary': 'The Realme 11x 5G offers excellent battery life with its 5000mAh capacity, easily lasting a full day of use, and supports 33W fast charging, which can fully charge the phone in about 80 minutes.','sentiments': 'positive'}

"""