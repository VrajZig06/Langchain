from langchain_core.prompts import PromptTemplate
from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen3-Coder-480B-A35B-Instruct",
    task="text-generation",
    max_new_tokens=100)

model = ChatHuggingFace(llm=llm)

templet = PromptTemplate(
    template=  """
    Summarize the following text into a short paragraph:
    {text}
    Summary:
    """,
    input_variables=['text'],
    validate_template=True
)

input_Data = input("Enter Something Here : \n")

# fill the placeholders
# prompt = templet.invoke({
#     "text" : input_Data
# })

# result = model.invoke(prompt)


# We can make chain for this two invoke Activity 
chain = templet | model
result = chain.invoke({
     "text" : input_Data
})


print(result.content)