from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel,Field
from typing import List
from langchain_community.tools import ShellTool
from langchain.tools import tool
import json


# Tools using Shelltool

shellTool = ShellTool()

@tool
def createFolder(foldername : str):
    """ 
    This is Function that is used to Create only Folder.
    """
    shellTool.invoke(f"mkdir {foldername}")

@tool
def createFolderAndFiles(foldername:str,files:List[str]):
    """
    This is the function that is used to Create Folder and Create Files in that Folder
    """
    shellTool.invoke(f"mkdir {foldername}")
    shellTool.invoke(f"cd {foldername} && touch {' '.join([file for file in files])}")

    

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen3-Coder-480B-A35B-Instruct",
    task="text-generation",
    max_new_tokens=100)

model = ChatHuggingFace(llm=llm)

class Output(BaseModel):
    foldername :str
    files : List[str] 

pydanticParser = PydanticOutputParser(pydantic_object=Output)

templet = PromptTemplate(
    template="""
        You are a Good Understanding Model.
        You need to extract foldername and Files list from following user Query

        Query : {query}

    """,
    input_variables = ['query'],
)

query = input("Enter Your Command Here : - ")

# Model Binding
llm_with_tools = model.bind_tools([createFolder,createFolderAndFiles])

chain = templet | llm_with_tools
result = chain.invoke(query)
tool_calls = result.tool_calls

if len(tool_calls):
    for tool in tool_calls:
        if tool['name'] == "createFolderAndFiles":
            createFolderAndFiles.invoke(tool)

        if tool['name'] == "createFolder":
            createFolder.invoke(tool)
        





