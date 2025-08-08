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

@tool
def OpenFileAndWrite(filepath:str,data:str):
    """
    This function opens a specified file and writes the given text content to it using shell commands. It appends the text if the file exists, or creates the file if it doesn't.
    """
    shellTool.invoke(f"echo {data} >> {filepath}")

    

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
        
        Query : {query}

    """,
    input_variables = ['query'],
)

query = input("Enter Your Command Here : - ")

# Model Binding
all_tools = [createFolder,createFolderAndFiles,OpenFileAndWrite]
llm_with_tools = model.bind_tools(all_tools)

chain = templet | llm_with_tools
result = chain.invoke(query)
print(result)
tool_calls = result.tool_calls

print(tool_calls)

if len(tool_calls) > 0:
    for tool in tool_calls:
        if tool['name'] == "createFolderAndFiles":
            createFolderAndFiles.invoke(tool)

        if tool['name'] == "createFolder":
            createFolder.invoke(tool)
        
        if tool['name'] == "OpenFileAndWrite":
            OpenFileAndWrite.invoke(tool)





