# BaseTool is Abstract class that in Present in Langchain that provide rules to make Tool 
from langchain.tools import BaseTool
from pydantic import BaseModel,Field
from typing import Type

# Pydantic Model for Multiplication 
class MultiplyInput(BaseModel):
    a: int = Field(required=True,description="First Number to Multiply")
    b: int = Field(required= True,description="Seconde Number to Multiply")

class MultiplyTool(BaseTool):
    name:str = "Multiply"
    description:str = "This Function Multiply Two Numbes"

    args_schema : Type[BaseModel] = MultiplyInput

    def _run(self,a:int,b:int):
        return a*b
    
multiply_tool = MultiplyTool()
result = multiply_tool.invoke({"a" :2,"b" : 2})
print(result)