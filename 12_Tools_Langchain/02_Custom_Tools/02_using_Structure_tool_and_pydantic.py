# Structured Tools : Where Input to that Tool is follows a Structured Schema, that we defined using Pydantic Model

from langchain.tools import StructuredTool
from pydantic import BaseModel,Field


# Pydantic Model for Multiplication 
class MultiplyInput(BaseModel):
    a: int = Field(required=True,description="First Number to Multiply")
    b: int = Field(required= True,description="Seconde Number to Multiply")

# Function for Multiplication 
def Multiply(a:int,b:int)-> int:
    return a*b

# Making Tool Here using StructureTool
multiply_tool = StructuredTool(
    func=Multiply,
    name ="Multiplly",
    description="Multiply Two Numbers",
    args_schema=MultiplyInput
)

# print(multiply_tool.invoke({"a" : 's',"b" : 3})) # It Will give Pydantic Error 
print(multiply_tool.invoke({"a" : '2',"b" : 3})) # It will validate Inputs and Give Answer 

