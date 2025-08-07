from langchain.tools import tool


# Toolkit : Toolkit is a kind of collectionsof Multiple Tools
@tool
def Multiply(a:int,b:int) -> int:
    """ This is Function for Two Numbers Multiplications. """
    return a*b

@tool
def Addition(a:int,b:int) -> int:
    """ This is Function for Two Numbers Addition. """
    return a + b

class MathToolkit():
    def get_tools(self):
        return [Addition,Multiply]
    
toolkit = MathToolkit()

all_tools = toolkit.get_tools()

for tool in all_tools:
    print(f"{tool.name} : {tool.description}")

