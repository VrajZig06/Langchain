
from langchain_experimental.tools import PythonREPLTool

python_tool = PythonREPLTool()

result = python_tool.invoke("3+2")
print(result) 

# code = """
# import math
# math.sqrt(144)
# """
# print(python_tool.run(code))  