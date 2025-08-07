from langchain_core.tools import tool

# --------- Please Following Actions must do during Making Custom Tool ---------

# 1. Add DocStrings in Function -> Because LLM easily understand which Function is Useful in particular Situation
# 2. Add type hints for each parameter.s
# 3. Add @tool Decorator to that function so it will become Tool

# ------ Ways to Create Tools -------
# 1. Using @tool Decorator
# 2. Using Structure Tool and Pydantic
# 3. Using BaseTool Class

@tool
def Multiply(a:int,b:int) -> int:
    """ This is Function for Two Numbers Multiplications. """
    return a*b

# We Will Give it From LLM Response (JSON Response)
result = Multiply.invoke({"a" : 2,"b" : 3})
print(result)

print(Multiply.args_schema.model_json_schema()) # LLM will See this 



