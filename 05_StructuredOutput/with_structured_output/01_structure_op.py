# There is Two types of Models.
# 1. Model that having Capabilities of giving Structure Output  --> with_structured_output()
# 2. Model that not having Capabilitites of giving Structure Output --> Output Parser

# --> TypeDict
# --> Pydantic
# --> jsonn_schema

from typing import TypedDict

class Person(TypedDict):
  name : str
  age : int

person_one : Person = {
  "name" : "Vraj",
  "age" : 23
}

print(person_one)