# Tool that Get Actual Currency rate from the API and Then Give You response

from langchain.tools import tool
from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from langchain.schema import HumanMessage
from langchain_core.tools import InjectedToolArg
import requests
import json
import streamlit as st

from typing import Annotated


API_KEY = "e4155a91db7442daf47d9055"

@tool
def getConversionRate(base_currency:str,nextCurrency:str):
    """
    This is Function that Get Currency Converstion Rate From API.
    """

    url = f"https://v6.exchangerate-api.com/v6/{API_KEY}/pair/{base_currency}/{nextCurrency}"

    response = requests.get(url)

    return response.json()

@tool
def ConvertCurrency(base_value:float,currency_rate:Annotated[float,InjectedToolArg]) -> float:
    """
    This is the Function that converts Base Currency Value to Target Currency value using Currency Converstion Rate.
    """

    return base_value * currency_rate

# Create LLM Model 
llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen3-Coder-480B-A35B-Instruct",
    task="text-generation",
    temperature=1.8,
    max_new_tokens=10
)

model = ChatHuggingFace(llm=llm)

model_with_llm = model.bind_tools([getConversionRate,ConvertCurrency])

messages = []

# query = input("Enter Your Query regarding to convert Currency :- ")
# query = "What is the Currency Conversion rate for USD and Convert Currency from 30000 USD to INR"
query = st.text_input("Enter Here")
btn = st.button("Ask")

if btn:
    messages.append(HumanMessage(content=query))

    result = model_with_llm.invoke(messages)
    messages.append(result)

    if len(result.tool_calls) > 0:
        for tool in result.tool_calls:
            if tool['name'] == "getConversionRate":
                response = getConversionRate.invoke(tool)
                conversion_rate = json.loads(response.content)['conversion_rate']
                messages.append(response.content)

            if tool['name'] == "ConvertCurrency":
                tool['args']['currency_rate'] = conversion_rate
                result = ConvertCurrency.invoke(tool)
                messages.append(result)

    final_result = model_with_llm.invoke(messages)
    if final_result.content == "":
        st.write("AI : Sorry Can you Try Again!")
    else:
        st.write("AI : ",final_result.content)

    # print(final_result.content)


# """
# --> OUTPUT
# (venv) ztlab141@ztlab141 Langchain Models % python3 13_Tool_Calling/02_Real_Time_Currency_Convertor.py
#     Enter Your Query regarding to convert Currency :- How much 250 USD to INR
#     The conversion rate from USD to INR is 87.8403. 

#     Now, let's convert 250 USD to INR.

#     250 USD * 87.8403 INR/USD = 21960.075 INR.  

#     So, 250 USD is approximately 21960.08 INR.

# """


