from langchain_huggingface import HuggingFaceEmbeddings,HuggingFaceEndpoint,ChatHuggingFace
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
import streamlit as st
import os
from dotenv import load_dotenv

load_dotenv()

# LangSmith Tracing 
os.environ['LANGSMITH_TRACING'] = os.getenv('LANGSMITH_TRACING')
os.environ['LANGSMITH_API_KEY'] = os.getenv('LANGSMITH_API_KEY')
    
# Prompt Template
prompt = ChatPromptTemplate(
    [
        ('system',"You are a Good and Smart Assistance. Please Answer all the Queries that asks by users."),
        ('user' , "User Query : {query}")
    ]
)

# Streamlit framework --
st.title("ChatBot Using HuggingFace Model")
user_query = st.text_input("Enter your Query Here")

# Hugging Face LLM Model
llm = HuggingFaceEndpoint(
  repo_id="Qwen/Qwen3-Coder-480B-A35B-Instruct",
  task="text-generation",
  temperature=1.8,
  max_new_tokens=10
)
model = ChatHuggingFace(llm=llm)

# output Parser
strParser = StrOutputParser()

# Chain 
chain = prompt | model | strParser

if user_query:
    # Getting Result from LLM Model
    result = chain.invoke({
        "query" : user_query
    })

    st.write(result)