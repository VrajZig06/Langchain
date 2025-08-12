from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from fastapi import FastAPI
from langserve import add_routes
from pydantic import BaseModel,Field
import os
from typing import List     

# Load environment variables first
load_dotenv()

# Create FastAPI app
app = FastAPI(
    title="Chatbot API",
    description="This is API Endpoints for HuggingFace Chatbot.",
    version="1.0.0"
)

# LangSmith Tracing 
os.environ['LANGSMITH_TRACING'] = os.getenv('LANGSMITH_TRACING', 'false')
os.environ['LANGSMITH_API_KEY'] = os.getenv('LANGSMITH_API_KEY', '')

# Hugging Face LLM Model
llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    huggingfacehub_api_token="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",
    task="text-generation",
    temperature=0.7,  # Reduced from 1.8 for better consistency
    max_new_tokens=100,  # Increased from 10 for proper responses
)
llm = ChatHuggingFace(llm=llm)

# Create prompt template
prompt1 = ChatPromptTemplate.from_template("Write an essay about {topic} with 100 words.")

# Essay Generation Route
add_routes(
    app,
    prompt1 | llm,
    path="/essay",
    enable_feedback_endpoint=True,
    enable_public_trace_link_endpoint=True,
    playground_type="default",
    
)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)