from langchain_community.document_loaders import YoutubeLoader
from youtube_transcript_api import YouTubeTranscriptApi 
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings,HuggingFaceEndpoint,ChatHuggingFace
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser,PydanticOutputParser
from pydantic import BaseModel
from youtube_id import extract_youtube_id
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.retrievers import ContextualCompressionRetriever

# Hsotory of Chat 
chat_history = []

# Pydantic Model for Response 
class APIResponse(BaseModel):
   result : str

#   Create LLM using HuggingFace Model
llm = HuggingFaceEndpoint(
    repo_id = "Qwen/Qwen3-Coder-480B-A35B-Instruct",
    task = "text-generation",
    max_new_tokens = 10
  )

LLM_Model = ChatHuggingFace(llm=llm)


# Embedding Model to Generate vectors 
embeddings = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2")

# Take Youtube URL from user
url = input("Enter your Youtube Url here ...")

# Extracting Id From URL 
videoId = extract_youtube_id(url)

try:
  ytt_api = YouTubeTranscriptApi()

  # Fetched Youtube Transcripts List from Video Id
  dataList = ytt_api.fetch(videoId,languages=['en'])

#   Now extract all Text from Fetched Data
  transcripts = [data.text for data in dataList]

# Joins All Text from List 
  texts = " ".join(transcripts)

# Create recursive text splitters 
  splitter = RecursiveCharacterTextSplitter(
      chunk_size = 500,
      chunk_overlap = 100
  )

# Split Text into 100 Character Size Chunks 
  chunksList = splitter.split_text(texts)

# Creating our External knowledge Base --- Vector Database
  vector_store = Chroma(
        embedding_function=embeddings,
        persist_directory="YT_Vector_DB",
        collection_name= f"Transcript_{videoId}"
    )

# Add Vectors to VectorStore
  vector_store.add_texts(chunksList)

# Get All Vectors Data
#   print(vector_store.get())

# Get User Query  
  query = input("Please Enter Your Question here ...") or "Exit"


# # Get Relative Documents 
#   relativeVectors = vector_store.similarity_search(
#         query=query,
#         k=4
#     )

#   Creating Contextual Compressor Retriever 
  base_retriever = vector_store.as_retriever(search_kwargs={"k": 5})
  compressor = LLMChainExtractor.from_llm(llm=LLM_Model)
  compression_retriever = ContextualCompressionRetriever(
    base_retriever=base_retriever,
    base_compressor=compressor
)

  relativeVectors = compression_retriever.invoke(query)

# Get List of All Relative Documents texts 
  all_relative_documents = [doc.page_content for doc in relativeVectors]

#   Creating Str Parsers
  strParser = StrOutputParser()
  pydanticParser = PydanticOutputParser(pydantic_object=APIResponse)


# Join all Texts to make Context from Relative Documents 
  context = "\n\n".join(all_relative_documents)


# Make one Prompt that generate Text using this Context for user Query 

  prompt = PromptTemplate(
    template="""
        You are a helpful and intelligent Assistant.
        Answer the user's query based strictly on the provided context.

        User Query:
        {query}

        Context:
        {context}

        Instructions:
        - Base your response only on the context.
        - If the answer is not present in the context, politely state that.
        - Keep your response clear and informative.\n{format_instruction}
        """,
        input_variables=['query','context'],
        partial_variables={"format_instruction" : pydanticParser.get_format_instructions()}
    )
  
  
# Creating Chain for this operation 
  chain = prompt | LLM_Model | pydanticParser

# Invoke Chain Here
  llmResponse = chain.invoke({
     'query' : query,
     "context" : context
  })

  print(llmResponse)
  chat_history.append({
     "query" : query,
     "response" : llmResponse.result
  })

  print(chat_history)

except Exception as e:
    print(e)

