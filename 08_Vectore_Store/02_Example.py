from langchain_huggingface import HuggingFaceEmbeddings,HuggingFaceEndpoint,ChatHuggingFace
from langchain.text_splitter import RecursiveCharacterTextSplitter
from CricketData import cricket_players_info
from langchain_chroma.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence
import warnings

# Suppress all FutureWarnings
warnings.simplefilter(action='ignore', category=FutureWarning)

textSplitters = RecursiveCharacterTextSplitter(
    chunk_size  = 200,
    chunk_overlap = 0,
    separators=["-"]
)
combined_text = "-".join(cricket_players_info)

docs = textSplitters.split_text(combined_text)

embeddingModel = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2")


vector_store = Chroma(
    embedding_function=embeddingModel,
    persist_directory="CricketVectoreStore",
    collection_name="Cricket_Data"
)

# vector_store.add_texts(docs)

query = input("Enter Your Question here... ")

relativeVectors = vector_store.similarity_search(
    query=query,
    k=4
)

llm = HuggingFaceEndpoint(
  repo_id="Qwen/Qwen3-Coder-480B-A35B-Instruct",
  task="text-generation",
  temperature=1.8,
  max_new_tokens=10
)

model = ChatHuggingFace(llm=llm)

templet = PromptTemplate(
    template="Give me Some Summary related to this user query {query} and following is the relative documents {docs}.",
    input_variables=['query','docs']
)

parser = StrOutputParser()

chain = RunnableSequence(templet,model,parser)

result = chain.invoke({
    'query':query,
    'docs' : relativeVectors
})

print(result)