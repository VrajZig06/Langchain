from langchain_huggingface import HuggingFaceEmbeddings,HuggingFaceEndpoint,ChatHuggingFace
from langchain_community.document_loaders import DirectoryLoader,PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.prompts import PromptTemplate
import warnings

warnings.filterwarnings(
    "ignore",
    message=".*langchain_community.vectorstores.*",
)

embeddingModel = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2")

loader = DirectoryLoader(
    path="Projects/ZignutsPolicy",
    glob="*.pdf",
    loader_cls=PyPDFLoader
)

llm = HuggingFaceEndpoint(
    repo_id = "Qwen/Qwen3-Coder-480B-A35B-Instruct",
    task = "text-generation",
    max_new_tokens = 10
)

model = ChatHuggingFace(llm=llm)

docs = loader.lazy_load()

all_text = [doc.page_content for doc in docs]

All_Plicy_Data = " ".join(all_text)

# Text Splitter 
splitter = RecursiveCharacterTextSplitter(
    chunk_size = 300,
    chunk_overlap = 50
)

Split_Text = splitter.split_text(All_Plicy_Data)

vectorStore = Chroma(
    embedding_function=embeddingModel,
    persist_directory="DATA",
    collection_name="DATA"
)

# vectorStore.add_texts(Split_Text)
query = input("Enter Your Question reagrding Zignuts Policy : - ")

base_retriever = vectorStore.as_retriever(search_kwargs= {"k" : 10})
compressor = LLMChainExtractor.from_llm(llm=model)
comression_retriever = ContextualCompressionRetriever(
    base_retriever=base_retriever,
    base_compressor=compressor
)

relatedDocs = comression_retriever.invoke(query)

context = "\n\n".join([doc.page_content for doc in relatedDocs])
print(context)


prompt = PromptTemplate(
    template="""
        You are a helpful and intelligent Assistant.
        Answer the user's query based strictly on the provided context.
        And Give me Short Answer for any Query.
        but Make Sure Short Description is Understandale easily

        User Query:
        {query}

        Context:
        {context}

        Instructions:
        - Base your response only on the context.
        - If the answer is not present in the context, just say I Don't Know!.
        - Keep your response clear and informative.
        """,
        input_variables=['query','context'],
    )
  
chain = prompt | model 

result = chain.invoke({
    "query" : query,
    "context" : context
})

print(result.content)