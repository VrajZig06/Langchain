from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2")

text = "Delhi is the Capital of India"
text_documents = ["Delhi is the Capital of India",
                  "Kolkata is the capital of West Bengal",
                  "Paris is the Capital of France"]

vector = embeddings.embed_documents(text_documents)

# By Default Vector size is 384 dimenstions Vector 
# vector = embeddings.embed_query(text)

print(str(vector)) 