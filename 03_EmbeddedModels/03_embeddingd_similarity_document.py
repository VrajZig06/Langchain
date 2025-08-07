from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

load_dotenv()

embedding = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2")

cricket_facts = [
    "Sachin Tendulkar is the only player to have scored 100 international centuries.",
    "The longest cricket match in history lasted 12 days between England and South Africa in 1939.",
    "The first official international cricket match was played between Canada and the USA in 1844.",
    "Chris Gayle is the only player to hit a six off the first ball of a Test match.",
    "Muttiah Muralitharan holds the record for the most wickets in Test cricket (800 wickets)."
]

query = "tell me about so many wickets of any player"

doc_embeddings = embedding.embed_documents(cricket_facts)
query_embedding = embedding.embed_query(query)

scores = cosine_similarity([query_embedding],doc_embeddings)[0] # must be 2D List

scores_with_index = list(enumerate(scores))

index,scored = sorted(scores_with_index,key=lambda x : x[1])[-1] # Reverse and First one Score

print(cricket_facts[index])

