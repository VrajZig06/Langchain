import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
import warnings
from vectors import Vectors
from cricketData import cricket_players_info

warnings.filterwarnings("ignore", category=FutureWarning)
embedding = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2",model_kwargs={"device": "cpu"})

st.header("Cricket Players Details Search")
data = st.text_input("Enter your Question Here...")
btn = st.button("Search")

if btn:
    if data.strip() == "" or data is None:
        st.write("Please Enter Some Prompt to input Field.")
    else: 
        query_embedding = embedding.embed_query(data)
        scores = cosine_similarity([query_embedding],Vectors)[0] # must be 2D List
        scores_with_index = list(enumerate(scores))

        index,scored = sorted(scores_with_index,key=lambda x : x[1])[-1] # Reverse and First one Score
        st.write(cricket_players_info[index])