
import time
import streamlit as st
import functools
import faiss
import numpy as np
from mistralai import Mistral, UserMessage
from bs4 import BeautifulSoup
import requests
 
# API Key Setup
API_KEY = "xjCgy80GBjYF4qDbKke2ZI98Q8jxoinY"

@st.cache_data(show_spinner=False)  # Cache API responses
def get_text_embedding_cached(list_txt_chunks):
    #time.sleep(1)  # Prevent rapid API calls
    client = Mistral(api_key=API_KEY)
    embeddings_batch_response = client.embeddings.create(model="mistral-embed", inputs=list_txt_chunks)
    return embeddings_batch_response.data

def get_text_embedding(list_txt_chunks):
    return get_text_embedding_cached(list_txt_chunks)

@st.cache_data(show_spinner=False)
def mistral_chat_cached(user_message):
    time.sleep(1)  # Prevent rapid API calls
    client = Mistral(api_key=API_KEY)
    messages = [UserMessage(content=user_message)]
    chat_response = client.chat.complete(model="mistral-large-latest", messages=messages)
    return chat_response.choices[0].message.content

def mistral_chat(user_message):
    return mistral_chat_cached(user_message)
    
def load_policy_data(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")
    text = soup.get_text()
    return text

def build_faiss_index(text_chunks):
    text_embeddings = get_text_embedding(text_chunks)
    embeddings = np.array([emb.embedding for emb in text_embeddings])
    index = faiss.IndexFlatL2(len(text_embeddings[0].embedding))
    index.add(embeddings)
    return index, text_chunks

def classify_intent(query, policy_index, policy_names):
    query_embedding = np.array([get_text_embedding([query])[0].embedding])
    D, I = policy_index.search(query_embedding, k=1)  # Find best match
    return policy_names[I[0][0]]

def retrieve_relevant_chunks(query, index, text_chunks, top_k=2):
    query_embedding = np.array([get_text_embedding([query])[0].embedding])
    D, I = index.search(query_embedding, k=top_k)
    return [text_chunks[i] for i in I.tolist()[0]]

# Load 20+ policies
policy_urls = {
    "1- Student Conduct Policy": "https://www.udst.edu.qa/.../student-conduct-policy",
    "2- Academic Schedule Policy": "https://www.udst.edu.qa/.../academic-schedule-policy",
    "3- Student Attendance Policy": "https://www.udst.edu.qa/.../student-attendance-policy",
    "4- Student Appeals Policy": "https://www.udst.edu.qa/.../student-appeals-policy",
    "5- Graduation Policy": "https://www.udst.edu.qa/.../graduation-policy",
    "6- Academic Standing Policy": "https://www.udst.edu.qa/.../academic-standing-policy",
    "7- Transfer Policy": "https://www.udst.edu.qa/.../transfer-policy",
    "8- Admissions Policy": "https://www.udst.edu.qa/.../admissions-policy",
    "9- Final Grade Policy": "https://www.udst.edu.qa/.../final-grade-policy",
    "10- Registration Policy": "https://www.udst.edu.qa/.../registration-policy",
    # Add 10 more policies
}

st.title("UDST Policy Chatbot (Agentic RAG)")

# Load policies and build intent classifier
policy_texts = [load_policy_data(url) for url in policy_urls.values()]
policy_chunks = [[text[i:i+512] for i in range(0, len(text), 512)] for text in policy_texts]
policy_names = list(policy_urls.keys())
policy_embeddings = [get_text_embedding([name]) for name in policy_names]
policy_index = faiss.IndexFlatL2(len(policy_embeddings[0][0].embedding))
policy_index.add(np.array([emb[0].embedding for emb in policy_embeddings]))

# Store policy data
st.session_state["policy_indexes"] = {}
st.session_state["policy_chunks"] = {}
for name, chunks in zip(policy_names, policy_chunks):
    index, chunk_data = build_faiss_index(chunks)
    st.session_state["policy_indexes"][name] = index
    st.session_state["policy_chunks"][name] = chunk_data

# Query Input
query = st.text_input("Enter your question:")
if st.button("Get Answer"):
    if query:
        policy_name = classify_intent(query, policy_index, policy_names)
        index = st.session_state["policy_indexes"].get(policy_name)
        chunks = st.session_state["policy_chunks"].get(policy_name)
        
        retrieved_chunks = retrieve_relevant_chunks(query, index, chunks)
        prompt = f"""
        Context information:
        {retrieved_chunks}
        Given the context, answer the query: {query}
        """
        response = mistral_chat(prompt)
        st.text_area("Answer:", response, height=200)
