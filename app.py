import os
from dotenv import load_dotenv
from openai import OpenAI
import PyPDF2
import numpy as np
import faiss
import streamlit as st

# -------------------------------
# Load API key
# -------------------------------
load_dotenv("ai-week1.env")
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY not found in environment")

client = OpenAI(api_key=api_key)

# -------------------------------
# Initialize memory for chat
# -------------------------------
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []

# -------------------------------
# Helper functions
# -------------------------------
def split_text(text, chunk_size=1000, overlap=100):
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

def get_embedding(text):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return np.array(response.data[0].embedding, dtype=np.float32)

def create_faiss_index(chunks):
    embeddings = [get_embedding(chunk) for chunk in chunks]
    dimension = len(embeddings[0])
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))
    return index, embeddings

def query_pdf(question, chunks, index, top_k=3):
    q_emb = get_embedding(question)
    q_emb = np.array([q_emb])
    distances, indices = index.search(q_emb, top_k)
    context = "\n".join([chunks[i] for i in indices[0]])
    prompt = f"Answer the question using the context below:\n\nContext:\n{context}\n\nQuestion: {question}"
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

def chat_with_memory(user_input):
    st.session_state.conversation_history.append({"role": "user", "content": user_input})
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=st.session_state.conversation_history
    )
    assistant_reply = response.choices[0].message.content
    st.session_state.conversation_history.append({"role": "assistant", "content": assistant_reply})
    return assistant_reply

# -------------------------------
# Streamlit UI
# -------------------------------
st.title("Core AI Integration")

option = st.sidebar.selectbox("Choose functionality:", ["PDF Q&A", "Customer Support Chatbot"])

if option == "PDF Q&A":
    uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")
    if uploaded_file is not None:
        reader = PyPDF2.PdfReader(uploaded_file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"

        st.success("PDF loaded successfully!")
        chunks = split_text(text)
        st.write(f"PDF split into {len(chunks)} chunks. Generating FAISS index...")
        index, embeddings = create_faiss_index(chunks)
        st.success("FAISS index created!")

        question = st.text_input("Enter your question about the PDF:")
        if st.button("Get Answer") and question:
            answer = query_pdf(question, chunks, index)
            st.write("**Answer:**", answer)

elif option == "Customer Support Chatbot":
    st.write("Chat with memory-enabled AI (type in the box and hit Enter)")
    user_input = st.text_input("You:", key="chat_input")
    if st.button("Send") and user_input:
        reply = chat_with_memory(user_input)
        st.write("**Bot:**", reply)

    if st.session_state.conversation_history:
        st.write("---")
        st.write("**Conversation History:**")
        for msg in st.session_state.conversation_history:
            role = "You" if msg["role"] == "user" else "Bot"
            st.write(f"**{role}:** {msg['content']}")
