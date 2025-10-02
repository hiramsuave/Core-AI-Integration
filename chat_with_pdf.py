# ===============================
# Week 1 – AI Playground
# ===============================

import os
from dotenv import load_dotenv
from openai import OpenAI
from PyPDF2 import PdfReader
import numpy as np
import faiss

# -------------------------------
# Step 0 – Load API Key
# -------------------------------
load_dotenv("ai-week1.env")
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY not found in environment")

# Initialize OpenAI client (new SDK)
client = OpenAI(api_key=api_key)

# -------------------------------
# Step 1 – Read PDF and Split
# -------------------------------
pdf_path = "example.pdf"  # Replace with your PDF
reader = PdfReader(pdf_path)
text = ""
for page in reader.pages:
    text += page.extract_text() + "\n"

print(f"PDF loaded. Total characters: {len(text)}")

def split_text(text, chunk_size=1000, overlap=100):
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

chunks = split_text(text)
print(f"Total chunks created: {len(chunks)}")

# -------------------------------
# Step 2 – Create Embeddings & FAISS Index
# -------------------------------
def get_embedding(text):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return np.array(response.data[0].embedding, dtype=np.float32)

print("Generating embeddings... (this may take a few minutes)")
embeddings = [get_embedding(chunk) for chunk in chunks]
dimension = len(embeddings[0])
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings))
print("FAISS index created with embeddings.")

# -------------------------------
# Step 3 – PDF Q&A Function
# -------------------------------
def query_pdf(question, top_k=3):
    # Embed the question
    q_emb = get_embedding(question)
    q_emb = np.array([q_emb])
    
    # Search in FAISS
    distances, indices = index.search(q_emb, top_k)
    
    # Combine relevant chunks
    context = "\n".join([chunks[i] for i in indices[0]])
    
    # GPT prompt
    prompt = f"Answer the question using the context below:\n\nContext:\n{context}\n\nQuestion: {question}"
    
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
    )
    
    return response.choices[0].message.content

# -------------------------------
# Step 4 – Customer Support Bot
# -------------------------------
conversation_history = []

def chat_with_memory(user_input):
    global conversation_history
    conversation_history.append({"role": "user", "content": user_input})
    
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=conversation_history
    )
    
    assistant_reply = response.choices[0].message.content
    conversation_history.append({"role": "assistant", "content": assistant_reply})
    
    return assistant_reply

# -------------------------------
# Step 5 – Main Menu
# -------------------------------
def main():
    print("\n=== Week 1 AI Playground ===")
    while True:
        print("\nSelect an option:")
        print("1. PDF Q&A")
        print("2. Customer Support Chatbot")
        print("3. Exit")
        choice = input("Enter 1/2/3: ").strip()
        
        if choice == "1":
            question = input("\nEnter your question about the PDF: ")
            answer = query_pdf(question)
            print("\nAnswer:", answer)
        
        elif choice == "2":
            print("\nCustomer Support Bot (type 'exit' to return to menu)")
            while True:
                user_input = input("You: ")
                if user_input.lower() in ["exit", "quit"]:
                    break
                reply = chat_with_memory(user_input)
                print("Bot:", reply)
        
        elif choice == "3":
            print("Exiting program. Goodbye!")
            break
        else:
            print("Invalid option. Try again.")

if __name__ == "__main__":
    main()
