import os
import requests
import json
from fastapi import FastAPI
from pydantic import BaseModel
from groq import Groq
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer

app = FastAPI()

# Load Keys
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
SERPER_API_KEY = os.environ.get("SERPER_API_KEY")
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY") # We will add this to Render next

client = Groq(api_key=GROQ_API_KEY)

# Initialize Pinecone and the embedding model
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index("my-templates-v2")
embed_model = SentenceTransformer('all-MiniLM-L6-v2')

class ChatRequest(BaseModel):
    question: str

@app.post("/chat")
async def chat(request: ChatRequest):
    question = request.question

    # 1. SEARCH PINECONE (Our new RAG step)
    print("Searching templates in Pinecone...")
    query_vector = embed_model.encode(question).tolist()
    # Get the top 3 most relevant chunks from your template
    search_results = index.query(vector=query_vector, top_k=3, include_metadata=True)
    
    template_context = ""
    for match in search_results['matches']:
        if 'text' in match['metadata']:
            template_context += match['metadata']['text'] + "\n\n"

    # 2. ASK GROQ
    print("Asking Groq...")
    chat_completion = client.chat.completions.create(
        messages=[
            {"role": "system", "content": "You are an expert web developer. Use the provided HTML/CSS template context to help the user build or modify their website. If the answer isn't in the context, use your own vast coding knowledge."},
            {"role": "user", "content": f"Template Context:\n{template_context}\n\nUser Request: {question}"}
        ],
        model="openai/gpt-oss-120b",
    )
    
    answer = chat_completion.choices[0].message.content
    return {"answer": answer}

@app.get("/")
def read_root():
    return {"status": "Backend is running"}
