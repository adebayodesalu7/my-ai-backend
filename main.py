import os
import requests
import json
import cohere # New import
from fastapi import FastAPI
from pydantic import BaseModel
from groq import Groq
from pinecone import Pinecone

app = FastAPI()

# Load Keys
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
SERPER_API_KEY = os.environ.get("SERPER_API_KEY")
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
COHERE_API_KEY = os.environ.get("COHERE_API_KEY") # New key

client = Groq(api_key=GROQ_API_KEY)
co = cohere.Client(COHERE_API_KEY) # Initialize Cohere

# Initialize Pinecone (REMOVE SentenceTransformer)
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index("my-templates-v3")

class ChatRequest(BaseModel):
    question: str

@app.post("/chat")
async def chat(request: ChatRequest):
    question = request.question

    # 1. SEARCH PINECONE using Cohere Embeddings
    print("Searching templates in Pinecone...")
    # Use Cohere to turn the question into a vector
    response = co.embed(
        texts=[question],
        model="embed-english-v3.0",
        input_type="search_query"
    )
    query_vector = response.embeddings[0]
    
    # Search Pinecone with the Cohere vector
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
