import os
import requests
import json
import cohere
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
from groq import Groq
from pinecone import Pinecone

app = FastAPI()

# Load Keys
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
COHERE_API_KEY = os.environ.get("COHERE_API_KEY")

client = Groq(api_key=GROQ_API_KEY)
co = cohere.Client(COHERE_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index("my-templates-v3")

class ChatRequest(BaseModel):
    question: str
    image_base64: Optional[str] = None

@app.post("/chat")
async def chat(request: ChatRequest):
    question = request.question
    template_context = ""

    if not request.image_base64:
        try:
            response = co.embed(
                texts=[question],
                model="embed-english-v3.0",
                input_type="search_query"
            )
            query_vector = response.embeddings[0]
            search_results = index.query(vector=query_vector, top_k=8, include_metadata=True)
            
            for match in search_results['matches']:
                if 'text' in match['metadata']:
                    template_context += match['metadata']['text'] + "\n\n"
        except Exception as e:
            print(f"Pinecone error: {e}")

    print("Asking Groq...")
    
    if request.image_base64:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": question},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{request.image_base64}"}}
                ]
            }
        ]
        model_to_use = "qwen/qwen3.8-27b"
    else:
        # THIS IS THE FIX: The new System Prompt
        system_prompt = "You are an expert Senior Web Developer. Do not immediately write code. First, briefly outline your plan and ask the user for permission to proceed. Only write the full code after the user says 'yes' or gives the go-ahead. Use the provided HTML/CSS template context to build the website."
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Template Context:\n{template_context}\n\nUser Request: {question}"}
        ]
        model_to_use = "openai/gpt-oss-120b"

    chat_completion = client.chat.completions.create(
        messages=messages,
        model=model_to_use,
        max_tokens=2048,  # THIS prevents the cut-off
    )
    
    answer = chat_completion.choices[0].message.content
    return {"answer": answer}

@app.get("/")
def read_root():
    return {"status": "Backend is running"}
