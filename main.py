import os
import requests
import json
import cohere
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
from pinecone import Pinecone

app = FastAPI()

# Load Keys
CLOUDFLARE_ACCOUNT_ID = os.environ.get("CLOUDFLARE_ACCOUNT_ID")
CLOUDFLARE_API_TOKEN = os.environ.get("CLOUDFLARE_API_TOKEN")
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
COHERE_API_KEY = os.environ.get("COHERE_API_KEY")

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

    # 1. SEARCH PINECONE
    if not request.image_base64:
        print("Searching templates in Pinecone...")
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

    # 2. ASK CLOUDFLARE
    print("Asking Cloudflare...")
    
    # Cloudflare's endpoint for their AI models
    cf_url = f"https://api.cloudflare.com/client/v4/accounts/{CLOUDFLARE_ACCOUNT_ID}/ai/run/@cf/meta/llama-3-8b-instruct"
    cf_headers = {
        "Authorization": f"Bearer {CLOUDFLARE_API_TOKEN}",
        "Content-Type": "application/json"
    }

    if request.image_base64:
        # Cloudflare Vision Model (using their LLaVA model)
        cf_url = f"https://api.cloudflare.com/client/v4/accounts/{CLOUDFLARE_ACCOUNT_ID}/ai/run/@cf/llava-hf/llava-1.5-7b-hf"
        payload = {
            "messages": [
                {"role": "user", "content": [
                    {"type": "text", "text": question},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{request.image_base64}"}}
                ]}
            ]
        }
    else:
        # Cloudflare Text Model
        system_prompt = "You are an expert Senior Web Developer. Use the provided HTML/CSS template context to help the user build or modify their website. Always output clean, modern, and fully responsive code. When providing HTML, CSS, or JS, always put it inside a single markdown code block so it can be easily previewed."
        payload = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Template Context:\n{template_context}\n\nUser Request: {question}"}
            ]
        }

    # Make the request to Cloudflare
    cf_response = requests.post(cf_url, headers=cf_headers, json=payload)
    cf_data = cf_response.json()
    
    # Cloudflare returns the text in data['result']['response']
    answer = cf_data.get("result", {}).get("response", "Sorry, I couldn't generate a response.")
    
    return {"answer": answer}

@app.get("/")
def read_root():
    return {"status": "Backend is running"}
