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
        try:
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
        except Exception as e:
            print(f"Pinecone error: {e}")

    # 2. ASK CLOUDFLARE
    print("Asking Cloudflare...")
    
    cf_url = f"https://api.cloudflare.com/client/v4/accounts/{CLOUDFLARE_ACCOUNT_ID}/ai/run/@cf/qwen/qwen2.5-coder-32b-instruct"
    cf_headers = {
        "Authorization": f"Bearer {CLOUDFLARE_API_TOKEN}",
        "Content-Type": "application/json"
    }

    if request.image_base64:
        # Cloudflare Vision Model
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
        # Cloudflare Text Model - Using 'prompt' instead of 'messages'
        system_prompt = "You are an expert Senior Web Developer. When the user asks a question, do not immediately write code. Instead, first briefly outline your plan and ask the user for permission to proceed. Only write the full code after the user says 'yes' or gives the go-ahead. Use the provided HTML/CSS template context to help build the website."
        full_prompt = f"{system_prompt}\n\nTemplate Context:\n{template_context}\n\nUser Request: {question}\n\nResponse:"
        payload = {
            "prompt": full_prompt,
            "max_tokens": 2048
        }

    # Make the request to Cloudflare
    cf_response = requests.post(cf_url, headers=cf_headers, json=payload)
    cf_data = cf_response.json()
    
    print(f"Cloudflare Raw Response: {cf_data}") # This will help us debug
    
    # Cloudflare returns the text in data['result']['response']
    answer = cf_data.get("result", {}).get("response", "Sorry, I couldn't generate a response. Check backend logs.")
    
    return {"answer": answer}

@app.get("/")
def read_root():
    return {"status": "Backend is running"}

