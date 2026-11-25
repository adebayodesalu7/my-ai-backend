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
SERPER_API_KEY = os.environ.get("SERPER_API_KEY")
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
COHERE_API_KEY = os.environ.get("COHERE_API_KEY")

client = Groq(api_key=GROQ_API_KEY)
co = cohere.Client(COHERE_API_KEY)

pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index("my-templates-v3")

class ChatRequest(BaseModel):
    question: str
    image_base64: Optional[str] = None  # We added an optional image field

@app.post("/chat")
async def chat(request: ChatRequest):
    question = request.question

    # 1. SEARCH PINECONE (Only if there is no image, to save tokens)
    template_context = ""
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

    # 2. ASK GROQ
    print("Asking Groq...")
    
    # IF THERE IS AN IMAGE, use a Vision Model
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
        model_to_use = "qwen/qwen3.8-27b" # Free vision model on Groq
        
    # IF NO IMAGE, use the Text Model
    else:
        messages = [
            {"role": "system", "content": "You are an expert Senior Web Developer. Use the provided HTML/CSS template context to help the user build or modify their website. Always output clean, modern, and fully responsive code. When providing HTML, CSS, or JS, always put it inside a single markdown code block so it can be easily previewed. If the answer isn't in the context, use your own vast coding knowledge."},
            {"role": "user", "content": f"Template Context:\n{template_context}\n\nUser Request: {question}"}
        ]
        model_to_use = "openai/gpt-oss-120b"

    chat_completion = client.chat.completions.create(
        messages=messages,
        model=model_to_use,
    )
    
    answer = chat_completion.choices[0].message.content
    return {"answer": answer}

@app.get("/")
def read_root():
    return {"status": "Backend is running"}
