import os
import requests
import json
from fastapi import FastAPI
from pydantic import BaseModel
from groq import Groq

app = FastAPI()

# Get keys from environment variables (we will set these in Render later)
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
SERPER_API_KEY = os.environ.get("SERPER_API_KEY")

client = Groq(api_key=GROQ_API_KEY)

class ChatRequest(BaseModel):
    question: str

@app.post("/chat")
async def chat(request: ChatRequest):
    question = request.question

    # 1. Search Google
    url = "https://google.serper.dev/search"
    payload = json.dumps({"q": question})
    headers = {'X-API-KEY': SERPER_API_KEY, 'Content-Type': 'application/json'}
    response = requests.post(url, headers=headers, data=payload)
    results = response.json()

    search_context = ""
    if "organic" in results:
        for result in results["organic"][:3]:
            search_context += f"Title: {result.get('title', '')}\nSnippet: {result.get('snippet', '')}\n\n"

    # 2. Ask Groq
    chat_completion = client.chat.completions.create(
        messages=[
            {"role": "system", "content": "You are an incredibly smart, helpful, and conversational AI assistant. Answer the user's question using the search results, but feel free to use your own vast knowledge to explain things clearly and deeply."},
            {"role": "user", "content": f"Results:\n{search_context}\n\nQuestion: {question}"}
        ],
        model="openai/gpt-oss-120b", # The model that worked for you
    )
    
    answer = chat_completion.choices[0].message.content
    return {"answer": answer}

@app.get("/")
def read_root():
    return {"status": "Backend is running"}