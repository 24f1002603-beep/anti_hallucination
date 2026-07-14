import os
from typing import List
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from openai import OpenAI

# 1. Initialize FastAPI Application
app = FastAPI(title="SafeAnswer AI Grounded QA API via AI Pipe")

# 2. Enable CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 3. Initialize the OpenAI Client pointing to AI Pipe's OpenRouter Proxy
# Make sure you set AIPIPE_TOKEN in your Render Environment Variables!
AIPIPE_TOKEN = os.environ.get("AIPIPE_TOKEN")

client = OpenAI(
    base_url="https://aipipe.org/openrouter/v1", # Point to AI Pipe instead of OpenAI directly
    api_key=AIPIPE_TOKEN
)

# 4. Define Data Schemas using Pydantic
class Chunk(BaseModel):
    chunk_id: str
    text: str

class QARequest(BaseModel):
    question: str
    chunks: List[Chunk]

class QAResponse(BaseModel):
    answer: str
    citations: List[str]
    confidence: float = Field(..., description="Calibrated confidence score between 0.0 and 1.0")
    answerable: bool


# 5. Core RAG Logic with Structured Outputs & Anti-Hallucination Guardrails
def get_grounded_answer(question: str, chunks: List[Chunk]) -> QAResponse:
    # Format the context chunks into a clean, structured text block
    formatted_context = ""
    for c in chunks:
        formatted_context += f"Chunk ID: {c.chunk_id}\nText: {c.text}\n---\n"
    
    # Strict compliance system prompt to eliminate external knowledge or guessing
    system_prompt = (
        "You are a strict, high-reliability compliance AI. Your task is to answer the user's question "
        "EXCLUSIVELY using the provided context chunks. Follow these absolute rules:\n"
        "1. If the question cannot be fully answered using ONLY the provided chunks, you MUST set "
        "'answerable' to false, 'answer' to exactly 'I don't know', 'citations' to [], and 'confidence' to 0.2.\n"
        "2. Do NOT use any outside knowledge. If the text doesn't say it explicitly, it is not true.\n"
        "3. Only include chunk IDs in the 'citations' array if that specific chunk explicitly contains the facts used to answer.\n"
        "4. If answerable is true, confidence must be greater than or equal to 0.8."
    )
    
    user_prompt = f"Context Chunks:\n{formatted_context}\n\nQuestion: {question}"
    
    try:
        # Request completion via AI Pipe proxy using OpenRouter model naming syntax
        completion = client.beta.chat.completions.parse(
            model="openai/gpt-4o-mini", # Standard format for models mapped through AI Pipe
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            response_format=QAResponse,
            temperature=0.0  # Keep it deterministic
        )
        return completion.choices[0].message.parsed
        
    except Exception as e:
        # Fallback safeguard
        return QAResponse(
            answer="I don't know",
            citations=[],
            confidence=0.0,
            answerable=False
        )


# 6. Public POST Endpoint
@app.post("/api/grounded-qa", response_model=QAResponse)
async def grounded_qa_endpoint(request: QARequest):
    if not request.question.strip() or not request.chunks:
        return QAResponse(
            answer="I don't know",
            citations=[],
            confidence=0.0,
            answerable=False
        )
    
    result = get_grounded_answer(request.question, request.chunks)
    
    # Post-processing layer to strictly enforce target criteria thresholds
    if not result.answerable or result.answer.strip().lower() == "i don't know":
        return QAResponse(
            answer="I don't know",
            citations=[],
            confidence=min(result.confidence, 0.3),  # Hard threshold verification
            answerable=False
        )
        
    return result