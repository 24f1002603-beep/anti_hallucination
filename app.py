import os
import json
from typing import List
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from openai import OpenAI

# 1. Initialize FastAPI Application
app = FastAPI(title="SafeAnswer AI Grounded QA API via AI Pipe")

# 2. Enable CORS Middleware for grading servers
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 3. Initialize OpenAI Client pointing to AI Pipe's OpenRouter endpoint
AIPIPE_TOKEN = os.environ.get("AIPIPE_TOKEN")
client = OpenAI(
    base_url="https://aipipe.org/openrouter/v1", 
    api_key=AIPIPE_TOKEN
)

# 4. Define Pydantic Schemas
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


# 5. RAG Engine mapped for Proxy Routers
def get_grounded_answer(question: str, chunks: List[Chunk]) -> QAResponse:
    formatted_context = ""
    for c in chunks:
        formatted_context += f"Chunk ID: {c.chunk_id}\nText: {c.text}\n---\n"
    
    system_prompt = (
        "You are a strict compliance AI. Your task is to answer the user's question "
        "EXCLUSIVELY using the provided context chunks. Follow these absolute rules:\n"
        "1. If the question cannot be answered using ONLY the provided chunks, you MUST set "
        "'answerable' to false, 'answer' to exactly 'I don't know', 'citations' to [], and 'confidence' to 0.2.\n"
        "2. Do NOT use outside knowledge. If the text doesn't say it explicitly, it is not true.\n"
        "3. Only include chunk IDs in the 'citations' array if that chunk contains the facts used to answer.\n"
        "4. If answerable is true, confidence must be greater than or equal to 0.8.\n\n"
        "You must respond with raw JSON matching this format exactly:\n"
        "{\n"
        "  \"answer\": \"string content or 'I don't know'\",\n"
        "  \"citations\": [\"chunk_id\"],\n"
        "  \"confidence\": 0.95,\n"
        "  \"answerable\": true\n"
        "}"
    )
    
    user_prompt = f"Context Chunks:\n{formatted_context}\n\nQuestion: {question}"
    
    try:
        # Standard OpenAI client completion structured parameter for maximum proxy compatibility
        completion = client.chat.completions.create(
            model="openai/gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            response_format={"type": "json_object"},  # Standard JSON mode supported by AI Pipe proxies
            temperature=0.0
        )
        
        # Parse the output string programmatically into our target Pydantic schema
        raw_json = json.loads(completion.choices[0].message.content)
        return QAResponse(**raw_json)
        
    except Exception as e:
        return QAResponse(
            answer="I don't know",
            citations=[],
            confidence=0.0,
            answerable=False
        )


# 6. Public API Endpoint Mapping
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
    
    # Absolute strict conditional compliance overrides for adversarial queries
    if not result.answerable or result.answer.strip().lower() == "i don't know":
        return QAResponse(
            answer="I don't know",
            citations=[],
            confidence=min(result.confidence, 0.3),  # Guarantee confidence <= 0.3
            answerable=False
        )
        
    return result
