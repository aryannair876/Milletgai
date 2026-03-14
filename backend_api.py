import sys
import re
from pathlib import Path
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# Ensure we can import rag_inference from the same directory
sys.path.append(str(Path(__file__).parent))

# Import the RAG class
from rag_inference import MilletsGAI_RAG

app = FastAPI(title="MilletsGAI API")

# Allow CORS for localhost:3000
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model instance
rag_system = None

@app.on_event("startup")
async def startup_event():
    global rag_system
    print("🚀 Starting MilletsGAI Backend...")
    # Initialize the RAG system (loads Llama 3 8B + ChromaDB)
    # This might take a minute
    try:
        rag_system = MilletsGAI_RAG()
        print("✅ MilletsGAI Backend Ready!")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        # We don't exit here so the server still runs, but requests will fail
        pass

class ChatRequest(BaseModel):
    message: str
    model_type: Optional[str] = "millet"  # "millet" or "base"
    mode: Optional[str] = "fast"  # "fast" or "thinking"

class ChatResponse(BaseModel):
    response: str
    citations: List[str]
    confidence: int
    thinking: Optional[str] = None
    is_complex: bool = False

def parse_response(full_text: str):
    """
    Splits the LLM output into the main response and the citation section.
    Returns (main_text, citations_list).
    """
    # specific split marker from rag_inference.py
    split_marker = "---"
    
    if split_marker in full_text:
        parts = full_text.split(split_marker)
        main_text = parts[0].strip()
        source_section = parts[-1].strip()
        
        # Extract individual sources from the bullet points or lines
        # Currently the backend returns a formatted text block for sources.
        # We'll try to extract lines that look like source names.
        citations = []
        for line in source_section.split("\n"):
            line = line.strip()
            if line.startswith("-") or line.startswith("*"):
                clean_line = line.lstrip("-* ").strip()
                if clean_line:
                    citations.append(clean_line)
        
        # If no bullet points found, just add a generic one
        if not citations:
            citations = ["MilletsGAI Knowledge Base"]
            
        return main_text, citations[:3] # limit to 3 for UI
    
    return full_text.strip(), ["MilletsGAI Knowledge Base"]

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    global rag_system
    if not rag_system:
        raise HTTPException(status_code=503, detail="AI Model is still loading or failed to load.")
    
    try:
        # Determine if we should use the adapter
        use_adapter = (request.model_type.lower() != "base")
        if not use_adapter:
            print(f"📉 Using BASE Llama 3 model for this request")
            
        # Determine if we should force CoT based on mode
        force_cot = None  # Auto-detect by default
        max_tokens = 400
        if request.mode == "thinking":
            force_cot = True
            max_tokens = 1000
        elif request.mode == "fast":
            force_cot = False
            max_tokens = 350
        
        # Run the RAG query - now returns a dict
        result = rag_system.query(
            request.message, 
            use_adapter=use_adapter,
            force_cot=force_cot,
            max_tokens=max_tokens
        )
        
        # Extract fields from result dict
        full_response = result.get("response", "")
        thinking = result.get("thinking", None)
        is_complex = result.get("is_complex", False)
        
        main_text, citations = parse_response(full_response)
        
        return ChatResponse(
            response=main_text,
            citations=citations,
            confidence=95,  # Mock confidence for now
            thinking=thinking,
            is_complex=is_complex
        )
        
    except Exception as e:
        print(f"Error processing request: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
