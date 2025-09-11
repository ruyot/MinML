#!/usr/bin/env python3
"""
Falcon 7B Model Server for MinML
Provides HTTP endpoints for prompt compression using Falcon 7B model.
"""

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Dict, Optional, Any, List

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Falcon 7B MinML Server", version="1.0.0")

# Global model and tokenizer
model = None
tokenizer = None
model_loaded = False

class CompressRequest(BaseModel):
    input: str

class AnalyzeRequest(BaseModel):
    input: str

class CompressResponse(BaseModel):
    compressed: str
    protectedSpans: List[str]
    stats: Dict[str, int]

class AnalyzeResponse(BaseModel):
    slots: Optional[Dict[str, Any]] = None
    protectedSpans: Optional[List[str]] = None

def count_tokens(text: str) -> int:
    """Count tokens in text using the tokenizer."""
    if tokenizer is None:
        # Fallback estimation
        return len(text) // 4
    
    try:
        tokens = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048)
        return tokens.input_ids.shape[1]
    except Exception as e:
        logger.warning(f"Token counting failed: {e}")
        return len(text) // 4

def load_model():
    """Load the Falcon 7B model and tokenizer."""
    global model, tokenizer, model_loaded
    
    try:
        model_path = Path("./models/falcon-7b")  # Adjust this path after you extract the tar file
        
        if not model_path.exists():
            logger.error(f"Model path does not exist: {model_path}")
            return False
        
        logger.info("Loading Falcon 7B tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            padding_side="left"
        )
        
        # Add pad token if it doesn't exist
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        logger.info("Loading Falcon 7B model...")
        
        # Try GPU loading with memory limit, fallback to CPU if needed
        try:
            if torch.cuda.is_available():
                logger.info("Attempting GPU loading with memory constraints...")
                model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    trust_remote_code=True,
                    low_cpu_mem_usage=True,
                    max_memory={0: "10GB"}  # Limit GPU memory to avoid OOM
                )
                logger.info("✅ Model loaded on GPU with memory limits")
            else:
                raise Exception("No CUDA available")
                
        except Exception as gpu_error:
            logger.warning(f"GPU loading failed: {gpu_error}")
            logger.info("Falling back to CPU loading...")
            
            # Clear any allocated GPU memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float32,
                device_map=None,  # Force CPU
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
            logger.info("✅ Model loaded on CPU")
        
        logger.info(f"Model loaded successfully. Device: {next(model.parameters()).device}")
        model_loaded = True
        return True
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        model_loaded = False
        return False

def compress_with_falcon(input_text: str) -> Optional[str]:
    """Use Falcon 7B to compress the input text."""
    if not model_loaded or model is None or tokenizer is None:
        return None
    
    try:
        # Create a specialized compression prompt for Falcon
        compression_prompt = f"""You are an intelligent prompt compression system. Your job is to compress prompts by removing redundancy, unnecessary words, and filler phrases while preserving all essential meaning, technical terms, and specific details.

Rules:
- Remove redundant words like "please", "can you", "I would like", "very", "really"
- Keep all technical terms and specific details
- Maintain the core request and context
- Remove excessive politeness but keep the intent
- Preserve numbers, names, and important qualifiers
- Make it concise but complete

Original prompt: {input_text}

Compressed prompt:"""
        
        # Tokenize input
        inputs = tokenizer(
            compression_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=1800,  # Leave room for generation
            padding=True
        )
        
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        # Generate compressed version
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=500,
                do_sample=True,
                temperature=0.3,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        # Decode and extract the compressed text
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract just the compressed part after "Compressed prompt:"
        if "Compressed prompt:" in generated_text:
            compressed = generated_text.split("Compressed prompt:")[-1].strip()
            return compressed if compressed else None
        
        return None
        
    except Exception as e:
        logger.error(f"Compression failed: {e}")
        return None

@app.on_event("startup")
async def startup_event():
    """Load the model on startup."""
    logger.info("Starting Falcon 7B server...")
    success = load_model()
    if success:
        logger.info("✅ Falcon 7B server ready!")
    else:
        logger.warning("⚠️ Falcon 7B model failed to load - server will return errors")

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy" if model_loaded else "model_not_loaded",
        "model_loaded": model_loaded,
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0
    }

@app.post("/compress", response_model=CompressResponse)
async def compress_text(request: CompressRequest):
    """Compress text using Falcon 7B."""
    if not model_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    start_time = time.time()
    input_text = request.input.strip()
    
    if not input_text:
        raise HTTPException(status_code=400, detail="Empty input text")
    
    # Count original tokens
    before_tokens = count_tokens(input_text)
    
    # Compress using Falcon 7B
    compressed = compress_with_falcon(input_text)
    
    if compressed is None:
        raise HTTPException(status_code=500, detail="Compression failed")
    
    # Count compressed tokens
    after_tokens = count_tokens(compressed)
    saved_tokens = max(0, before_tokens - after_tokens)
    reduction_pct = int((saved_tokens / before_tokens * 100)) if before_tokens > 0 else 0
    
    processing_time = time.time() - start_time
    
    logger.info(f"Compression: {before_tokens} -> {after_tokens} tokens ({reduction_pct}%) in {processing_time:.2f}s")
    
    return CompressResponse(
        compressed=compressed,
        protectedSpans=[],  # Falcon 7B handles this internally
        stats={
            "beforeTokens": before_tokens,
            "afterTokens": after_tokens,
            "saved": saved_tokens,
            "pct": reduction_pct
        }
    )

@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze_text(request: AnalyzeRequest):
    """Analyze text structure (placeholder for now)."""
    # For now, return empty analysis
    # You could extend this to use Falcon 7B for text analysis
    return AnalyzeResponse(
        slots=None,
        protectedSpans=None
    )

@app.get("/")
async def root():
    """Root endpoint with server info."""
    return {
        "service": "Falcon 7B MinML Server",
        "version": "1.0.0",
        "model_loaded": model_loaded,
        "endpoints": {
            "health": "GET /health",
            "compress": "POST /compress",
            "analyze": "POST /analyze"
        }
    }

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Falcon 7B MinML Server")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8081, help="Port to bind to")
    parser.add_argument("--workers", type=int, default=1, help="Number of worker processes")
    
    args = parser.parse_args()
    
    uvicorn.run(
        "falcon_server:app",
        host=args.host,
        port=args.port,
        workers=args.workers,
        log_level="info"
    )
