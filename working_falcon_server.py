#!/usr/bin/env python3
"""
Working Quantized Falcon 7B Server - The version that successfully loads to 100%
Uses 8-bit quantization and loads completely
"""

import asyncio
import json
import logging
import time
import threading
import gc
from pathlib import Path
from typing import Dict, Optional, Any, List

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Working Quantized Falcon 7B", version="1.0.0")

# Global state
model = None
tokenizer = None
model_loading = False
model_loaded = False
model_error = None
loading_strategy = None

class CompressRequest(BaseModel):
    input: str

class CompressResponse(BaseModel):
    compressed: str
    protectedSpans: List[str]
    stats: Dict[str, int]

def load_model_with_working_quantization():
    """Load model with the quantization method that works"""
    global model, tokenizer, model_loading, model_loaded, model_error, loading_strategy
    
    try:
        model_loading = True
        logger.info("🚀 Starting quantized model loading...")
        
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
        
        # Clear memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        model_path = Path("./models/falcon-7b")
        
        logger.info("📝 Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            padding_side="left"
        )
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        logger.info("🔢 Attempting 8-bit quantized loading...")
        
        # Use the exact configuration that worked before
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0,
            llm_int8_enable_fp32_cpu_offload=True
        )
        
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            torch_dtype=torch.float16
        )
        
        loading_strategy = "8-bit quantized"
        model_loaded = True
        model_loading = False
        
        # Report final status
        device = next(model.parameters()).device
        dtype = next(model.parameters()).dtype
        logger.info(f"✅ Model loaded with 8-bit quantization!")
        logger.info(f"🎯 Model ready! Strategy: {loading_strategy}")
        logger.info(f"   Device: {device}, Type: {dtype}")
        
        # Report memory usage
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated(0) / 1024**3
            reserved = torch.cuda.memory_reserved(0) / 1024**3
            logger.info(f"   GPU Memory: {allocated:.1f}GB allocated, {reserved:.1f}GB reserved")
        
        return True
        
    except Exception as e:
        model_error = str(e)
        model_loading = False
        model_loaded = False
        logger.error(f"❌ Model loading failed: {e}")
        return False

def count_tokens_safe(text: str) -> int:
    """Safe token counting"""
    if tokenizer is None:
        return len(text) // 4
    
    try:
        tokens = tokenizer.encode(text, add_special_tokens=True, truncation=True, max_length=2048)
        return len(tokens)
    except:
        return len(text) // 4

def compress_with_working_falcon(input_text: str) -> Optional[str]:
    """Working compression method that avoids tokenizer issues"""
    if not model_loaded or model is None or tokenizer is None:
        return None
    
    try:
        import torch
        
        # Few-shot prompting with examples to teach the model the exact format
        prompt = f"""Task: Compress prompts by removing redundant words. Return ONLY the compressed version.

Example 1:
Original: Please help me understand how to create a comprehensive guide with detailed explanations.
Compressed: Create comprehensive guide with detailed explanations.

Example 2:
Original: Can you please provide me with a very thorough explanation of machine learning algorithms?
Compressed: Explain machine learning algorithms thoroughly.

Example 3:
Original: I would really like you to help me understand neural networks step by step.
Compressed: Explain neural networks step by step.

Now compress this:
Original: {input_text}
Compressed:"""
        
        logger.info(f"Compressing text of length {len(input_text)}")
        
        # Use encode directly to avoid parameter issues
        input_ids = tokenizer.encode(
            prompt,
            add_special_tokens=True,
            truncation=True,
            max_length=1200,
            return_tensors="pt"
        )
        
        logger.info(f"Input tokens: {input_ids.shape[1]}")
        
        # Move to model device
        device = next(model.parameters()).device
        input_ids = input_ids.to(device)
        
        # Generate with constrained parameters to prevent rambling
        logger.info("Starting generation...")
        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                max_new_tokens=50,  # Much shorter to force conciseness
                do_sample=False,  # Greedy decoding for consistency
                temperature=0.1,  # Low temperature for focused output
                top_p=0.8,  # Slightly constrained sampling
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.2,  # Prevent repetition
                early_stopping=True,  # Stop at natural completion
                num_beams=1  # No beam search for speed
            )
        
        logger.info("Generation complete, decoding...")
        
        # Decode result
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        logger.info(f"Generated text length: {len(generated_text)}")
        
        # Extract compressed text
        compressed = None
        if "Compressed:" in generated_text:
            compressed = generated_text.split("Compressed:")[-1].strip()
            # Take only the first line of the compressed result
            compressed = compressed.split('\n')[0].strip()
        
        # Fallback extraction methods
        if not compressed:
            for marker in ["Short:", "Concise:", "Brief:"]:
                if marker in generated_text:
                    compressed = generated_text.split(marker)[-1].strip()
                    compressed = compressed.split('\n')[0].strip()
                    break
        
        if not compressed:
            # Take the last meaningful line
            lines = generated_text.strip().split('\n')
            compressed = lines[-1].strip() if lines else None
        
        # Quality check
        if compressed and len(compressed) > 5 and compressed != input_text.strip():
            logger.info(f"Compression successful: {len(compressed)} chars")
            return compressed
        else:
            logger.warning("No valid compression found")
            return None
        
    except Exception as e:
        logger.error(f"Compression failed: {e}")
        import traceback
        traceback.print_exc()
        return None
    finally:
        # Clean up GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

@app.on_event("startup")
async def startup_event():
    """Start model loading"""
    logger.info("🚀 Starting Quantized Falcon 7B Server...")
    threading.Thread(target=load_model_with_working_quantization, daemon=True).start()
    logger.info("✅ Server started! Quantized model loading in background...")

@app.get("/health")
async def health_check():
    """Health check with detailed info"""
    import torch
    
    status = {
        "status": "healthy",
        "model_loading": model_loading,
        "model_loaded": model_loaded,
        "model_error": model_error,
        "loading_strategy": loading_strategy,
        "cuda_available": torch.cuda.is_available()
    }
    
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        total_memory = torch.cuda.get_device_properties(device).total_memory
        allocated_memory = torch.cuda.memory_allocated(device)
        reserved_memory = torch.cuda.memory_reserved(device)
        
        status["gpu_info"] = {
            "device_name": torch.cuda.get_device_name(device),
            "total_memory": f"{total_memory / 1024**3:.1f}GB",
            "allocated_memory": f"{allocated_memory / 1024**3:.1f}GB",
            "reserved_memory": f"{reserved_memory / 1024**3:.1f}GB",
            "free_memory": f"{(total_memory - reserved_memory) / 1024**3:.1f}GB"
        }
    
    if model_loaded:
        status["model_device"] = str(next(model.parameters()).device)
        status["model_dtype"] = str(next(model.parameters()).dtype)
    
    return status

@app.post("/compress", response_model=CompressResponse)
async def compress_text(request: CompressRequest):
    """Working compression endpoint"""
    if model_loading:
        raise HTTPException(
            status_code=503, 
            detail=f"Model still loading with {loading_strategy or 'quantization'}, please wait..."
        )
    
    if not model_loaded:
        error_msg = f"Model not loaded. Strategy: {loading_strategy}. Error: {model_error}" if model_error else "Model not loaded"
        raise HTTPException(status_code=503, detail=error_msg)
    
    start_time = time.time()
    input_text = request.input.strip()
    
    if not input_text:
        raise HTTPException(status_code=400, detail="Empty input")
    
    # Count tokens safely
    before_tokens = count_tokens_safe(input_text)
    
    # Compress
    try:
        compressed = compress_with_working_falcon(input_text)
        
        if compressed is None:
            raise HTTPException(status_code=500, detail="Compression failed - no valid output generated")
        
        after_tokens = count_tokens_safe(compressed)
        saved_tokens = max(0, before_tokens - after_tokens)
        reduction_pct = int((saved_tokens / before_tokens * 100)) if before_tokens > 0 else 0
        
        processing_time = time.time() - start_time
        
        logger.info(f"✅ {loading_strategy}: {before_tokens}→{after_tokens} tokens ({reduction_pct}%) in {processing_time:.1f}s")
        
        return CompressResponse(
            compressed=compressed,
            protectedSpans=[],
            stats={
                "beforeTokens": before_tokens,
                "afterTokens": after_tokens,
                "saved": saved_tokens,
                "pct": reduction_pct
            }
        )
        
    except Exception as e:
        processing_time = time.time() - start_time
        logger.error(f"❌ Compression failed after {processing_time:.1f}s: {e}")
        raise HTTPException(status_code=500, detail=f"Compression failed: {str(e)}")

@app.get("/")
async def root():
    """Server info"""
    return {
        "service": "Working Quantized Falcon 7B Server",
        "version": "1.0.0",
        "model_loading": model_loading,
        "model_loaded": model_loaded,
        "loading_strategy": loading_strategy
    }

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8081)
    args = parser.parse_args()
    
    uvicorn.run("working_falcon_server:app", host=args.host, port=args.port, log_level="info")
