"""
Utilities for loading and using trained tokenizers.
"""

import os
from typing import List, Optional, Union

from tokenizers import Tokenizer
from transformers import AutoTokenizer


def load_tokenizer(tokenizer_path: str) -> Union[Tokenizer, AutoTokenizer]:
    """
    Load a tokenizer from the specified path.
    
    Args:
        tokenizer_path: Path to the tokenizer directory or file
    
    Returns:
        Loaded tokenizer instance
    """
    if os.path.isdir(tokenizer_path):
        # Check for custom trained tokenizer
        json_path = os.path.join(tokenizer_path, "tokenizer.json")
        if os.path.exists(json_path):
            print(f"Loading custom tokenizer from {json_path}")
            return Tokenizer.from_file(json_path)
        else:
            # Try loading as Hugging Face tokenizer
            try:
                print(f"Loading Hugging Face tokenizer from {tokenizer_path}")
                return AutoTokenizer.from_pretrained(tokenizer_path)
            except Exception as e:
                raise ValueError(f"Could not load tokenizer from {tokenizer_path}: {e}")
    elif os.path.isfile(tokenizer_path):
        # Load from JSON file
        print(f"Loading tokenizer from {tokenizer_path}")
        return Tokenizer.from_file(tokenizer_path)
    else:
        # Try loading as model name from Hugging Face
        try:
            print(f"Loading Hugging Face tokenizer: {tokenizer_path}")
            return AutoTokenizer.from_pretrained(tokenizer_path)
        except Exception as e:
            raise ValueError(f"Could not load tokenizer {tokenizer_path}: {e}")


def get_token_count(text: str, tokenizer: Union[Tokenizer, AutoTokenizer]) -> int:
    """
    Get the number of tokens for a given text.
    
    Args:
        text: Input text to tokenize
        tokenizer: Tokenizer instance
    
    Returns:
        Number of tokens
    """
    if isinstance(tokenizer, Tokenizer):
        # Custom tokenizer
        encoded = tokenizer.encode(text)
        return len(encoded.ids)
    else:
        # Hugging Face tokenizer
        encoded = tokenizer.encode(text, add_special_tokens=True)
        return len(encoded)


def tokenize_text(text: str, tokenizer: Union[Tokenizer, AutoTokenizer]) -> List[str]:
    """
    Tokenize text and return list of tokens.
    
    Args:
        text: Input text to tokenize
        tokenizer: Tokenizer instance
    
    Returns:
        List of tokens
    """
    if isinstance(tokenizer, Tokenizer):
        # Custom tokenizer
        encoded = tokenizer.encode(text)
        return encoded.tokens
    else:
        # Hugging Face tokenizer
        tokens = tokenizer.tokenize(text)
        return tokens


def encode_text(text: str, tokenizer: Union[Tokenizer, AutoTokenizer]) -> List[int]:
    """
    Encode text to token IDs.
    
    Args:
        text: Input text to encode
        tokenizer: Tokenizer instance
    
    Returns:
        List of token IDs
    """
    if isinstance(tokenizer, Tokenizer):
        # Custom tokenizer
        encoded = tokenizer.encode(text)
        return encoded.ids
    else:
        # Hugging Face tokenizer
        encoded = tokenizer.encode(text, add_special_tokens=True)
        return encoded


def decode_tokens(token_ids: List[int], tokenizer: Union[Tokenizer, AutoTokenizer]) -> str:
    """
    Decode token IDs back to text.
    
    Args:
        token_ids: List of token IDs
        tokenizer: Tokenizer instance
    
    Returns:
        Decoded text
    """
    if isinstance(tokenizer, Tokenizer):
        # Custom tokenizer
        return tokenizer.decode(token_ids)
    else:
        # Hugging Face tokenizer
        return tokenizer.decode(token_ids, skip_special_tokens=True)


def compare_tokenizers(text: str, tokenizer1: Union[Tokenizer, AutoTokenizer], 
                      tokenizer2: Union[Tokenizer, AutoTokenizer]) -> dict:
    """
    Compare tokenization results between two tokenizers.
    
    Args:
        text: Input text to compare
        tokenizer1: First tokenizer
        tokenizer2: Second tokenizer
    
    Returns:
        Comparison results dictionary
    """
    tokens1 = tokenize_text(text, tokenizer1)
    tokens2 = tokenize_text(text, tokenizer2)
    count1 = len(tokens1)
    count2 = len(tokens2)
    
    return {
        "text": text,
        "tokenizer1": {
            "tokens": tokens1,
            "count": count1
        },
        "tokenizer2": {
            "tokens": tokens2,
            "count": count2
        },
        "difference": count1 - count2,
        "compression_ratio": (count1 - count2) / count1 if count1 > 0 else 0
    }


def get_default_tokenizer() -> AutoTokenizer:
    """
    Get a default tokenizer for fallback usage.
    
    Returns:
        Default GPT-2 tokenizer
    """
    return AutoTokenizer.from_pretrained("gpt2") 