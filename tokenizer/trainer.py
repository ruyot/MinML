#!/usr/bin/env python3
"""
Tokenizer training script using Hugging Face Tokenizers.
Trains a BPE tokenizer on a corpus of prompts.
"""

import argparse
import os
from pathlib import Path
from typing import List, Optional

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing


def load_corpus(corpus_path: str) -> List[str]:
    """Load text corpus from file."""
    with open(corpus_path, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]
    return lines


def create_tokenizer() -> Tokenizer:
    """Create a new BPE tokenizer with basic configuration."""
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    
    # Add special tokens
    tokenizer.post_processor = TemplateProcessing(
        single="[CLS] $A [SEP]",
        pair="[CLS] $A [SEP] $B:1 [SEP]:1",
        special_tokens=[
            ("[CLS]", 1),
            ("[SEP]", 2),
        ],
    )
    
    return tokenizer


def train_tokenizer(
    corpus_path: str,
    output_dir: str = "models/custom_tokenizer",
    vocab_size: int = 8000,
    min_frequency: int = 2,
    special_tokens: Optional[List[str]] = None
) -> Tokenizer:
    """
    Train a BPE tokenizer on the given corpus.
    
    Args:
        corpus_path: Path to the text corpus file
        output_dir: Directory to save the trained tokenizer
        vocab_size: Size of the vocabulary
        min_frequency: Minimum frequency for subword inclusion
        special_tokens: List of special tokens to add
    
    Returns:
        Trained tokenizer instance
    """
    if special_tokens is None:
        special_tokens = ["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]", "[COMPRESS]", "[DECOMPRESS]"]
    
    print(f"Loading corpus from {corpus_path}")
    corpus = load_corpus(corpus_path)
    print(f"Loaded {len(corpus)} lines")
    
    print("Initializing tokenizer...")
    tokenizer = create_tokenizer()
    
    print("Setting up trainer...")
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=special_tokens,
        show_progress=True
    )
    
    print("Training tokenizer...")
    tokenizer.train_from_iterator(corpus, trainer)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save tokenizer
    tokenizer_path = os.path.join(output_dir, "tokenizer.json")
    tokenizer.save(tokenizer_path)
    print(f"Tokenizer saved to {tokenizer_path}")
    
    # Save vocabulary for inspection
    vocab_path = os.path.join(output_dir, "vocab.txt")
    vocab = tokenizer.get_vocab()
    with open(vocab_path, 'w', encoding='utf-8') as f:
        for token, id_ in sorted(vocab.items(), key=lambda x: x[1]):
            f.write(f"{token}\t{id_}\n")
    print(f"Vocabulary saved to {vocab_path}")
    
    # Test tokenizer with sample text
    sample_text = "Please explain machine learning concepts in simple terms."
    tokens = tokenizer.encode(sample_text)
    print(f"\nSample tokenization:")
    print(f"Text: {sample_text}")
    print(f"Tokens: {tokens.tokens}")
    print(f"IDs: {tokens.ids}")
    print(f"Token count: {len(tokens.ids)}")
    
    return tokenizer


def main():
    """Main function for command line usage."""
    parser = argparse.ArgumentParser(description="Train a BPE tokenizer on a text corpus")
    parser.add_argument("--corpus", required=True, help="Path to the text corpus file")
    parser.add_argument("--output-dir", default="models/custom_tokenizer", 
                        help="Output directory for the trained tokenizer")
    parser.add_argument("--vocab-size", type=int, default=8000, 
                        help="Vocabulary size")
    parser.add_argument("--min-frequency", type=int, default=2, 
                        help="Minimum frequency for subword inclusion")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.corpus):
        print(f"Error: Corpus file {args.corpus} not found")
        return 1
    
    try:
        train_tokenizer(
            corpus_path=args.corpus,
            output_dir=args.output_dir,
            vocab_size=args.vocab_size,
            min_frequency=args.min_frequency
        )
        print("Training completed successfully!")
        return 0
    except Exception as e:
        print(f"Error during training: {e}")
        return 1


if __name__ == "__main__":
    exit(main()) 