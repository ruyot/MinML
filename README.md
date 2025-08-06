# Minimal - Semantic Compression Layer for LLM Prompts

A high-performance semantic compression system that reduces token usage in LLM prompts and responses while preserving meaning through intelligent compression techniques.

## Features

- **Custom BPE Tokenizer**: Train domain-specific tokenizers on your prompt corpus
- **Multi-layer Compression**: Stop-word removal, keyword extraction, and aggressive shorthand
- **FastAPI Service**: RESTful API for compression and LLM interaction
- **Flexible LLM Integration**: Support for OpenAI API and local models
- **Docker Ready**: Complete containerization and deployment setup
- **Comprehensive Testing**: Full test suite with CI/CD pipeline

## Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd minimal

# Install dependencies
pip install -r requirements.txt

# Download NLTK data (required for stop words)
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt')"
```

### 2. Train Custom Tokenizer

```bash
# Train tokenizer on sample corpus
python tokenizer/trainer.py --corpus data/prompts.txt --vocab-size 8000

# The trained tokenizer will be saved to models/custom_tokenizer/
```

### 3. Start the Compression Service

```bash
# Set your OpenAI API key (optional)
export OPENAI_API_KEY=your_api_key_here

# Start the FastAPI service
uvicorn middleware.main:app --host 0.0.0.0 --port 8000 --reload
```

### 4. Test the API

```bash
# Compress and query example
curl -X POST "http://localhost:8000/compress-and-query" \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Please explain the concept of machine learning in simple terms for a beginner."}'

# Compression only
curl -X POST "http://localhost:8000/compress" \
  -H "Content-Type: application/json" \
  -d '{"text": "Please explain the concept of machine learning in simple terms for a beginner."}'
```

## Project Structure

```
minimal/
├── tokenizer/              # Custom tokenizer training and utilities
│   ├── __init__.py
│   ├── trainer.py          # BPE tokenizer training script
│   └── tokenizer_utils.py  # Tokenizer loading and utilities
├── compression/            # Compression modules
│   ├── __init__.py
│   ├── base.py            # Base compression interface
│   ├── stop_word_remover.py
│   ├── keyword_extractor.py
│   └── shorthand_compressor.py
├── middleware/             # FastAPI service
│   ├── __init__.py
│   ├── main.py            # FastAPI application
│   ├── models.py          # Pydantic models
│   └── compression_pipeline.py
├── examples/               # Usage examples
│   ├── compress_and_query.py
│   └── evaluate_savings.py
├── tests/                  # Test suite
│   ├── __init__.py
│   ├── test_compression.py
│   ├── test_tokenizer.py
│   └── test_middleware.py
├── data/                   # Sample data
│   └── prompts.txt
├── models/                 # Trained models (created after training)
├── .github/workflows/      # CI/CD configuration
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

## Usage Examples

### Python SDK

```python
from compression.compression_pipeline import CompressionPipeline
from tokenizer.tokenizer_utils import load_tokenizer

# Initialize compression pipeline
tokenizer = load_tokenizer("models/custom_tokenizer")
pipeline = CompressionPipeline(tokenizer)

# Compress text
original = "Please explain machine learning concepts"
compressed = pipeline.compress(original)
print(f"Compressed: {compressed}")

# Decompress (approximation)
decompressed = pipeline.decompress(compressed)
print(f"Decompressed: {decompressed}")
```

### API Examples

#### Compression Only
```bash
curl -X POST "http://localhost:8000/compress" \
  -H "Content-Type: application/json" \
  -d '{"text": "Your long prompt here..."}'
```

#### Compress and Query LLM
```bash
curl -X POST "http://localhost:8000/compress-and-query" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Explain quantum computing",
    "model": "gpt-3.5-turbo",
    "use_compression": true
  }'
```

## Docker Deployment

### Build and Run with Docker

```bash
# Build the image
docker build -t minimal-compression .

# Run the container
docker run -p 8000:8000 -e OPENAI_API_KEY=your_key minimal-compression
```

### Docker Compose (with local model)

```bash
# Start all services
docker-compose up -d

# The API will be available at http://localhost:8000
# Local model server at http://localhost:8001
```

## Configuration

### Environment Variables

- `OPENAI_API_KEY`: Your OpenAI API key (optional)
- `MODEL_PATH`: Path to local model directory (optional)
- `TOKENIZER_PATH`: Path to custom tokenizer (default: models/custom_tokenizer)
- `COMPRESSION_LEVEL`: Compression aggressiveness (1-3, default: 2)

### Compression Levels

1. **Level 1**: Stop-word removal only
2. **Level 2**: Stop-words + keyword extraction
3. **Level 3**: Full aggressive compression (stop-words + keywords + shorthand)

## Evaluation

```bash
# Run compression evaluation
python examples/evaluate_savings.py --input data/prompts.txt

# Example output:
# Original tokens: 1250
# Compressed tokens: 687
# Compression ratio: 45.04%
# Estimated cost savings: $0.00282 per request
```

## Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=compression --cov=tokenizer --cov=middleware

# Run specific test modules
pytest tests/test_compression.py -v
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Roadmap

- [ ] Advanced semantic compression using embeddings
- [ ] Support for more local model formats (GGML, ONNX)
- [ ] Real-time compression metrics dashboard
- [ ] Batch processing API
- [ ] Integration with LangChain and other LLM frameworks 