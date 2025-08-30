# Learning RAG Systems - Vulnerability Search Demo

## What This Is
This is my learning project for understanding how Retrieval-Augmented Generation (RAG) works. I built a simple vulnerability search system to experiment with semantic search and vector databases. It's a way to learn how these AI technologies actually work in practice.

## What I Learned

### Basic RAG Concepts
- **Vector Embeddings**: Converting text to numbers that capture meaning
- **Similarity Search**: Finding similar content using vector math
- **Chunking**: Breaking text into pieces for better search
- **Hybrid Approach**: Combining keyword search with semantic search

### Technologies I Experimented With
- **ChromaDB**: Vector database (easier than I expected!)
- **sentence-transformers**: Pre-trained models for text embeddings
- **SQLite**: Regular database for structured data
- **FastAPI**: Web framework for APIs (pretty straightforward)

### What Actually Works vs What Doesn't
- Basic semantic search works well for finding related vulnerabilities
- Simple keyword search is still useful for exact matches
- Similarity thresholds need lots of tweaking to get good results
- Small datasets make it hard to see the real benefits
- My first attempts at "advanced analysis" were mostly overcomplicated

## How It Actually Works (Simplified)

1. **Setup**: Load vulnerability descriptions into both ChromaDB (for vectors) and SQLite (for metadata)
2. **Search**: When you search, it converts your query to a vector and finds similar vectors
3. **Results**: Returns the most similar vulnerabilities with scores
4. **Comparison**: You can compare keyword vs semantic search to see the differences

The basic idea: instead of just matching words, it tries to understand what you mean and find conceptually similar things.

## Running the Code

### Prerequisites
You'll need Python and these packages:
```bash
pip install chromadb sentence-transformers fastapi uvicorn pydantic
```

### What to Try
```bash
# Simple demo with sample data
python simple_demo.py

# Enhanced version with semantic search
python enhanced_demo.py

# Compare different search methods
python semantic_test.py
```

## Example of What You'll See

When you search for "memory corruption bugs", keyword search might not find anything, but semantic search could find "Buffer Overflow" vulnerabilities because it understands they're related concepts.

```python
# This is basically what the code does
from enhanced_demo import EnhancedRAGDatabase

db = EnhancedRAGDatabase()
# ... add some sample vulnerabilities ...

# Try different search approaches
keyword_results = db.keyword_search("memory corruption")  # Might miss things
semantic_results = db.semantic_search("memory corruption")  # Finds related concepts
```

## What I'm Still Figuring Out

- **Similarity Thresholds**: Getting the right balance between too strict (no results) and too loose (irrelevant results)
- **Chunking Strategy**: How to break up text for the best search results
- **Model Selection**: Whether different embedding models work better for security content
- **Hybrid Search**: When to use keyword vs semantic vs both together

## Honest Assessment

This is a learning project, not production code. It works for experimenting with RAG concepts, but it's:
- Limited to a few sample vulnerabilities
- Not optimized for performance
- Missing proper error handling in many places
- More of a proof-of-concept than a real tool

But it's a good way to understand how modern AI search actually works!
