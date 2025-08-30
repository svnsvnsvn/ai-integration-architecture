# AI Fundamentals Learning Journey

My hands-on exploration of AI concepts thropython semantic_test.py      # Compare search methods
```

## What I Learned Building Thisthree practical projects. Each one taught me something different about how modern AI systems actually work.

## What I Built

Three components that helped me understand core AI patterns:

1. **Weather API (MCP)** - Learning API integration with security basics
2. **Document Processing (LangGraph)** - Understanding AI workflow orchestration  
3. **Search System (RAG)** - Exploring semantic search and vector databases

## How They Connect

```
Simple → Complex → Practical

MCP Server              LangGraph Pipeline           RAG System
┌─────────────────┐     ┌─────────────────────┐     ┌─────────────────┐
│ Basic API       │ →   │ Multi-step AI       │ →   │ AI + Knowledge  │
│ + Security      │     │ + State Management  │     │ + Vector Search │
│ + Rate Limiting │     │ + Error Recovery    │     │ + Semantic AI   │
└─────────────────┘     └─────────────────────┘     └─────────────────┘
```

Each project builds on concepts from the previous one, but they can also be explored independently.

## Getting Started

### You'll Need
- Python 3.12+ 

### Setup
```bash
# Clone and setup
git clone <this-repo>
cd ai_fundamentals_demo

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
```

### Try Each Component

#### 1. Weather API (Start Here)
```bash
cd MCP
# Get a free API key from openweathermap.org
echo "OPENWEATHER_API_KEY=your_key_here" > .env
pip install fastapi uvicorn httpx python-dotenv pydantic

python weather_server.py      # Start server
python test_client.py         # Test it (new terminal)
```

#### 2. Document Processing 
```bash
cd model_w_lang
pip install -r requirements.txt

# Option A: Use free local AI 
brew install ollama           # or download from ollama.ai
ollama serve                  # start ollama (new terminal)
ollama pull llama3.2:1b      # download small model

python demo.py               # Process sample documents

# Option B: Use paid AI (better quality)
# Add OPENAI_API_KEY=your_key to .env and set AI_PROVIDER=openai
```

#### 3. Search System
```bash
cd model_w_rag
pip install -r requirements.txt

python simple_demo.py        # Basic keyword search
python enhanced_demo.py      # AI-powered semantic search
python semantic_test.py      # Compare search methods
```

## � **What I Learned Building This**

### The Good Stuff
- **Real AI is accessible**: Ollama makes powerful models free and local
- **Workflows matter**: LangGraph helps organize complex AI operations  
- **Search isn't just keywords**: Vector embeddings find meaning, not just matches
- **APIs need security**: Rate limiting and validation are essential

### The Challenges  
- **Python versions**: AI libraries are picky (3.12 works best)
- **Model quality**: Free models (like Llama 3.2 1B) are good but not perfect
- **Vector search**: Similarity thresholds need tuning for your data

### Practical Takeaways
- Start simple (weather API) before complex (RAG systems)
- Local AI (Ollama) is great for learning and prototyping
- Real document processing needs handling edge cases
- Semantic search requires good embeddings and data preprocessing

This isn't production-ready code - it's learning code. But the concepts are solid and you can build real applications from these foundations.

## What You'll Learn

Each component teaches different AI concepts through hands-on code:

### MCP - API Integration
- **File**: `MCP/weather_server.py` (180 lines)
- **Concepts**: FastAPI, rate limiting, input validation, environment variables
- **Skills**: Building secure web APIs, handling external services
- **Reality Check**: This is real API integration - you'll need an actual API key

### Model with LangGraph - Workflow Management  
- **File**: `model_w_lang/document_pipeline.py` (200 lines)
- **Concepts**: State machines, multi-step processing, LLM integration
- **Skills**: Orchestrating complex AI workflows, document analysis
- **Reality Check**: Uses real AI (Ollama or OpenAI) - actual language understanding

### Model with RAG - Semantic Search
- **File**: `model_w_rag/enhanced_demo.py` (150 lines) 
- **Concepts**: Vector embeddings, similarity search, hybrid retrieval
- **Skills**: Building intelligent search systems, semantic understanding
- **Reality Check**: True semantic search - finds meaning, not just keywords

### Learning Path
1. **Start with MCP** - easiest to understand, good introduction to APIs
2. **Try Document Processing** - see real AI in action with your documents  
3. **Explore RAG** - understand how modern AI search actually works

Each component stands alone, so you can explore in any order!

## Contributing & Next Steps

### If You Want to Improve This
- **Add error handling**: The code is basic - production apps need more robust error handling
- **Test coverage**: Write tests for each component (currently minimal testing)
- **UI interfaces**: Add web frontends to make the demos more visual
- **More AI providers**: Add support for Claude, Gemini, etc.

### Ideas for Extensions
- **Combine components**: Use the weather API data in the RAG system
- **Real datasets**: Process actual documents instead of sample files  
- **Performance**: Add caching, async processing, better vector search
- **Deployment**: Containerize and deploy to cloud platforms

### Resources for Learning More
- [FastAPI Documentation](https://fastapi.tiangolo.com/) - for API development
- [LangGraph Guide](https://langchain-ai.github.io/langgraph/) - for AI workflows
- [ChromaDB Docs](https://docs.trychroma.com/) - for vector databases
- [Ollama Models](https://ollama.ai/library) - for local AI models

Feel free to fork, experiment, and make this code better! The goal is learning, not perfection.

---

Built for learning AI fundamentals through practical coding
