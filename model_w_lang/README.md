# Learning LangGraph - Document Processing Workflow

This is my experiment with LangGraph to understand how AI workflow orchestration works. I built a document processing pipeline that takes files, runs them through multiple steps with real AI analysis, and stores the results. It allowed me to learn how complex AI systems manage state and handle errors.

## What I Built

### **The Basic Flow**
1. **Ingest**: Read PDF/DOCX/TXT files and extract text
2. **Process**: Use AI (Ollama/OpenAI/Groq) to summarize and analyze content
3. **Store**: Save everything to SQLite database with chunks

### **What I Learned**
- **LangGraph Workflows**: How to chain AI operations with state management
- **Error Handling**: What happens when AI calls fail (and how to recover)
- **Document Processing**: Extracting text from different file formats
- **AI Integration**: Working with multiple AI providers (local and cloud)
- **State Management**: Passing data between workflow steps

### **What Actually Works**
- Real AI summaries using Llama 3.2 (or other models)
- Content safety analysis with AI reasoning
- Multi-step workflows that handle failures gracefully
- Database storage with proper relationships
- Support for PDF, DOCX, and TXT files

### **What's Still Basic**
- The "security scanning" is pretty simple keyword detection
- Key point extraction needs better prompt engineering
- No real authentication or access control
- Error handling could be more sophisticated

## How the Workflow ...Works

```
Document File → [Ingest] → [AI Analysis] → [Store] → Database
                    ↓           ↓            ↓
                 Extract     Summarize    Save with
                  Text      + Safety      Chunks
                            Check
                    
If anything fails → [Error Handler] → Log the problem
```

The cool part is how LangGraph manages the state between steps. Each step gets the results from the previous step and can decide what to do next based on success/failure.

## Running It

### **Easy Way**
```bash
# 1. Install packages
pip install -r requirements.txt

# 2. Make sure Ollama is running (for free AI)
ollama serve  # in another terminal
ollama pull llama3.2:1b  # download a small model

# 3. Run the demo
python demo.py
```

### **What You'll See**
The demo processes 3 sample documents and shows you:
- Real AI summaries of the content
- Safety analysis results
- How the workflow handles each step
- What gets stored in the database

### **Using Different AI Models**
Edit the `.env` file to switch AI providers:
```bash
# For free local AI (default)
AI_PROVIDER=ollama
OLLAMA_MODEL=llama3.2:1b

# For better quality (costs money)
AI_PROVIDER=openai
OPENAI_API_KEY=your_key_here

# For fast free cloud AI
AI_PROVIDER=groq
GROQ_API_KEY=your_key_here
```

## What I'm Still Learning

### **LangGraph Concepts**
- **Conditional Edges**: How to make the workflow branch based on results
- **State Design**: What data to pass between steps and how to structure it
- **Error Recovery**: Building workflows that handle failures gracefully
- **Complex Workflows**: When to use graphs vs simple function chains

### **AI Integration Challenges**
- **Prompt Engineering**: Getting consistent results from AI models
- **Rate Limiting**: Handling API limits and timeouts
- **Model Differences**: How different AI providers behave differently
- **Cost Management**: Balancing quality vs cost for AI calls

### **Real-World Considerations**
- **Scaling**: How to handle lots of documents without breaking
- **Security**: What actually makes a document processing system secure
- **Monitoring**: Knowing when things go wrong in production
- **Performance**: Making it fast enough for real use

## Code Example

```python
from document_pipeline import DocumentPipeline

# Initialize the pipeline
pipeline = DocumentPipeline()

# Process a document (this is async!)
result = await pipeline.process_document("my_document.pdf")

# Check what happened
if result["status"] == "completed":
    print(f"Processed: {result['document_id']}")
    
    # Get detailed results
    doc = pipeline.get_document_details(result['document_id'])
    print(f"Summary: {doc.summary}")
    print(f"Key Points: {doc.key_points}")
else:
    print(f"Failed: {result['error']}")
```

## Honest Assessment

### **What This Is Good For:**
✅ Learning LangGraph workflow concepts
✅ Understanding AI integration patterns  
✅ Experimenting with document processing
✅ Seeing real AI summarization in action

### **What This Is NOT:**
❌ Production-ready document processing system
❌ Enterprise-grade security solution  
❌ High-performance document analysis tool
❌ Comprehensive content management system


## Files You'll Get

When you run the demo:
- `sample_docs/` - Test documents (created automatically)
- `documents.db` - SQLite database with processed results
- `error_log.json` - Only created if errors occur (good sign if it's missing!)

