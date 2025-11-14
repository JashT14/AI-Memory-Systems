# AI-Memory-Systems
Basic demonstration between STM and LTM used in Agentic AI

---

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Concepts Explained](#concepts-explained)
- [Running the Code](#running-the-code)
- [Troubleshooting](#troubleshooting)
- [Advanced Usage](#advanced-usage)

---

## Overview

This project demonstrates two fundamental memory systems for AI agents:

1. **Short-Term Memory (STM)**: Maintains conversation context within a session using LangGraph's checkpoint system
2. **Long-Term Memory (LTM)**: Stores and retrieves information across sessions using FAISS vector database

### Use Cases

- **STM**: Chatbots that need to remember context during a conversation (e.g., "What's my name?")
- **LTM**: Knowledge bases, personal assistants, document retrieval systems

---

## Project Structure

```
project/
├── stm_agent.py          # Short-term memory implementation
├── ltm.py                # Long-term memory implementation
├── requirements.txt      # Python dependencies
├── .env                  # Environment variables (create this)
└── ltm_store/           # Vector database storage (auto-generated)
```

---

## Prerequisites

- **Python**: 3.9 or higher
- **HuggingFace Account**: Free account at [huggingface.co](https://huggingface.co)
- **API Token**: Required for accessing HuggingFace models
- **Operating System**: Windows, macOS, or Linux

---

## Installation

### Step 1: Clone or Create Project Directory

```bash
mkdir AI-Memory-Systems
cd AI-Memory-Systems
```

### Step 2: Create Virtual Environment (Recommended)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**Installation time**: 3-5 minutes depending on internet speed

### Step 4: Verify Installation

```bash
python -c "import langchain; import langgraph; import faiss; print('All packages installed successfully!')"
```

---

## Configuration

### 1. Get HuggingFace API Token

1. Go to [huggingface.co](https://huggingface.co) and sign up/login
2. Navigate to **Settings** → **Access Tokens**
3. Click **New Token** → Select **Read** permission → **Generate**
4. Copy the token (starts with `hf_...`)

### 2. Create `.env` File

Create a file named `.env` in the project root:

```bash
# .env
HUGGINGFACEHUB_API_TOKEN=hf_your_token_here
```

**Security Note**: Never commit `.env` to version control. Add it to `.gitignore`.

---

## Concepts Explained

### Short-Term Memory (STM)

**How it works:**
- Uses LangGraph's `InMemorySaver` checkpointer
- Stores conversation history tied to a `thread_id`
- Memory persists only during program execution
- Perfect for maintaining conversation context

**Key Components:**

1. **StateGraph**: Defines the conversation flow
   ```python
   graph_builder = StateGraph(ChatState)
   ```

2. **Checkpointer**: Stores conversation state
   ```python
   checkpointer = InMemorySaver()
   ```

3. **Thread ID**: Identifies unique conversation sessions
   ```python
   config = {"configurable": {"thread_id": "demo_session_001"}}
   ```

**Flow Diagram:**
```
User Input → Graph Node → LLM → Response → Checkpoint Storage
                ↑                              ↓
                └──────── Context Recall ──────┘
```

### Long-Term Memory (LTM)

**How it works:**
- Uses FAISS (Facebook AI Similarity Search) vector database
- Converts text to embeddings using `all-MiniLM-L6-v2` model
- Enables semantic search across stored information
- Persists data to disk for long-term storage

**Key Components:**

1. **Embeddings Model**: Converts text to numerical vectors
   ```python
   HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
   ```

2. **FAISS Vector Store**: Efficient similarity search
   ```python
   FAISS.from_texts(["text"], embeddings)
   ```

3. **Semantic Search**: Retrieves relevant information
   ```python
   store.similarity_search(query, k=2)
   ```

**Flow Diagram:**
```
Text Input → Embedding → Vector DB → Similarity Search → Relevant Memories
                            ↓
                     Disk Persistence
```

---

## Running the Code

### Running Short-Term Memory (STM)

```bash
python stm_agent.py
```

**Expected Output:**

```
new checkpointer created→
user message: Hi, my name is Jash Thakkar and I am a developer.
ai agent response: Hello Jash! Nice to meet you. How can I assist you today?

user message: What is my name?
ai agent response: Your name is Jash Thakkar.

NAME RECALLED
```

**What's happening:**
1. First message stores your name in conversation history
2. Second message retrieves it from the checkpoint
3. The script validates if the name was correctly recalled

### Running Long-Term Memory (LTM)

```bash
python ltm.py
```

**Expected Output:**

```
Memory storage using facts (example):

[LTM-STORE] Adding: Project Nexus launches in Q2 2026 with quantum encryption.
[LTM-STORE] Adding: Team lead is Dr. Sarah, specializing in cryptography & Advance AI Techniques.
[LTM-STORE] Adding: Budget approved: $2.7M for initial development phases
[LTM-STORE] Adding: Primary competitor is DataHSA COrp. with classical methods.

Questions:
[LTM-SEARCH] Query: Who is leading the project?
  Match 1: Team lead is Dr. Sarah, specializing in cryptography & Advance AI Techniques.
  Match 2: Memory initialized

[LTM-SEARCH] Query: What is the budget and timeline?
  Match 1: Budget approved: $2.7M for initial development phases
  Match 2: Project Nexus launches in Q2 2026 with quantum encryption.

[LTM-SEARCH] Query: Tell me about competitors
  Match 1: Primary competitor is DataHSA COrp. with classical methods.
  Match 2: Memory initialized
```

**What's happening:**
1. Facts are stored and converted to embeddings
2. Queries are embedded and matched against stored vectors
3. Most semantically similar memories are retrieved
4. Data is persisted in `ltm_store/` directory

---

## Troubleshooting

### Issue: "HUGGINGFACEHUB_API_TOKEN not found"

**Solution:**
```bash
# Verify .env file exists
cat .env  # Linux/macOS
type .env  # Windows

# Ensure token starts with 'hf_'
# Restart terminal after creating .env
```

### Issue: "ModuleNotFoundError"

**Solution:**
```bash
pip install -r requirements.txt --force-reinstall
```

### Issue: "Model loading timeout"

**Solution:**
```python
# In stm_agent.py, add timeout parameter:
llm=HuggingFaceEndpoint(
    repo_id="HuggingFaceH4/zephyr-7b-beta",
    task="text-generation",
    huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN,
    max_new_tokens=256,
    timeout=120  # Add this
)
```

### Issue: "FAISS dangerous deserialization"

**Explanation:** This warning is expected. The code uses `allow_dangerous_deserialization=True` because we control the data being loaded.

### Issue: Memory not persisting between runs (STM)

**Explanation:** STM uses `InMemorySaver()` which resets on each run. This is by design. For persistent memory:

```python
from langgraph.checkpoint.sqlite import SqliteSaver

# Replace InMemorySaver with:
checkpointer = SqliteSaver.from_conn_string("checkpoints.db")
```

---

## Advanced Usage

### Customizing STM Thread Management

```python
# Multiple conversation threads
def chat_with_thread(user_message, thread_id):
    config = {"configurable": {"thread_id": thread_id}}
    response = app.invoke(
        {"messages": [HumanMessage(content=user_message)]}, 
        config=config
    )
    return response['messages'][-1].content.strip()

# Different users/sessions
chat_with_thread("Hello", "user_123")
chat_with_thread("Hi there", "user_456")
```

### Customizing LTM Search Parameters

```python
# Retrieve more results
memories = ltm.retrieve_memory("query", k=5)

# Use different embedding model
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

# Add metadata filtering
ltm.store.add_texts(
    ["text"], 
    metadatas=[{"category": "finance", "date": "2025-01"}]
)
```

### Combining STM and LTM

```python
def hybrid_memory_agent(user_message, thread_id):
    # 1. Retrieve relevant long-term memories
    ltm_context = ltm.retrieve_memory(user_message, k=3)
    
    # 2. Add to conversation context
    enhanced_message = f"Context: {ltm_context}\n\nUser: {user_message}"
    
    # 3. Use STM for conversation flow
    config = {"configurable": {"thread_id": thread_id}}
    response = app.invoke(
        {"messages": [HumanMessage(content=enhanced_message)]},
        config=config
    )
    
    return response['messages'][-1].content
```

---

## Performance Considerations

### STM Performance
- **Memory Usage**: ~50-100MB per conversation thread
- **Response Time**: 2-5 seconds (depends on LLM)
- **Scalability**: Limited by RAM (InMemorySaver)

### LTM Performance
- **Index Building**: ~1-2 seconds per 1000 documents
- **Search Speed**: <100ms for 10k documents
- **Storage**: ~1KB per document + embeddings

---

## Security Best Practices

1. **Never commit `.env` files**
   ```bash
   echo ".env" >> .gitignore
   ```

2. **Use read-only tokens** for production

3. **Sanitize user inputs** before storing:
   ```python
   import re
   def sanitize(text):
       return re.sub(r'[^\w\s\-.,!?]', '', text)
   ```

4. **Encrypt sensitive data** in LTM:
   ```python
   from cryptography.fernet import Fernet
   # Implement encryption before storing
   ```

---

## Additional Resources

- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [LangChain Guide](https://python.langchain.com/docs/get_started/introduction)
- [FAISS Documentation](https://github.com/facebookresearch/faiss)
- [HuggingFace Models](https://huggingface.co/models)

---

## License

This code is provided as-is for educational purposes.

---

**Questions?** Open an issue or refer to the troubleshooting section above.
