# RAG-Based AI Assistant - AAIDC Project 1

## 🤖 Overview

This is a **complete RAG (Retrieval-Augmented Generation) AI assistant** that enables intelligent question-answering over custom document collections. RAG systems combine document search with AI chat - they can answer questions about your specific documents by finding relevant information and using it to generate responses.

**Think of it as:** ChatGPT that knows about YOUR documents and can answer questions about them.

## 🎯 What this system does

This AI assistant can:

- 📄 **Load your documents** (text files, etc.)
- 🔍 **Search through them** to find relevant information
- 💬 **Answer questions** using the information it found
- 🧠 **Combine multiple sources** to give comprehensive answers

## 🚀 Setup Instructions

### Prerequisites

Before starting, make sure you have:

- Python 3.8 or higher installed
- An API key from **one** of these providers:
  - [OpenAI](https://platform.openai.com/api-keys) (most popular)
  - [Groq](https://console.groq.com/keys) (free tier available)
  - [Google AI](https://aistudio.google.com/app/apikey) (competitive pricing)

**Important:** This project uses specific packages:

- LangChain
- Vector DB: ChromaDB
- Embedding Model: sentence-transformers/all-MiniLM-L6-v2
- LLM: gemini-2.5-flash (I used but can be used anyone as mentioned earlier)

### Quick Setup

1. **Clone and install dependencies:**

   ```bash
   git clone [your-repo-url]
   cd [project folder]
   python3 -m venv venv
   .\venv\Scripts\Activate.ps1   # Activate Virtual environment (Windows powershell)
   pip install -r requirements.txt
   ```

2. **Configure your API key:**

   ```bash
   # Create environment file (choose the method that works on your system)
   cp .env.example .env    # Linux/Mac
   copy .env.example .env  # Windows
   ```

   Edit `.env` and add your API key:

   ```
   OPENAI_API_KEY=your_key_here
   # OR
   GROQ_API_KEY=your_key_here
   # OR
   GOOGLE_API_KEY=your_key_here
   ```



3. **Run the application:**

    ```bash
    python src/app.py
    ```
## 💬 Usage

The system automatically processes documents in the `data/` directory and provides an interactive interface for asking questions.

### Example Queries

Try these example questions:

- "What is [topic from your documents]?"
- "Explain [concept from your documents]"
- "How does [process from your documents] work?"



## 📝 Implementation Steps

The project has 7 main steps:

1. **Prepare Your Documents** - Add your own documents to the data directory
2. **Document Loading** - Load documents from files into the system
3. **Text Chunking** - Split documents into smaller, searchable chunks
4. **Document Ingestion** - Process and store documents in the vector database  
5. **Similarity Search** - Find relevant documents based on queries
6. **RAG Prompt Template** - Design effective prompts for the LLM
7. **RAG Query Pipeline** - Complete query-response pipeline using retrieved context

---

### Step 1: Prepare Your Documents

**Replace the sample documents with your own content**

The `data/` directory contains sample files on various topics. Replace these with documents relevant to your domain:

```
data/
├── your_topic_1.txt
├── your_topic_2.txt
└── your_topic_3.txt
```

Each file should contain text content you want your RAG system to search through.

---

### Step 2: Implement Document Loading

**Location:** `src/app.py`
**Function** `load_documents`

- Read files from the `data/` directory
- Load the content of each file into memory
- Return a list of document dictionaries with content and metadata
- Currently the implementation handles the text type of files
---

### Step 3: Implement Text Chunking

**Location:** `src/vectordb.py`
**Function** `chunk_text`

- Choose a chunking strategy (LangChain's `RecursiveCharacterTextSplitter`)
- Split the input text into manageable chunks
- Return a list of text strings
---

### Step 4: Implement Document Ingestion

**Location:** `src/vectordb.py`
**Function** `add_documents`


- Loop through the documents list
- Extract content and metadata from each document
- Use your `chunk_text()` method to split documents
- Create embeddings using `self.embedding_model.encode()`
- Store everything in ChromaDB using `self.collection.add()`

---

### Step 5: Implement Similarity Search

**Location:** `src/vectordb.py`
**Function** `search`

- Create an embedding for the query using `self.embedding_model.encode()`
- Search the ChromaDB collection using `self.collection.query()`
- Return results in the expected format with keys: `documents`, `metadatas`, `distances`, `ids`

---

### Step 6: Implement RAG Prompt Template

**Location:** `src/app.py`

- Design a prompt template that effectively combines retrieved context with user questions
- Use `ChatPromptTemplate.from_template()` to create the template
- Include placeholders for `{context}` (retrieved documents) and `{question}` (user query)
- Consider how to instruct the LLM to use the context appropriately
- Handle cases where the context might not contain relevant information
---

### Step 7: Implement RAG Query Pipeline

**Location:** `src/app.py`
**Function** `query`

- Use `self.vector_db.search()` to find relevant context
- Combine retrieved chunks into a context string
- Use `self.chain.invoke()` to generate a response
- Return structured results

---
## 📁 Project Structure

```
rt-aaidc-project1-template/
├── src/
│   ├── app.py           # Main RAG application (implement Steps 2, 6-7)
│   └── vectordb.py      # Vector database wrapper (implement Steps 3-5)
├── data/               # Replace with your documents (Step 1)
│   ├── *.txt          # Your text files here
├── requirements.txt    # All dependencies included
├── .env.example       # Environment template
└── README.md          # This guide
```

---

## 📄 License
This project is licensed under the MIT License.
See the [LICENSE] file for details

## 👤 Author
Brij Mohan
- GitHub: https://github.com/techbrij/rag-ai-assistant-langchain-aaidc-project-1
