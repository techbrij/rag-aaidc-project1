# RAG-Based AI Assistant - AAIDC Project 1

## 🤖 Overview

This is a **RAG (Retrieval-Augmented Generation) AI assistant** that enables intelligent question-answering over custom document collections. RAG systems combine document search with AI chat - they can answer questions about your specific documents by finding relevant information and using it to generate responses.

**Think of it as:** ChatGPT that knows about YOUR documents and can answer questions about them.

This project implements a Retrieval-Augmented Generation (RAG) Assistant that combines a vector database (ChromaDB) with multiple Large Language Models (LLMs), supports OpenAI GPT, Groq LLaMA, and Google Gemini. The system enables users to ask questions over their own local text documents and receive accurate, context-aware responses.

## 📝 Project Features & Steps

The project has 7 main steps:

1. **Prepare Your Documents** - Add your own documents to the data directory in text format
2. **Document Loading** - loads `.txt` files from data directory
3. **Text Chunking** - Split documents into smaller, searchable chunks
4. **Document Ingestion** - Process, embedding the chunks and store the information in the vector database  
5. **Similarity Search** - On user query, the question is embedded and get relevant documents from vector search
6. **RAG Prompt Template** - Design effective prompts for the LLM
7. **RAG Query Pipeline** - Complete query-response pipeline using retrieved context

---

## Target Audience

AI/ML Practitioners, Engineering team and anyone who want to implement or use RAG based AI Assistant.


## Technical Stack 

| Component                                    | Purpose                |
| -------------------------------------------- | ---------------------- |
| **Python 3.10+**                              | Core language          |
| **LangChain**                                | RAG orchestration      |
| **ChromaDB**                                 | Vector store           |
| **SentenceTransformers (all-MiniLM-L6-v2)**  | Embedding & Generation |
| **LangChain RecursiveCharacterTextSplitter** | Chunking documents     |
| **OpenAI GPT, Groq Llama, Google Gemini**    | Multi-model support    |
| **Text files**                               | Document source        |


## 📁 Project Structure

![Project Structure](https://github.com/techbrij/rag-aaidc-project1/blob/main/images/rt-rag-project-structure.png?raw=true)

---
## 🚀 Setup Instructions

### Prerequisites

Before starting, make sure you have:

- Python 3.11 or higher installed
- An API key from **one** of these providers:
  - [OpenAI](https://platform.openai.com/api-keys) (most popular)
  - [Groq](https://console.groq.com/keys) (free tier available)
  - [Google AI](https://aistudio.google.com/app/apikey) (competitive pricing)

**Important:** This project uses specific packages:

- LangChain
- Vector DB: ChromaDB
- Embedding Model: sentence-transformers/all-MiniLM-L6-v2
- LLM: gemini-2.5-flash (I used but can be used anyone as mentioned earlier)

### Setup & Installation

1. **Clone and install dependencies:**

   - Clone the repository
   ```bash
   git clone https://github.com/techbrij/rag-aaidc-project1
   cd rag-aaidc-project1
   ```
   - Create Virtual Environment
   ```
   python3 -m venv venv
   ```
   - Activate the virtual environment:

      On Windows:
      ```
      venv\Scripts\activate
      ```
      On macOS/Linux:
      ```
      source venv/bin/activate
      ```
   - Install dependencies
   ```
   pip install -r requirements.txt
   ```

2. **Configure Environment Variables:**

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
   Add any API key and comment the other API key parameters to avoid the conflicts. At a time, only one key should be active.


3. **Run the application:**

    ```bash
    python src/app.py
    ```
## 💬 Usage

The system automatically processes documents in the `data/` directory and provides an interactive interface for asking questions.

### Example 

Try the following question:

- "What is AI responsible development?"


### Output Screenshot

![RAG AI assistant](https://github.com/techbrij/rag-aaidc-project1/blob/main/images/rt-rag-assistant.png?raw=true)


## Testing
The project uses pytest for performance testing. Tests can be run using the following command:

```bash
pytest tests/test_performance.py -s
```

## Implementation
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

## Evaluation and Analysis

### Evaluation Methodology

The evaluation focused on three core components:

- **Document Loading:** Measuring the time required to load all source documents from disk.
- **Document Ingestion:** Measuring the time to add documents to the vector database.
- **Assistant Response:** Measuring the response time for the assistant to answer a set of representative user queries.

For each component, we executed the relevant function multiple times and recorded the elapsed time using Python’s time module. For the assistant’s response, we used a set of diverse questions and averaged the response times over several runs to ensure reliability and account for variability due to external API calls.

run following command to run the test
```
pytest tests/test_performance.py -s
```

### Performance Metrics
- Load Time: Time (in seconds) to read and parse all documents from the data directory.
- Ingestion Time: Time (in seconds) to add all documents to the vector database.
- Response Time: Average time (in seconds) for the assistant to generate an answer to a user query, measured over multiple runs and multiple questions.
- Correctness: Each response was checked to ensure it was a non-empty string, indicating successful processing.

### Test Lab Setup

- Infrastructure: Windows 11, 32 GB RAM, 2TB SSD, i7-13650HX
- Environment: Python 3.13.7, gemini-2.5-flash, LangChain 0.3.27
- Total Documents: 7
- Total Chunks: 35

Use different types of questions for different test cases:

- Q1: What is artificial intelligence? (Simple)
- Q2: What are Machine Learning and MLOps? (Multi-documents question)
- Q3: Who is home minister of India? (Out of context)

## Contributing
If you want to contribute to improve the framework, please refer to the guidelines below for how you can contribute:

- Fork the repository.
- Create a new branch for your changes.
- Submit a pull request with a detailed description of your modifications.

## License
This project is licensed under the MIT License. 
See the [LICENSE] file for details

## Contact
Brij Mohan
- GitHub: https://github.com/techbrij
