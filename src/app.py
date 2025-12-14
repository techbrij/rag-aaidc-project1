import os
from typing import Dict, List
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from vectordb import VectorDB
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from paths import DATA_DIR

# Load environment variables
load_dotenv()


def load_documents() -> List[Dict]:
    """
    Read all .txt files from data directory

    Returns:
         [{"content": str, "metadata": {"name": filename, "last_modified_timestamp" : timestamp }}]
    """
    results = []

    for file_name in os.listdir(DATA_DIR):
        if file_name.endswith(".txt"):
            # Read and get the file content
            try:
                file_path = os.path.join(DATA_DIR, file_name)
                with open(file_path, "r", encoding="utf-8") as file:
                    content = file.read()
                    if content:
                        results.append({'content': content, 'metadata': {'name': file_name, 'last_modified_timestamp': os.path.getmtime(file_path)}})
            except IOError as e:
                print(f"Error reading file {file_name}: {e}")            
    return results


class RAGAssistant:
    """
    A simple RAG-based AI assistant using ChromaDB and multiple LLM providers.
    Supports OpenAI, Groq, and Google Gemini APIs.
    """

    def __init__(self):
        """Initialize the RAG assistant."""
        # Initialize LLM - check for available API keys in order of preference
        self.llm = self._initialize_llm()
        if not self.llm:
            raise ValueError(
                "No valid API key found. Please set one of: "
                "OPENAI_API_KEY, GROQ_API_KEY, or GOOGLE_API_KEY in your .env file"
            )

        # Initialize vector database
        self.vector_db = VectorDB()       

        # RAG prompt template to effectively use retrieved context to answer questions
        template_str = """
        You are a knowledgeable assistant. Answer the user's question **ONLY** using
        the information provided in the context below.

        ### Rules:

        - Use clear, concise language with bullet points where appropriate.
        - Given the some documents that should be relevant to the user's question, answer the user's question.

        ### Output Constraint

        - Only answer questions based on the provided documents.
        - If the user's question is not related to the documents, then you SHOULD NOT answer the question. Say "The question is not answerable from the provided documents".
        - Never answer a question from your own knowledge.
        - Provide answers in markdown format.
        - Provide concise answers in bullet points when relevant.

        ### Context:
        {context}

        ### Question:
        {question}

        """

        self.prompt_template = ChatPromptTemplate.from_template(template_str)

        # Create the chain
        self.chain = self.prompt_template | self.llm | StrOutputParser()

        print("RAG Assistant initialized successfully")

    def _initialize_llm(self):
        """
        Initialize the LLM by checking for available API keys.
        Tries OpenAI, Groq, and Google Gemini in that order.
        """
        # Check for OpenAI API key
        if os.getenv("OPENAI_API_KEY"):
            model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
            print(f"Using OpenAI model: {model_name}")
            return ChatOpenAI(
                api_key=os.getenv("OPENAI_API_KEY"), model=model_name, temperature=0.0
            )

        elif os.getenv("GROQ_API_KEY"):
            model_name = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
            print(f"Using Groq model: {model_name}")
            return ChatGroq(
                api_key=os.getenv("GROQ_API_KEY"), model=model_name, temperature=0.0
            )

        elif os.getenv("GOOGLE_API_KEY"):
            model_name = os.getenv("GOOGLE_MODEL", "gemini-2.0-flash")
            print(f"Using Google Gemini model: {model_name}")
            return ChatGoogleGenerativeAI(
                google_api_key=os.getenv("GOOGLE_API_KEY"),
                model=model_name,
                temperature=0.0,
            )

        else:
            raise ValueError(
                "No valid API key found. Please set one of: OPENAI_API_KEY, GROQ_API_KEY, or GOOGLE_API_KEY in your .env file"
            )

    def add_documents(self, documents: List) -> None:
        """
        Add documents to the knowledge base.

        Args:
            documents: List of documents
        """
        self.vector_db.add_documents(documents)

    def invoke(self, input: str, n_results: int = 3) -> str:
        """
        Query the RAG assistant.

        Args:
            input: User's input
            n_results: Number of relevant chunks to retrieve

        Returns:
            String containing the answer
        """
        llm_answer = ""

        # Implement the RAG query pipeline

        # Retrieve relevant context chunks
        results = self.vector_db.search(input, n_results)
        # documents, metadatas, distances = results["documents"], results["metadatas"], results["distances"]
        relevant_documents = results["documents"]

        # Print the relevant documents
        print("-" * 100)
        print("Relevant documents: \n")
        for doc in relevant_documents:
            print(doc)
            print("-" * 100)
        print("")     

        # Build the context and generate the response
        context = "\n\n---\n\n".join(relevant_documents)
        llm_answer = self.chain.invoke({"context": context, "question": input})               
        return llm_answer


def main():
    """Main function to demonstrate the RAG assistant."""
    try:
        # Initialize the RAG assistant
        print("Initializing RAG Assistant...")
        assistant = RAGAssistant()

        # Load sample documents
        print("\nLoading documents...")
        sample_docs = load_documents()
        print(f"Loaded {len(sample_docs)} sample documents")

        assistant.add_documents(sample_docs)

        done = False

        while not done:
            question = input("Enter a question or 'quit' to exit: ")
            if question.lower() == "quit":
                done = True
            else:
                result = assistant.invoke(question)
                print("Response:\n\n" + result + "\n" + ("-" * 100) )

    except Exception as e:
        print(f"Error running RAG assistant: {e}")
        print("Make sure you have set up your .env file with at least one API key:")
        print("- OPENAI_API_KEY (OpenAI GPT models)")
        print("- GROQ_API_KEY (Groq Llama models)")
        print("- GOOGLE_API_KEY (Google Gemini models)")


if __name__ == "__main__":
    if not os.path.exists(DATA_DIR):
        raise FileNotFoundError(f"Data directory not found: {DATA_DIR}")
    
    main()
