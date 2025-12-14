import os
import shutil
import time
import chromadb
from typing import List, Dict, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

from paths import VECTOR_DB_DIR


class VectorDB:
    """
    A simple vector database wrapper using ChromaDB with HuggingFace embeddings.
    """

    def __init__(self, collection_name: str = None, embedding_model: str = None):
        """
        Initialize the vector database.

        Args:
            collection_name: Name of the ChromaDB collection
            embedding_model: HuggingFace model name for embeddings
        """
        self.collection_name = collection_name or os.getenv(
            "CHROMA_COLLECTION_NAME", "rag_documents"
        )
        self.embedding_model_name = embedding_model or os.getenv(
            "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        )

        # Delete Database Directory(if exists) as we are always ingesting the documents
        db_dir = VECTOR_DB_DIR
        if os.path.exists(db_dir):
            shutil.rmtree(db_dir)

        os.makedirs(db_dir, exist_ok=True)

        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(path=db_dir)

        # Load embedding model
        print(f"Loading embedding model: {self.embedding_model_name}")
        self.embedding_model = SentenceTransformer(self.embedding_model_name)

        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"description": "RAG document collection"},
        )

        print(f"Vector database initialized with collection: {self.collection_name}")

    def chunk_text(self, text: str, chunk_size: int = 500) -> List[str]:
        """
        Implement text chunking logic with LangChain's RecursiveCharacterTextSplitter.
        Automatically handles sentence boundaries and preserves context better
        Considering chunk_overlap = around 10% of chunk_size

        Args:
            text: Input text to chunk
            chunk_size: Approximate number of characters per chunk

        Returns:
            List of text chunks
        """

        chunks = []
        if text:
            text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=int(chunk_size * 0.10),  # 10% chunk size for Overlapping
            )
            chunks = text_splitter.split_text(text)
        return chunks

    def add_documents(self, documents: List) -> None:
        """
        Document ingestion logic: Add documents to the vector database.

        Args:
            documents: List of documents
        """

        print(f"Processing {len(documents)} documents...")
        processing_timestamp = int(time.time())
        
        # Extract 'content' and 'metadata' from each document dict
        for doc_index, doc in enumerate(documents):
            content = doc.get('content')
            metadata = doc.get('metadata')

            if content:
                # split each document into chunks
                chunks = self.chunk_text(content)

                # create embeddings for all chunks
                embeddings = self.embedding_model.encode(chunks)
                
                # print message to know how many chunks are generated for each document
                print(f"Processing {doc_index+1}. {metadata.get('name','')} total chunks: {len(chunks)}")

                ids = []
                metadatas = []
                for index, chunk in enumerate(chunks):
                    # Create unique IDs for each chunk (e.g., "147896523_doc_0_chunk_0")
                    id = f"{processing_timestamp}_doc_{doc_index}_chunk_{index}"    
                    
                    # Extend meta information with id and chunk_size
                    meta = metadata | {'id': id, 'chunk_size' : len(chunk)}         
                    ids.append(id)
                    metadatas.append(meta)     
                
                # Store the embeddings, documents, metadata, and IDs in your vector database
                self.collection.add(
                    embeddings=embeddings,
                    ids=ids,
                    documents=chunks,
                    metadatas=metadatas
                )    

        print("Documents added to vector database")

    def search(self, query: str, n_results: int = 5) -> Dict[str, Any]:
        """
        Search for similar documents in the vector database.

        Args:
            query: Search query
            n_results: Number of results to return

        Returns:
            Dictionary containing search results with keys: 'documents', 'metadatas', 'distances', 'ids'
        """

        print("Embedding query...")
        embedded_query = self.embedding_model.encode([query])[0]

        # Query the collection
        results = self.collection.query(
            query_embeddings=[embedded_query],
            n_results=n_results,
            include=["documents", "distances", "metadatas"],
        )

        relevant_results = {
            "ids": results.get("ids", [[]])[0],
            "documents": results.get("documents", [[]])[0],
            "metadatas": results.get("metadatas", [[]])[0],
            "distances": results.get("distances", [[]])[0],
        }

        return relevant_results


 
