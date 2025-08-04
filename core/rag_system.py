"""
Retrieval-Augmented Generation (RAG) System for Company Knowledge Base

This module implements a RAG system that processes company documents and provides
contextual responses to user feedback. It uses ChromaDB for vector storage and
LangChain for the RAG pipeline implementation.

Key features:
- Automatic document loading and processing (PDF and text files)
- Vector embedding storage with ChromaDB
- Contextual response generation using company knowledge
- Professional response formatting and tone
"""

import os
import sys
from typing import List, Dict, Any
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain.chains import RetrievalQA
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from models.model import load_model

class CompanyRAGSystem:
    """
    Company-specific Retrieval-Augmented Generation system.
    
    This class manages the complete RAG pipeline for processing company documents
    and generating contextual responses to user feedback based on company knowledge.
    
    Attributes:
        docs_dir (str): Directory path containing company documents
        vectorstore: ChromaDB vector database for document embeddings
        rag_pipeline: LangChain RetrievalQA pipeline for response generation
        embeddings: HuggingFace embeddings model for document processing
        llm: Language model for response generation
    """
    
    def __init__(self, docs_directory: str = 'data/company_docs'):
        """
        Initialize the CompanyRAGSystem with document processing and RAG setup.
        
        Sets up the document directory, embeddings model, language model,
        and ensures the document directory exists before initializing the RAG pipeline.
        
        Args:
            docs_directory (str): Path to directory containing company documents
        """
        self.docs_dir = docs_directory
        self.vectorstore = None
        self.rag_pipeline = None
        
        # Initialize embeddings model for document processing
        self.embeddings = HuggingFaceEmbeddings(
            model_name='sentence-transformers/paraphrase-MiniLM-L6-v2'
        )
        
        # Load the language model for response generation
        self.llm = load_model()
        
        # Ensure document directory exists and set up RAG system
        self.ensure_docs_directory()
        self.setup_rag()
        
    def ensure_docs_directory(self) -> None:
        """
        Ensure the document directory exists and is accessible.
        
        Creates the document directory if it doesn't exist and handles
        permissions issues by falling back to a temporary directory.
        """
        try:
            # Get absolute path to documents directory
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(current_dir)
            self.docs_dir = os.path.join(project_root, self.docs_dir)
            
            # Create directory if it doesn't exist
            os.makedirs(self.docs_dir, exist_ok=True)
            
            # Verify directory is accessible
            if not os.path.exists(self.docs_dir):
                raise OSError(f"Directory inaccessible: {self.docs_dir}")
                
            print(f"Document directory configured: {self.docs_dir}")
            
        except Exception as e:
            print(f"Error setting up document directory: {e}")
            # Fallback to temporary directory
            import tempfile
            self.docs_dir = tempfile.mkdtemp(prefix="company_docs_")
            print(f"Using temporary directory: {self.docs_dir}")
    
    def load_documents(self) -> List[Any]:
        """
        Load all valid documents from the document directory.
        
        Processes PDF and text files using appropriate loaders and handles
        errors gracefully to avoid interrupting the loading process.
        
        Returns:
            List[Any]: List of loaded document objects
        """
        documents = []
        valid_extensions = ('.pdf', '.txt')
        
        # Process each file in the documents directory
        for file in os.listdir(self.docs_dir):
            file_path = os.path.join(self.docs_dir, file)
            
            try:
                if file.endswith('.pdf'):
                    loader = PyPDFLoader(file_path)
                    documents.extend(loader.load())
                    print(f"Loaded PDF: {file}")
                elif file.endswith('.txt'):
                    loader = TextLoader(file_path, encoding='utf-8')
                    documents.extend(loader.load())
                    print(f"Loaded text file: {file}")
            except Exception as e:
                print(f"Error loading {file}: {e}")
                continue
        
        print(f"Total documents loaded: {len(documents)}")
        return documents
    
    def setup_rag(self) -> None:
        """
        Set up the complete Retrieval-Augmented Generation pipeline.
        
        This method orchestrates the entire RAG setup process:
        1. Load and process documents
        2. Split documents into chunks for better retrieval
        3. Create vector embeddings and store in ChromaDB
        4. Configure prompt template for response generation
        5. Initialize the RetrievalQA pipeline
        
        Raises:
            ValueError: If no valid documents are found to process
        """
        # Load company documents
        documents = self.load_documents()
        if not documents:
            raise ValueError("No valid documents found in the specified directory")
        
        # Split documents into smaller chunks for better processing
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,    # Maximum characters per chunk
            chunk_overlap=200   # Overlap between chunks to maintain context
        )
        chunks = text_splitter.split_documents(documents)
        print(f"Document chunks created: {len(chunks)}")
        
        # Create vector database for document embeddings
        self.vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
        # Create vector database for document embeddings
        self.vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory='data/company_chroma_db'
        )
        
        # Configure prompt template for professional response generation
        prompt_template = """
        # ROLE: You are a professional customer service assistant representing our company.
        
        # CONTEXT: Use the following company information to provide accurate responses:
        {context}

        # USER FEEDBACK:
        {question}

        # INSTRUCTIONS:
        - Respond professionally representing "our company" 
        - Address the user directly as "you"
        - Keep responses concise (5-10 sentences)
        - Maintain a friendly but professional tone
        - If the context doesn't contain relevant information, acknowledge this politely
        - Focus on being helpful and solution-oriented
        
        # RESPONSE:
        """
        
        # Create prompt template object
        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        
        # Initialize the RetrievalQA pipeline with configured components
        self.rag_pipeline = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",  # Concatenate all retrieved documents
            retriever=self.vectorstore.as_retriever(
                search_kwargs={"k": 3}  # Retrieve top 3 most relevant documents
            ),
            chain_type_kwargs={
                "prompt": prompt,
                "document_variable_name": "context"
            },
            input_key="question",
            output_key="result",
            return_source_documents=False  # Don't return source docs for cleaner output
        )
        print("RAG system successfully initialized and ready for queries")

    def query(self, question: str) -> Dict[str, Any]:
        """
        Execute a query against the RAG system to generate contextual responses.
        
        Takes user feedback as input and generates a response based on company
        knowledge base and the configured prompt template.
        
        Args:
            question (str): User feedback or question to process
            
        Returns:
            Dict[str, Any]: Dictionary containing the query result with keys:
                - 'question': The original question/feedback
                - 'result': The generated response
                - 'error': Error message if processing fails
        """
        try:
            print(f"Processing query: {question[:100]}...")
            result = self.rag_pipeline({"question": question})
            print("Query processed successfully")
            return result
        except Exception as e:
            error_msg = f"Error during RAG query processing: {e}"
            print(error_msg)
            return {
                "question": question,
                "result": "I apologize, but I'm experiencing technical difficulties. Please try again later or contact support.",
                "error": str(e)
            }
    
    def add_document(self, file_path: str) -> bool:
        """
        Add a new document to the RAG system knowledge base.
        
        Processes a new document and adds it to the existing vector store.
        This allows for dynamic knowledge base updates without full reinitialization.
        
        Args:
            file_path (str): Path to the document file to add
            
        Returns:
            bool: True if document was successfully added, False otherwise
        """
        try:
            # Load the new document
            if file_path.endswith('.pdf'):
                loader = PyPDFLoader(file_path)
            elif file_path.endswith('.txt'):
                loader = TextLoader(file_path, encoding='utf-8')
            else:
                print(f"Unsupported file type: {file_path}")
                return False
            
            documents = loader.load()
            
            # Split into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            chunks = text_splitter.split_documents(documents)
            
            # Add to existing vector store
            self.vectorstore.add_documents(chunks)
            print(f"Successfully added document: {os.path.basename(file_path)}")
            return True
            
        except Exception as e:
            print(f"Error adding document {file_path}: {e}")
            return False
    
    def get_system_info(self) -> Dict[str, Any]:
        """
        Get information about the RAG system configuration and status.
        
        Returns:
            Dict[str, Any]: System information including document count and configuration
        """
        return {
            "docs_directory": self.docs_dir,
            "vector_store_initialized": self.vectorstore is not None,
            "rag_pipeline_initialized": self.rag_pipeline is not None,
            "embedding_model": "sentence-transformers/paraphrase-MiniLM-L6-v2",
            "chunk_size": 1000,
            "chunk_overlap": 200,
            "retrieval_k": 3
        }

if __name__ == "__main__":
    """
    Example usage and testing of the CompanyRAGSystem.
    
    This section demonstrates how to initialize and use the RAG system
    for processing user feedback queries.
    """
    try:
        # Initialize the RAG system
        print("Initializing CompanyRAGSystem...")
        rag_system = CompanyRAGSystem()
        
        # Display system information
        system_info = rag_system.get_system_info()
        print(f"System Info: {system_info}")
        
        # Test query
        test_question = "How can I improve the quality of your mobile products?"
        print(f"\nProcessing test query: {test_question}")
        
        results = rag_system.query(test_question)
        user_input = results.get('question', test_question)
        rag_response = results.get('result', 'No response generated.')
        
        print(f"\n===> User Input: {user_input}")
        print(f"===> RAG Response: {rag_response}")
        
        # Check for errors
        if 'error' in results:
            print(f"===> Error: {results['error']}")
            
    except Exception as e:
        print(f"Error during RAG system testing: {e}")
