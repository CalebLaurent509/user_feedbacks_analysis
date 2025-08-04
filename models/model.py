# models/model.py - Version corrigée
"""
LLM Model Configuration and Initialization Module
"""

import os
import logging
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from sentence_transformers import SentenceTransformer,  util
from transformers import pipeline

# Load environment variables
load_dotenv()

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Model configuration constants
MODEL_NAME = "gpt-3.5-turbo"
TEMPERATURE = 0.1
# Initialize the embeddings and intent classifier
embeddings = SentenceTransformer("paraphrase-MiniLM-L6-v2")
intent_classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

def load_model() -> ChatOpenAI:
    """Initialize and cache the ChatOpenAI language model using singleton pattern."""
    if not hasattr(load_model, "llm"):
        logger.info(f"Initializing ChatOpenAI model ({MODEL_NAME})...")
        
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            logger.error("OPENAI_API_KEY not found in environment variables")
            raise ValueError(
                "OPENAI_API_KEY not found in environment variables. "
                "Please ensure it's set in your .env file or system environment."
            )
        
        try:
            llm = ChatOpenAI(
                openai_api_key=api_key,
                model_name=MODEL_NAME,
                temperature=TEMPERATURE
            )
            
            load_model.llm = llm
            logger.info(f"ChatOpenAI {MODEL_NAME} model loaded successfully!")
            logger.info(f"Temperature: {TEMPERATURE}")
            
        except Exception as e:
            logger.error(f"Failed to initialize ChatOpenAI model: {e}")
            raise
    else:
        logger.debug(f"Using cached {MODEL_NAME} model instance")
    
    return load_model.llm

def get_model_info() -> dict:
    """Retrieve current model configuration information."""
    return {
        "model_name": MODEL_NAME,
        "temperature": TEMPERATURE,
        "is_loaded": hasattr(load_model, "llm"),
        "api_key_configured": bool(os.getenv("OPENAI_API_KEY"))
    }

def reset_model_cache() -> None:
    """Clear the cached model instance, forcing reinitialization on next load."""
    if hasattr(load_model, "llm"):
        delattr(load_model, "llm")
        logger.info("Model cache cleared successfully")
    else:
        logger.info("No cached model to clear")

# Test function
if __name__ == "__main__":
    print("<==Testing LLM Model Module ==>")
    try:
        model_info = get_model_info()
        print(f"Model Info: {model_info}")
        
        if model_info["api_key_configured"]:
            llm = load_model()
            print("Model loaded successfully")
            test_prompt = "Hello! Please introduce yourself as an AI shopping assistant."
            response = llm.invoke(test_prompt)
            print(f"====> Test Response: {response.content}")
        else:
            print("OPENAI_API_KEY not configured - skipping model tests")
            
    except Exception as e:
        print(f"Error during testing: {e}")