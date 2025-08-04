"""
FastAPI Application for User Feedback Analysis System

This module provides a REST API for processing user feedback through intent classification
and RAG-based response generation. It serves as the main entry point for the feedback
analysis system.
"""

import os
import sys

# Add project root to Python path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, Body, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
import logging

from core.logic import MeaningEngine
from core.rag_system import CompanyRAGSystem

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI application
app = FastAPI(
    title="User Feedback Analysis API",
    description="Intelligent feedback processing with intent classification and RAG responses",
    version="1.0.0"
)

# Initialize core components
engine = MeaningEngine()
rag_system = CompanyRAGSystem()

class FeedbackRequest(BaseModel):
    """
    Request model for feedback processing endpoint.
    
    Attributes:
        feedback (str): The user feedback text to analyze
        admin_email (str): Email address for admin notifications
    """
    feedback: str = Field(..., description="User feedback text to process", min_length=1)
    admin_email: str = Field(..., description="Admin email for notifications", regex=r'^[^@]+@[^@]+\.[^@]+$')

class FeedbackResponse(BaseModel):
    """
    Response model for feedback processing results.
    
    Attributes:
        results (dict): RAG system query results
        intent_label (str): Classified intent label
        user_input (str): Original user input
        rag_response (str): Generated RAG response
    """
    results: dict
    intent_label: str
    user_input: str
    rag_response: str

@app.post("/process-feedback", response_model=FeedbackResponse)
def process_feedback_api(request: FeedbackRequest) -> FeedbackResponse:
    """
    Process user feedback through intent classification and RAG system.
    
    This endpoint analyzes user feedback to:
    1. Classify the intent using the MeaningEngine
    2. Generate contextual responses using the RAG system
    3. Handle emerging intents and send alerts when necessary
    
    Args:
        request (FeedbackRequest): The feedback processing request
        
    Returns:
        FeedbackResponse: Processed feedback with intent and RAG response
        
    Raises:
        HTTPException: If processing fails or invalid input provided
    """
    try:
        feedback = request.feedback
        admin_email = request.admin_email
        
        logger.info(f"Processing feedback: {feedback[:50]}...")
        
        # Classify intent using the meaning engine
        intent_tuple = engine.predict_intent(
            feedback, 
            top_k=3, 
            admin_email=admin_email
        )
        
        # Extract intent label from classification result
        if intent_tuple[0] == "EmergingIntent":
            intent_label = intent_tuple[1]
        else:
            intent_label = intent_tuple[0]
            
        logger.info(f"Classified intent: {intent_label}")
        
        # Generate RAG response
        try:
            results = rag_system.query(feedback)
            user_input = results.get('question', feedback)
            rag_response = results.get('result', 'No response generated.')
            logger.info("RAG response generated successfully")
            
        except Exception as e:
            logger.error(f"RAG system error: {str(e)}")
            results = {"error": str(e)}
            rag_response = "I couldn't process your request at the moment. Please try again later."
            user_input = feedback
        
        return FeedbackResponse(
            results=results,
            intent_label=intent_label,
            user_input=user_input,
            rag_response=rag_response
        )
        
    except Exception as e:
        logger.error(f"Feedback processing error: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to process feedback: {str(e)}"
        )

@app.get("/")
def read_root():
    """
    Root endpoint providing API information.
    
    Returns:
        dict: Basic API information and status
    """
    return {
        "message": "User Feedback Analysis API",
        "version": "1.0.0",
        "status": "active",
        "docs": "/docs",
        "redoc": "/redoc"
    }

@app.get("/health")
def health_check():
    """
    Health check endpoint for monitoring system status.
    
    Returns:
        dict: System health status and component availability
    """
    try:
        # Test core components
        engine_status = "healthy" if engine else "unavailable"
        rag_status = "healthy" if rag_system else "unavailable"
        
        return {
            "status": "healthy",
            "components": {
                "meaning_engine": engine_status,
                "rag_system": rag_status
            },
            "timestamp": __import__("datetime").datetime.now().isoformat()
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": __import__("datetime").datetime.now().isoformat()
        }

if __name__ == "__main__":
    import uvicorn
    
    # Run the application
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )