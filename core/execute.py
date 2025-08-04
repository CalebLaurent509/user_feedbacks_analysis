import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.logic import MeaningEngine
from rag_system import CompanyRAGSystem

# Initialisation des systèmes
engine = MeaningEngine()
rag_system = CompanyRAGSystem()

def process_feedback(feedback, admin_email):
    # On récupère le tuple complet de predict_intent (label, simplified, score) ou (label, score)
    intent_tuple = engine.predict_intent(feedback, top_k=3, admin_email=admin_email)
    # Pour l'affichage, on prend le label et, si Emerging, la simplification
    if intent_tuple[0] == "EmergingIntent":
        intent_label = intent_tuple[1]
    else:
        intent_label = intent_tuple[0]
    # On génère une réponse avec le rag_system
    print(f"====> Intent Label: {intent_label}")
    # Try both invocation methods for compatibility
    try:
        results = rag_system.query(feedback)
        user_input = results.get('question')
        rag_response = results.get('result', 'No response generated.')
    except Exception as e:
        print(f"Error processing feedback: {str(e)}")
        results = "I couldn't process your request at the moment. Please try again later."
    print("" * 50)
    print(f"===> User Input: {user_input}")
    print(f"===> Response: {rag_response}")
    print("=" * 50)
    return results, intent_label

# 4 user feedbacks to test feedback processing
# These feedbacks are examples and can be replaced with real user inputs.
feedbacks = [
    "I don't know what happened, with my account, I can't log in anymore. even after resetting my password.",
    "I can’t log in to my account, it keeps saying my password is incorrect.",
    "Unable to access my account—even after resetting my password.",
    "I need help with my account, I already reset my password but still can't log in.",
]
admin_email = "dev@mywebsite.com"
for feedback in feedbacks:
    # Get the response and intent
    response, label = process_feedback(feedback, admin_email)