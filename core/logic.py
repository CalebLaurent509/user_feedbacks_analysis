"""
Core Logic Module for Intent Classification and Meaning Analysis

This module implements the MeaningEngine class which handles:
- Intent classification using similarity matching
- Emerging intent detection and promotion
- Feedback summarization and trend analysis
- Integration with external systems for alerts
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import json
from collections import defaultdict
import torch.nn.functional as F
from models.model import load_model, embeddings, intent_classifier, util
from core.alert_sender import send_alert_email
from data.data_manager import (
    save_intents,
    save_intent_counts,
    load_intent_counts,
    save_intent_feedbacks,
    load_intent_feedbacks
)

# Load the LLM model for text processing
llm = load_model()

def summarize_feedbacks(feedbacks):
    """
    Generate a comprehensive summary of multiple feedback entries with the same intent.
    
    This function uses the LLM to analyze and synthesize feedback trends,
    providing insights into common patterns and user concerns.
    
    Args:
        feedbacks (list): List of feedback strings expressing similar intentions
        
    Returns:
        str: A 5-10 sentence summary synthesizing the feedback trends
    """
    prompt = (
        "Here are some user feedbacks expressing the same intention:\n\n"
        + "\n".join([f"- {fb}" for fb in feedbacks])
        + "\n\nPlease provide a global summary in 5 or 10 sentences, that synthesizes the trend."
    )
    try:
        response = llm.invoke(prompt)
        return response.content
    except Exception as e:
        print(f"[Error OpenAI summary] {e}")
        return "Summary not available."

class MeaningEngine:
    """
    Advanced intent classification and meaning analysis engine.
    
    The MeaningEngine processes user feedback to classify intents, detect emerging
    patterns, and maintain persistent statistics. It combines similarity matching
    with dynamic intent discovery and automated alert systems.
    
    Attributes:
        alpha (float): Weight for logical implication score in coherence calculation
        intent_file (str): Path to base intentions JSON file
        count_file (str): Path to intent statistics JSON file
        threshold_promotion (int): Minimum occurrences to promote emerging intent
        encoder: Sentence transformer model for embeddings
        intent_classifier: Zero-shot classification model
        embedding_cache (dict): Cache for computed embeddings
        emerging_intents (list): List of detected emerging intents
        intent_feedbacks (defaultdict): Storage for feedback by intent
        intent_labels (list): List of official intent categories
        intent_embeddings: Precomputed embeddings for intent labels
        intent_count (defaultdict): Persistent counters for intent occurrences
    """
    
    def __init__(self, intent_file="data/base_intentions.json", alpha=0.4, 
                 count_file="data/intent_stats.json", threshold_promotion=3):
        """
        Initialize the meaning engine with configuration and data loading.
        
        Args:
            intent_file (str): Path to the JSON file containing base intentions
            alpha (float): Weight for logical implication score in coherence
            count_file (str): Path to the JSON file containing intent statistics
            threshold_promotion (int): Minimum occurrences to promote emerging intent
        """
        self.alpha = alpha
        self.intent_file = intent_file
        self.count_file = count_file
        self.threshold_promotion = threshold_promotion
        self.encoder = embeddings
        self.intent_classifier = intent_classifier
        self.embedding_cache = {}
        self.emerging_intents = []
        
        # Initialize intent feedbacks storage
        self.intent_feedbacks = defaultdict(list)
        load_intent_feedbacks(intent_feedbacks=self.intent_feedbacks)
        
        # Load base intentions from JSON file
        with open(self.intent_file, "r") as f:
            self.intent_labels = json.load(f)
        
        # Precompute embeddings for all intent labels
        self.intent_embeddings = self.encoder.encode(
            self.intent_labels, 
            convert_to_tensor=True, 
            normalize_embeddings=True
        )
        
        # Load persistent intent counters
        self.intent_count = defaultdict(int)
        load_intent_counts(intent_count=self.intent_count, count_file=self.count_file)

    def simplify_intention(self, text):
        """
        Use GPT-3.5-turbo to simplify a complex sentence into a general intention.
        
        This method extracts the core intent from complex or verbose feedback,
        making it easier to classify and process.
        
        Args:
            text (str): Text to simplify into intent format
            
        Returns:
            str: Simplified intention in 5 words or less
        """
        try:
            response = llm.invoke(
                "ROLE: You are an assistant that extracts the intent of a user sentence. "
                "IMPORTANT: Return only the intent, in 5 words or less.\n"
                f"Extract the intent from: {text}"
            )
            return response.content
        except Exception as e:
            print(f"[OpenAI Error] {e}")
            return "UnknownIntent"

    def get_embedding(self, text):
        """
        Generate or retrieve cached embedding for given text.
        
        Uses caching to improve performance for repeated text processing.
        
        Args:
            text (str): Text to generate embedding for
            
        Returns:
            torch.Tensor: Normalized embedding vector
        """
        if text not in self.embedding_cache:
            with torch.no_grad():
                self.embedding_cache[text] = self.encoder.encode(text, convert_to_tensor=True)
        return self.embedding_cache[text]

    def intent(self, text, top_k=3, threshold=0.5):
        """
        Identify the closest intentions to a given text using similarity matching.
        
        Computes cosine similarity between input text and known intent embeddings
        to find the best matching intent categories.
        
        Args:
            text (str): Text to analyze for intent classification
            top_k (int): Maximum number of intentions to return
            threshold (float): Minimum score to consider an intention as valid
            
        Returns:
            list: List of tuples containing (intent_label, confidence_score)
        """
        query_embedding = self.encoder.encode(text, convert_to_tensor=True, normalize_embeddings=True)
        scores = util.pytorch_cos_sim(query_embedding, self.intent_embeddings)[0]
        top_results = torch.topk(scores, k=top_k)
        labels = [self.intent_labels[i] for i in top_results.indices]
        confidences = [float(scores[i]) for i in top_results.indices]

        if confidences[0] < threshold:
            return [("EmergingIntent", confidences[0])]
        return list(zip(labels, confidences))

    def predict_intent(self, text, threshold=0.7, top_k=3, admin_email=None):
        """
        Predict intent for given text with emerging intent detection and handling.
        
        This method performs the complete intent prediction workflow:
        1. Initial intent classification
        2. Emerging intent detection and simplification
        3. Similar intent merging
        4. Intent promotion when threshold is reached
        
        Args:
            text (str): Input text for intent prediction
            threshold (float): Confidence threshold for known intents
            top_k (int): Number of top intents to consider
            admin_email (str): Email for admin notifications
            
        Returns:
            tuple: (intent_type, intent_label, confidence_score)
        """
        intent = self.intent(text, top_k=top_k, threshold=threshold)[0]

        if "EmergingIntent" in intent[0]:
            print(f"Emerging intent detected: '{text}' (score: {intent[1]:.2f})")
            simplified = self.simplify_intention(text)
            print(f"Simplified as → '{simplified}'")

            # Search for similar intent already in statistics
            closest_intent, score = self.find_similar_intent(simplified)

            if closest_intent:
                print(f"Merged with similar intent: '{closest_intent}' (score: {score:.2f})")

                # Merge WITHOUT DUPLICATES
                if simplified in self.intent_feedbacks and self.intent_feedbacks[simplified]:
                    for fb in self.intent_feedbacks[simplified]:
                        if fb not in self.intent_feedbacks[closest_intent]:
                            self.intent_feedbacks[closest_intent].append(fb)
                    self.intent_feedbacks[simplified] = []
                
                # Add current feedback ONLY if not already present
                if text not in self.intent_feedbacks[closest_intent]:
                    self.intent_feedbacks[closest_intent].append(text)
                
                # Remove simplified key if empty
                if not self.intent_feedbacks[simplified]:
                    del self.intent_feedbacks[simplified]
                
                save_intent_feedbacks(self.intent_feedbacks)
                self.intent_count[closest_intent] += 1
                save_intent_counts(self.intent_count)
                
                if self.intent_count[closest_intent] >= self.threshold_promotion:
                    self.promote_intent(closest_intent, admin_email)
            else:
                print(f"New simplified intent: '{simplified}'")
                self.intent_feedbacks[simplified].append(text)
                save_intent_feedbacks(self.intent_feedbacks)
                self.intent_count[simplified] += 1
                save_intent_counts(self.intent_count)
                
                if self.intent_count[simplified] >= self.threshold_promotion:
                    self.promote_intent(simplified, admin_email)
                else:
                    self.emerging_intents.append({"text": text, "simplified": simplified, "score": intent[1]})

            return ("EmergingIntent", simplified, intent[1])

        # Otherwise: classic intention already known
        return intent

    def find_similar_intent(self, new_intent, threshold=0.5):
        """
        Find an existing intent that is very similar to a new intent.
        
        Uses cosine similarity to identify if a new intent should be merged
        with an existing one rather than creating a duplicate category.
        
        Args:
            new_intent (str): New intent to compare against existing ones
            threshold (float): Minimum similarity score for merging
            
        Returns:
            tuple: (best_matching_intent, similarity_score) or (None, 0)
        """
        new_vec = self.get_embedding(new_intent)
        best_match = None
        best_score = 0

        for existing_intent in self.intent_count:
            existing_vec = self.get_embedding(existing_intent)
            score = float(util.cos_sim(new_vec, existing_vec)[0][0])
            if score > best_score:
                best_score = score
                best_match = existing_intent

        if best_score >= threshold:
            return best_match, best_score
        return None, 0

    def promote_intent(self, text, admin_email=None):
        """
        Promote an emerging intent to an official intent category.
        
        When an emerging intent reaches the promotion threshold, it becomes
        an official intent category. This triggers:
        1. Addition to official intent labels
        2. Embedding computation and storage
        3. Cleanup of temporary data
        4. Admin notification with summary
        
        Args:
            text (str): Text of the intent to promote
            admin_email (str): Email address for admin notification
        """
        if text not in self.intent_labels:
            print(f"=====> Intent Promoted: '{text}'")
            self.intent_labels.append(text)
            
            # Add encoded embedding
            new_embedding = self.encoder.encode([text], convert_to_tensor=True, normalize_embeddings=True)
            self.intent_embeddings = torch.cat([self.intent_embeddings, new_embedding])
            
            # Save to official intentions
            save_intents(intent_labels=self.intent_labels)
            
            # Remove from counter and save
            if text in self.intent_count:
                del self.intent_count[text]
                save_intent_counts(intent_count=self.intent_count)
            
            # Optional: also remove from emerging intents
            self.emerging_intents = [e for e in self.emerging_intents if e["text"] != text]
            
            # Get recent feedbacks for summary
            feedbacks = self.intent_feedbacks[text][-3:]  # Last 3 feedbacks
            print(f"=====> Feedbacks for promotion: {feedbacks} & {len(feedbacks)} total")
            
            # Generate summary for admin notification
            summary = summarize_feedbacks(feedbacks)
            print(f"=====> Intent promoted: {text} \n =====> With summary: {summary}")
            
            # Send admin alert if email provided
            if admin_email:
                send_alert_email(feedbacks, text, admin_email, summary)
            
            # Clean up feedbacks for this intent
            if text in self.intent_feedbacks:
                del self.intent_feedbacks[text]
                save_intent_feedbacks(intent_feedbacks=self.intent_feedbacks)
        """
        query_embedding = self.encoder.encode(text, convert_to_tensor=True, normalize_embeddings=True)
        scores = util.pytorch_cos_sim(query_embedding, self.intent_embeddings)[0]
        top_results = torch.topk(scores, k=top_k)
        labels = [self.intent_labels[i] for i in top_results.indices]
        confidences = [float(scores[i]) for i in top_results.indices]

        if confidences[0] < threshold:
            return [("EmergingIntent", confidences[0])]
        return list(zip(labels, confidences))

    def predict_intent(self, text, threshold=0.7, top_k=3, admin_email=None):
        intent = self.intent(text, top_k=top_k, threshold=threshold)[0]

        if "EmergingIntent" in intent[0]:
            print(f"Emerging intent detected: '{text}' (score: {intent[1]:.2f})")
            simplified = self.simplify_intention(text)
            print(f"Simplified as → '{simplified}'")

            # Recherche d’intention similaire déjà dans les stats
            closest_intent, score = self.find_similar_intent(simplified)

            if closest_intent:
                print(f"Merged with similar intent: '{closest_intent}' (score: {score:.2f})")

                # Fusionne SANS DOUBLON
                if simplified in self.intent_feedbacks and self.intent_feedbacks[simplified]:
                    for fb in self.intent_feedbacks[simplified]:
                        if fb not in self.intent_feedbacks[closest_intent]:
                            self.intent_feedbacks[closest_intent].append(fb)
                    self.intent_feedbacks[simplified] = []
                # Ajoute le feedback courant UNIQUEMENT s'il n'est pas déjà présent
                if text not in self.intent_feedbacks[closest_intent]:
                    self.intent_feedbacks[closest_intent].append(text)
                # Supprime la clé simplifiée si elle est vide
                if not self.intent_feedbacks[simplified]:
                    del self.intent_feedbacks[simplified]
                save_intent_feedbacks(self.intent_feedbacks)
                self.intent_count[closest_intent] += 1
                save_intent_counts(self.intent_count)
                if self.intent_count[closest_intent] >= self.threshold_promotion:
                    self.promote_intent(closest_intent, admin_email)
            else:
                print(f"New simplified intent: '{simplified}'")
                self.intent_feedbacks[simplified].append(text)
                save_intent_feedbacks(self.intent_feedbacks)
                self.intent_count[simplified] += 1
                save_intent_counts(self.intent_count)
                if self.intent_count[simplified] >= self.threshold_promotion:
                    self.promote_intent(simplified, admin_email)
                else:
                    self.emerging_intents.append({"text": text, "simplified": simplified, "score": intent[1]})

            return ("EmergingIntent", simplified, intent[1])

        # Sinon : intention classique déjà connue
        return intent

    # Trouver une intention déjà existante très similaire
    def find_similar_intent(self, new_intent, threshold=0.5):
        new_vec = self.get_embedding(new_intent)
        best_match = None
        best_score = 0

        for existing_intent in self.intent_count:
            existing_vec = self.get_embedding(existing_intent)
            score = float(util.cos_sim(new_vec, existing_vec)[0][0])
            if score > best_score:
                best_score = score
                best_match = existing_intent

        if best_score >= threshold:
            return best_match, best_score
        return None, 0

    def promote_intent(self, text, admin_email=None):
        """
        Promote an emerging intent to an official intent category.
        
        When an emerging intent reaches the promotion threshold, it becomes
        an official intent category. This triggers:
        1. Addition to official intent labels
        2. Embedding computation and storage
        3. Cleanup of temporary data
        4. Admin notification with summary
        
        Args:
            text (str): Text of the intent to promote
            admin_email (str): Email address for admin notification
        """
        if text not in self.intent_labels:
            print(f"=====> Intent Promoted: '{text}'")
            self.intent_labels.append(text)
            
            # Add encoded embedding
            new_embedding = self.encoder.encode([text], convert_to_tensor=True, normalize_embeddings=True)
            self.intent_embeddings = torch.cat([self.intent_embeddings, new_embedding])
            
            # Save to official intentions
            save_intents(intent_labels=self.intent_labels)
            
            # Remove from counter and save
            if text in self.intent_count:
                del self.intent_count[text]
                save_intent_counts(intent_count=self.intent_count)
            
            # Optional: also remove from emerging intents
            self.emerging_intents = [e for e in self.emerging_intents if e["text"] != text]
            
            # Get recent feedbacks for summary
            feedbacks = self.intent_feedbacks[text][-3:]  # Last 3 feedbacks
            print(f"=====> Feedbacks for promotion: {feedbacks} & {len(feedbacks)} total")
            
            # Generate summary for admin notification
            summary = summarize_feedbacks(feedbacks)
            print(f"=====> Intent promoted: {text} \n =====> With summary: {summary}")
            
            # Send admin alert if email provided
            if admin_email:
                send_alert_email(feedbacks, text, admin_email, summary)
            
            # Clean up feedbacks for this intent
            if text in self.intent_feedbacks:
                del self.intent_feedbacks[text]
                save_intent_feedbacks(intent_feedbacks=self.intent_feedbacks)