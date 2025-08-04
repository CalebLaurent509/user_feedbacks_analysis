"""
Data Management Module for User Feedback Analysis System

This module provides persistent storage and retrieval functions for:
- Intent labels and categories
- Intent occurrence statistics  
- Feedback data by intent
- Emerging intent patterns

All data is stored in JSON format for easy inspection and modification.
"""

import json

# Default file paths for data persistence
base_intentions_file = "base_intentions.json"
intent_stats_file = "intent_stats.json"
emerging_intents_file = "emerging_intents.json"
intent_feedbacks_file = "intent_feedbacks.json"


def save_intents(intent_labels, intent_file=base_intentions_file):
    """
    Save official intent labels to a JSON file.
    
    This function persists the current list of official intent categories
    that are used for classification.
    
    Args:
        intent_labels (list): List of official intent category strings
        intent_file (str): Path to the JSON file for storage
    """
    with open(intent_file, "w") as f:
        json.dump(intent_labels, f, indent=2)

def save_intent_counts(intent_count, count_file=intent_stats_file):
    """
    Save intent occurrence counters to a JSON file.
    
    Stores the frequency of each intent occurrence for tracking
    emerging patterns and promotion thresholds.
    
    Args:
        intent_count (dict): Dictionary mapping intent names to occurrence counts
        count_file (str): Path to the JSON file for storage
    """
    with open(count_file, "w") as f:
        json.dump(dict(intent_count), f, indent=2)

def load_intent_counts(intent_count, count_file=intent_stats_file):
    """
    Load intent occurrence counters from a JSON file.
    
    Restores previously saved intent occurrence statistics to maintain
    state across application restarts.
    
    Args:
        intent_count (dict): Dictionary to populate with loaded data
        count_file (str): Path to the JSON file containing saved data
    """
    try:
        with open(count_file, "r") as f:
            data = json.load(f)
            for k, v in data.items():
                intent_count[k] = v
    except FileNotFoundError:
        # File doesn't exist yet - this is normal for first run
        pass

def save_emerging_intents(emerging_intents, filepath=emerging_intents_file):
    """
    Save emerging intent patterns to a JSON file.
    
    Stores detected emerging intents that haven't reached promotion
    threshold yet for analysis and tracking.
    
    Args:
        emerging_intents (list): List of emerging intent data structures
        filepath (str): Path to the JSON file for storage
    """
    with open(filepath, "w") as f:
        json.dump(emerging_intents, f, indent=2)

def save_intent_feedbacks(intent_feedbacks, filepath=intent_feedbacks_file):
    """
    Save intent-grouped feedback data to a JSON file.
    
    Persists feedback messages organized by their classified intent
    for analysis and summarization purposes.
    
    Args:
        intent_feedbacks (dict): Dictionary mapping intents to feedback lists
        filepath (str): Path to the JSON file for storage
    """
    with open(filepath, "w") as f:
        json.dump(intent_feedbacks, f, indent=2)

def load_intent_feedbacks(intent_feedbacks, filepath=intent_feedbacks_file):
    """
    Load intent-grouped feedback data from a JSON file.
    
    Restores previously saved feedback data organized by intent
    to maintain historical context across application restarts.
    
    Args:
        intent_feedbacks (dict): Dictionary to populate with loaded feedback data
        filepath (str): Path to the JSON file containing saved data
    """
    try:
        with open(filepath, "r") as f:
            data = json.load(f)
            for k, v in data.items():
                intent_feedbacks[k] = v
    except FileNotFoundError:
        # File doesn't exist yet - this is normal for first run
        pass