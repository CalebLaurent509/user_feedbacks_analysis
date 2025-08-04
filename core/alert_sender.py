"""
Alert System Module for User Feedback Analysis

This module handles automated email notifications for:
- Intent promotion alerts when emerging patterns reach thresholds
- Feedback trend summaries for administrative oversight
- System status and error notifications

Uses SMTP with Gmail configuration by default.
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from email.mime.text import MIMEText
import smtplib
import dotenv

# Load environment variables from .env file
dotenv.load_dotenv()

def send_alert_email(feedback, intent, to_email, summary=None):
    """
    Send an alert email notification for promoted intents.
    
    This function sends automated email alerts to administrators when
    an emerging intent pattern is promoted to an official intent category.
    
    Args:
        feedback (list): List of feedback messages that triggered the promotion
        intent (str): The intent category that was promoted
        to_email (str): Recipient email address for the alert
        summary (str, optional): Generated summary of the feedback trend
        
    Raises:
        Exception: If email sending fails due to SMTP or authentication errors
    """
    # SMTP configuration for Gmail
    smtp_host = "smtp.gmail.com"
    smtp_port = 587
    
    # Get SMTP credentials from environment variables
    smtp_user = os.getenv("SMTP_USER")
    smtp_pass = os.getenv("SMTP_PASS")
    to_email = os.getenv("TO_EMAIL")

    # Create email content
    subject = "Alert: A New Intent Was Promoted!"
    body = f"""
    Intent: {intent}
    
    Details:
        {summary if summary else 'No summary available.'}
    
    Recent Feedback Examples:
    {chr(10).join([f"- {fb}" for fb in feedback[:3]])}
    
    This intent has reached the promotion threshold and is now an official category.
    Please review and consider any necessary actions.
    
    ---
    User Feedback Analysis System
    """
    
    # Create email message
    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = smtp_user
    msg["To"] = to_email

    try:
        # Send email using SMTP
        with smtplib.SMTP(smtp_host, smtp_port) as server:
            server.starttls()  # Enable encryption
            server.login(smtp_user, smtp_pass)
            server.send_message(msg)
        print(f"Alert email sent successfully to {to_email}")
    except Exception as e:
        print(f"Error sending email alert: {e}")
        # Log the error but don't crash the application
        import logging
        logging.error(f"Failed to send alert email: {e}")

def send_system_notification(message, subject="System Notification", to_email=None):
    """
    Send a general system notification email.
    
    Used for system health alerts, errors, or other administrative
    notifications that don't fit the intent promotion pattern.
    
    Args:
        message (str): The notification message content
        subject (str): Email subject line
        to_email (str, optional): Override recipient email
    """
    if not to_email:
        to_email = os.getenv("TO_EMAIL")
    
    smtp_host = "smtp.gmail.com"
    smtp_port = 587
    smtp_user = os.getenv("SMTP_USER")
    smtp_pass = os.getenv("SMTP_PASS")

    body = f"""
    {message}
    
    ---
    User Feedback Analysis System
    Timestamp: {__import__('datetime').datetime.now().isoformat()}
    """
    
    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = smtp_user
    msg["To"] = to_email

    try:
        with smtplib.SMTP(smtp_host, smtp_port) as server:
            server.starttls()
            server.login(smtp_user, smtp_pass)
            server.send_message(msg)
        print(f"System notification sent to {to_email}")
    except Exception as e:
        print(f"Error sending system notification: {e}")
        import logging
        logging.error(f"Failed to send system notification: {e}")