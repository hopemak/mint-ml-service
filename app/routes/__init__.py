"""
API Routes Package
"""

from .evaluate import evaluate_bp
from .predict import predict_bp
from .match import match_bp
from .duplicate import duplicate_bp
from .rag_chat import chat_bp
from .train import train_bp

__all__ = [
    'evaluate_bp',
    'predict_bp',
    'match_bp',
    'duplicate_bp',
    'chat_bp',
    'train_bp'
]

from app.routes.chat_groq import chat_bp
