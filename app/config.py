"""
Configuration for ML AI Service
"""

import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    """Base configuration"""
    SECRET_KEY = os.environ.get('SECRET_KEY', 'dev-secret-key')
    MODEL_DIR = os.environ.get('MODEL_DIR', './data/models')
    KNOWLEDGE_BASE_DIR = os.environ.get('KNOWLEDGE_BASE_DIR', './data/knowledge')
    LOG_LEVEL = os.environ.get('LOG_LEVEL', 'INFO')
    
    # Model settings
    TFIDF_MAX_FEATURES = 10000
    N_GRAM_RANGE = (1, 3)
    
    # Evaluation weights
    WEIGHTS = {
        'innovation': 0.25,
        'feasibility': 0.20,
        'market_potential': 0.20,
        'social_impact': 0.15,
        'financial_viability': 0.20
    }
    
    # Success prediction thresholds
    SUCCESS_THRESHOLD = 0.5
    HIGH_SUCCESS_THRESHOLD = 0.7
    
    # RAG settings
    RAG_MAX_DOCS = 5
    RAG_SCORE_THRESHOLD = 0.3


class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True
    LOG_LEVEL = 'DEBUG'


class ProductionConfig(Config):
    """Production configuration"""
    DEBUG = False
    LOG_LEVEL = 'INFO'


class TestingConfig(Config):
    """Testing configuration"""
    TESTING = True
    LOG_LEVEL = 'DEBUG'
