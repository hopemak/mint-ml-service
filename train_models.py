"""
Initial Training Script for ML Models
Run this script to train models with sample data
"""

import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pickle
import json
from app.models.idea_evaluator import TFIDFVectorizer, NaiveBayesClassifier
from app.models.success_predictor import LogisticRegression


def generate_sample_data():
    """Generate sample training data"""
    # Sample innovation ideas with categories
    samples = [
        # AI/ML
        ("AI-powered healthcare diagnostics using deep learning", "ai_ml"),
        ("Machine learning for fraud detection in banking", "ai_ml"),
        ("Natural language processing for customer service", "ai_ml"),
        ("Computer vision for autonomous vehicles", "ai_ml"),
        
        # Biotech
        ("Gene editing for rare disease treatment", "biotech"),
        ("Sustainable biofuel from algae", "biotech"),
        ("Personalized medicine using genomics", "biotech"),
        ("Biodegradable materials from agricultural waste", "biotech"),
        
        # Fintech
        ("Blockchain for cross-border payments", "fintech"),
        ("AI-driven investment portfolio management", "fintech"),
        ("Digital identity verification using biometrics", "fintech"),
        ("Peer-to-peer lending platform", "fintech"),
        
        # CleanTech
        ("Solar panel efficiency improvement", "cleantech"),
        ("Ocean plastic cleanup system", "cleantech"),
        ("Smart grid energy optimization", "cleantech"),
        ("Carbon capture technology", "cleantech"),
        
        # EdTech
        ("Adaptive learning platform for students", "edtech"),
        ("Virtual reality for education", "edtech"),
        ("AI tutor for personalized learning", "edtech"),
        ("Gamified language learning app", "edtech"),
        
        # General Tech
        ("Cloud-based collaboration platform", "general"),
        ("IoT smart home system", "general"),
        ("Cybersecurity threat detection", "general"),
        ("Edge computing for IoT", "general"),
    ]
    
    texts = [s[0] for s in samples]
    labels = [s[1] for s in samples]
    
    return texts, labels


def train_idea_classifier():
    """Train the idea classifier"""
    print("Training idea classifier...")
    
    texts, labels = generate_sample_data()
    
    # Train vectorizer
    vectorizer = TFIDFVectorizer(max_features=1000, ngram_range=(1, 3))
    vectorizer.fit(texts)
    
    # Train classifier
    classifier = NaiveBayesClassifier(alpha=1.0)
    classifier.fit(texts, labels)
    
    # Save models
    model_dir = './data/models'
    os.makedirs(model_dir, exist_ok=True)
    
    with open(os.path.join(model_dir, 'vectorizer.pkl'), 'wb') as f:
        pickle.dump(vectorizer, f)
    
    with open(os.path.join(model_dir, 'classifier.pkl'), 'wb') as f:
        pickle.dump(classifier, f)
    
    print(f"Classifier trained on {len(texts)} samples")
    
    return vectorizer, classifier


def train_success_predictor():
    """Train the success predictor"""
    print("Training success predictor...")
    
    # Generate synthetic data
    np.random.seed(42)
    n_samples = 500
    
    # Generate features (12 features)
    X = np.random.randn(n_samples, 12)
    
    # Generate labels with weights
    weights = np.array([0.3, 0.2, 0.1, 0.05, 0.3, 0.1, 0.2, 0.2, 0.1, 0.2, 0.1, 0.05])
    noise = np.random.randn(n_samples) * 0.3
    y_logits = np.dot(X, weights) + noise
    y_probs = 1 / (1 + np.exp(-y_logits))
    y = (y_probs > 0.5).astype(int)
    
    # Train model
    model = LogisticRegression(learning_rate=0.1, n_iterations=500)
    model.fit(X, y)
    
    # Save model
    model_dir = './data/models'
    os.makedirs(model_dir, exist_ok=True)
    
    with open(os.path.join(model_dir, 'success_predictor.pkl'), 'wb') as f:
        pickle.dump(model, f)
    
    print(f"Success predictor trained on {n_samples} samples")
    
    return model


def train_mentor_matcher():
    """Train the mentor matcher"""
    print("Training mentor matcher...")
    # Mentor matcher doesn't require training, it uses pre-defined profiles
    # and skill embeddings which are initialized in the class
    print("Mentor matcher ready (no training required)")
    return None


def main():
    """Main training function"""
    print("=" * 60)
    print("Training ML Models for Innovation Incubator Platform")
    print("=" * 60)
    
    # Create directories
    os.makedirs('./data/models', exist_ok=True)
    os.makedirs('./data/knowledge', exist_ok=True)
    os.makedirs('./data/raw', exist_ok=True)
    os.makedirs('./data/processed', exist_ok=True)
    
    # Train models
    try:
        train_idea_classifier()
        train_success_predictor()
        train_mentor_matcher()
        
        print("\n" + "=" * 60)
        print("Training completed successfully!")
        print(f"Models saved to: {os.path.abspath('./data/models')}")
        print("=" * 60)
        
    except Exception as e:
        print(f"\nError during training: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()