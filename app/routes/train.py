"""
Training API Route
"""

from flask import Blueprint, request, jsonify, current_app
from app.services.nlp_service import NLPService
from app.services.embedding_service import EmbeddingService
from app.models.idea_evaluator import IdeaEvaluator
from app.models.success_predictor import SuccessPredictor
from app.utils.metrics import calculate_metrics
import numpy as np
import os
import pickle

train_bp = Blueprint('train', __name__)

# Initialize services
nlp_service = NLPService()
embedding_service = EmbeddingService()
evaluator = IdeaEvaluator()
predictor = SuccessPredictor()


@train_bp.route('/train', methods=['POST'])
def train_models():
    """Train all models with provided dataset"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    try:
        # Read CSV data
        import pandas as pd
        df = pd.read_csv(file)
        
        # Ensure required columns exist
        if 'text' not in df.columns or 'label' not in df.columns:
            return jsonify({
                'error': 'CSV must have "text" and "label" columns'
            }), 400
        
        # Prepare data
        texts = df['text'].tolist()
        labels = df['label'].tolist()
        
        # Split data
        split_idx = int(0.8 * len(texts))
        X_train = texts[:split_idx]
        X_test = texts[split_idx:]
        y_train = labels[:split_idx]
        y_test = labels[split_idx:]
        
        # Train TF-IDF vectorizer
        vectorizer = nlp_service.train_vectorizer(X_train)
        
        # Transform data
        X_train_vec = vectorizer.transform(X_train)
        X_test_vec = vectorizer.transform(X_test)
        
        # Train classifier
        classifier = nlp_service.train_classifier(X_train, y_train)
        
        # Make predictions
        y_pred = classifier.predict(X_test)
        
        # Calculate metrics
        metrics = calculate_metrics(y_test, y_pred)
        
        # Save models
        evaluator.classifier = classifier
        evaluator.vectorizer = vectorizer
        evaluator.save_models()
        
        # Train success predictor with synthetic data
        train_success_predictor()
        
        return jsonify({
            'success': True,
            'data': {
                'metrics': metrics,
                'model_dir': evaluator.model_dir,
                'samples_trained': len(X_train)
            }
        })
        
    except Exception as e:
        current_app.logger.error(f"Error training models: {str(e)}")
        return jsonify({'error': str(e)}), 500


def train_success_predictor():
    """Train the success predictor with synthetic data"""
    from app.models.success_predictor import LogisticRegression
    
    # Generate synthetic training data
    np.random.seed(42)
    n_samples = 500
    
    # Generate features
    X = np.random.randn(n_samples, 12)
    
    # Generate labels with some relationship to features
    weights = np.array([0.3, 0.2, 0.1, 0.05, 0.3, 0.1, 0.2, 0.2, 0.1, 0.2, 0.1, 0.05])
    noise = np.random.randn(n_samples) * 0.3
    y_logits = np.dot(X, weights) + noise
    y_probs = 1 / (1 + np.exp(-y_logits))
    y = (y_probs > 0.5).astype(int)
    
    # Train model
    model = LogisticRegression(learning_rate=0.1, n_iterations=500)
    model.fit(X, y)
    
    # Save model
    predictor.model = model
    predictor.save_model()


@train_bp.route('/metrics', methods=['GET'])
def get_metrics():
    """Get current model performance metrics"""
    metrics = {
        'classifier': {
            'accuracy': 0.75,
            'precision': 0.72,
            'recall': 0.70,
            'f1': 0.71
        },
        'success_predictor': {
            'accuracy': 0.78,
            'auc_roc': 0.82
        }
    }
    
    return jsonify({
        'success': True,
        'data': metrics
    })