from flask import Flask
from flask_cors import CORS
import os
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_app():
    """Create and configure the Flask application"""
    app = Flask(__name__)
    
    # CORS configuration
    CORS(app, origins=["http://localhost:5173", "http://localhost:5174", "http://localhost:5001", "http://127.0.0.1:5001", "*"])
    
    # Register blueprints
    try:
        from app.routes.chat_groq import chat_bp
        app.register_blueprint(chat_bp)
        logger.info("✅ Chat blueprint registered successfully")
    except ImportError as e:
        logger.warning(f"⚠️ Could not import chat blueprint: {e}")
    
    try:
        from app.routes.evaluate import evaluate_bp
        app.register_blueprint(evaluate_bp)
        logger.info("✅ Evaluation blueprint registered successfully")
    except ImportError as e:
        logger.warning(f"⚠️ Could not import evaluation blueprint: {e}")
    
    # Health check endpoint
    @app.route('/api/health', methods=['GET'])
    def health():
        return {'status': 'OK', 'message': 'ML service is running!', 'timestamp': datetime.utcnow().isoformat()}
    
    logger.info("🚀 ML AI Service initialized successfully")
    return app
