"""
Duplicate Detection API Route
"""

from flask import Blueprint, request, jsonify, current_app
from app.models.duplicate_detector import DuplicateDetector

duplicate_bp = Blueprint('duplicate', __name__)

# Initialize detector
detector = DuplicateDetector()

# Seed with some sample ideas
sample_ideas = [
    {'id': 'idea_001', 'text': 'AI-powered healthcare platform for remote patient monitoring using wearable devices'},
    {'id': 'idea_002', 'text': 'Blockchain-based supply chain tracking system with smart contracts'},
    {'id': 'idea_003', 'text': 'Sustainable packaging solution using biodegradable materials from agricultural waste'},
    {'id': 'idea_004', 'text': 'EdTech platform for personalized learning using AI and adaptive assessments'},
    {'id': 'idea_005', 'text': 'Renewable energy storage system using advanced battery technology'},
]
detector.load_corpus(sample_ideas)


@duplicate_bp.route('/duplicate', methods=['POST'])
def check_duplicate():
    """Check if an idea is a duplicate"""
    data = request.get_json()
    
    if not data:
        return jsonify({'error': 'No data provided'}), 400
    
    text = data.get('text', '')
    
    if not text:
        return jsonify({'error': 'Text is required'}), 400
    
    try:
        result = detector.detect(text)
        
        current_app.logger.info(
            f"Duplicate detection: {len(text)} chars | Duplicate: {result['is_duplicate']}"
        )
        
        return jsonify({
            'success': True,
            'data': result
        })
        
    except Exception as e:
        current_app.logger.error(f"Error detecting duplicate: {str(e)}")
        return jsonify({'error': str(e)}), 500


@duplicate_bp.route('/duplicate/corpus', methods=['POST'])
def add_to_corpus():
    """Add an idea to the corpus"""
    data = request.get_json()
    
    if not data:
        return jsonify({'error': 'No data provided'}), 400
    
    idea_id = data.get('id', '')
    text = data.get('text', '')
    
    if not idea_id or not text:
        return jsonify({'error': 'ID and text are required'}), 400
    
    detector.add_idea(idea_id, text)
    
    return jsonify({
        'success': True,
        'message': 'Idea added to corpus'
    })