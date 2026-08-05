"""
RAG Chat API Route
"""

from flask import Blueprint, request, jsonify, current_app
from app.services.rag_service import RAGService

chat_bp = Blueprint('chat', __name__)

# Initialize RAG service
rag_service = RAGService()


@chat_bp.route('/chat', methods=['POST'])
def chat():
    """Handle chat requests with RAG"""
    data = request.get_json()
    
    if not data:
        return jsonify({'error': 'No data provided'}), 400
    
    query = data.get('query', '')
    user_id = data.get('user_id', 'anonymous')
    context = data.get('context', {})
    
    if not query:
        return jsonify({'error': 'Query is required'}), 400
    
    try:
        response = rag_service.generate_response(query, user_id, context)
        
        current_app.logger.info(
            f"RAG chat: {user_id} | Query: {query[:50]}..."
        )
        
        return jsonify({
            'success': True,
            'data': response
        })
        
    except Exception as e:
        current_app.logger.error(f"Error in RAG chat: {str(e)}")
        return jsonify({'error': str(e)}), 500


@chat_bp.route('/chat/contexts', methods=['GET'])
def get_contexts():
    """Get available contexts"""
    contexts = [
        {'id': 'general', 'name': 'General Innovation'},
        {'id': 'startup', 'name': 'Startup Guidance'},
        {'id': 'funding', 'name': 'Funding & Grants'},
        {'id': 'mentorship', 'name': 'Mentorship'},
        {'id': 'market', 'name': 'Market Strategy'}
    ]
    
    return jsonify({
        'success': True,
        'data': {'contexts': contexts}
    })