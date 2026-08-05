"""
Mentor Matching API Route
"""

from flask import Blueprint, request, jsonify, current_app
from app.models.mentor_matcher import MentorMatcher

match_bp = Blueprint('match', __name__)

# Initialize matcher
matcher = MentorMatcher()


@match_bp.route('/match', methods=['POST'])
def match_mentors():
    """Match mentors for a startup"""
    data = request.get_json()
    
    if not data:
        return jsonify({'error': 'No data provided'}), 400
    
    idea_id = data.get('idea_id', 'unknown')
    required_skills = data.get('required_skills', [])
    startup_data = data.get('startup_data', {})
    
    if not required_skills and not startup_data:
        return jsonify({'error': 'Required skills or startup data needed'}), 400
    
    try:
        matches = matcher.match(
            idea_id=idea_id,
            required_skills=required_skills,
            startup_data=startup_data
        )
        
        current_app.logger.info(
            f"Mentor matching for idea {idea_id}: {len(matches)} matches found"
        )
        
        return jsonify({
            'success': True,
            'data': {
                'idea_id': idea_id,
                'matches': matches[:10]  # Return top 10 matches
            }
        })
        
    except Exception as e:
        current_app.logger.error(f"Error matching mentors: {str(e)}")
        return jsonify({'error': str(e)}), 500


@match_bp.route('/mentors', methods=['GET'])
def list_mentors():
    """List available mentors"""
    mentors = matcher.mentor_profiles
    return jsonify({
        'success': True,
        'data': {
            'mentors': [
                {
                    'id': m['id'],
                    'name': m['name'],
                    'title': m['title'],
                    'expertise': m['expertise'],
                    'available': m.get('available', True),
                    'rating': m.get('rating', 0)
                }
                for m in mentors
            ]
        }
    })